from PIL import Image, ImageDraw
import math
import os
import copy


class BinaryPose:
    prev_x0, prev_y0 = None, None
    prev_x1, prev_y1 = None, None
    neck_dist = None

    @classmethod # use BinaryPose.normalize(<keypoints>)
    def normalize(cls, orig_keypoints, copy=True): # returns normalized pose keypoints (returns none if cannot be normalized) (copies dictionary by default)
        # print("Normalize Method: ")
        # print("\tprev x0: " , cls.prev_x0," prev y0: " , cls.prev_y0," prev x1: " , cls.prev_x1," prev y1: " , cls.prev_y1,)

        if copy:
            kp = copy.deepcopy(orig_keypoints['keypoints']) # copy the keypoints so that orig values won't be affected
        else:
            kp = orig_keypoints['keypoints']

        # Normalize keypoints based on neck & lumbar
        x0, y0 = kp[0]['x'], kp[0]['y']
        x1, y1 = kp[1]['x'], kp[1]['y']

        prev = False

        if x0 is None or x1 is None:
            if cls.prev_x0 is None or cls.prev_x1 is None: return None # IF AT LEAST ONE PREV KEYPOINT IS MISSING DO NOT CREATE IMAGE
            x0, y0 = cls.prev_x0, cls.prev_y0
            x1, y1 = cls.prev_x1, cls.prev_y1
            prev = True

        cls.prev_x0 = x0
        cls.prev_y0 = y0
        cls.prev_x1 = x1
        cls.prev_y1 = y1

        cls.neck_dist = math.sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2))
        print("NECK NORMALIZATION", cls.neck_dist)

        for i in kp:
            if i['x'] is None or i['y'] is None: continue
            i['x'] = (i['x'] - x0) / cls.neck_dist
            i['y'] = (i['y'] - y0) / cls.neck_dist
        
        return kp

    @classmethod
    def createBinaryPose(cls, orig_keypoints, frame_number, folder_path):
        keypoints = copy.deepcopy(orig_keypoints) 
        kp = cls.normalize(keypoints, copy=False) # no need to copy because we already copied
        
        if kp is None: return None, None

        # create image PIL 
        image = Image.new('1', (512, 512), 0) # binary, size, background
        
        # draw object from PIL
        draw = ImageDraw.Draw(image)
        
        # white line
        line_color = 1
        line_thickness = 4
        
        # stickman scale
        scale = 2 * cls.neck_dist # will experiment more on this

        # custom function to check if keypoint is missing
        def draw_line(x1,y1,x2,y2):
            if not (x1 is None or x2 is None or y1 is None or y2 is None):
                origin_x, origin_y = 255, 255
                draw.line(
                        (origin_x + x1 * scale, origin_y + y1 * scale,
                        origin_x + x2 * scale, origin_y + y2 * scale), 
                        fill=line_color, width=line_thickness)
                draw.ellipse([(origin_x + x1 * scale-3,origin_y + y1 * scale-3),(origin_x + x1 * scale+3,origin_y + y1 * scale+3)], fill=line_color, width=1)
                draw.ellipse([(origin_x + x2 * scale-3,origin_y + y2 * scale-3),(origin_x + x2 * scale+3,origin_y + y2 * scale+3)], fill=line_color, width=1)
                
                
        # nose to neck
        draw_line(kp[0]['x'], kp[0]['y'], kp[1]['x'], kp[1]['y']) 
        # left arm
        draw_line(kp[5]['x'], kp[5]['y'], kp[1]['x'], kp[1]['y']) 
        draw_line(kp[5]['x'], kp[5]['y'], kp[6]['x'], kp[6]['y'])
        draw_line(kp[7]['x'], kp[7]['y'], kp[6]['x'], kp[6]['y'])
        # right arm
        draw_line(kp[2]['x'], kp[2]['y'], kp[1]['x'], kp[1]['y'])
        draw_line(kp[2]['x'], kp[2]['y'], kp[3]['x'], kp[3]['y'])
        draw_line(kp[4]['x'], kp[4]['y'], kp[3]['x'], kp[3]['y'])
        # left leg
        draw_line(kp[8]['x'], kp[8]['y'], kp[1]['x'], kp[1]['y'])
        draw_line(kp[8]['x'], kp[8]['y'], kp[9]['x'], kp[9]['y'])
        draw_line(kp[10]['x'], kp[10]['y'], kp[9]['x'], kp[9]['y'])
        # right leg
        draw_line(kp[11]['x'], kp[11]['y'], kp[1]['x'], kp[1]['y'])
        draw_line(kp[11]['x'], kp[11]['y'], kp[12]['x'], kp[12]['y'])
        draw_line(kp[13]['x'], kp[13]['y'], kp[12]['x'], kp[12]['y'])
        # left face
        draw_line(kp[0]['x'], kp[0]['y'], kp[14]['x'], kp[14]['y'])
        draw_line(kp[16]['x'], kp[16]['y'], kp[14]['x'], kp[14]['y'])
        # right face
        draw_line(kp[0]['x'], kp[0]['y'], kp[16]['x'], kp[16]['y'])
        draw_line(kp[17]['x'], kp[17]['y'], kp[16]['x'], kp[16]['y'])

        # File Path
        folder_path = f'{folder_path}{keypoints["person_id"]}/'
        file_name = f'{folder_path}pose_{frame_number}.png'

        # Check Directory
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save Image
        image.save(file_name)

        # Print Log
        print(f'Binary Pose Image Save in: {file_name}')

        keypoints['keypoints'] = kp # save the normalized pose keypoints
        return keypoints, file_name
