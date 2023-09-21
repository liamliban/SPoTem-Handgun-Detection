from PIL import Image, ImageDraw
import math
import os

def create_binary_pose(keypoints, frame_number, folder_name):
    kp = keypoints['keypoints']

    # Normalize keypoints
    x0, y0 = kp[0]['x'], kp[0]['y']
    neck_dist = math.sqrt(pow(kp[1]['y'] - x0, 2) + pow(kp[1]['y'] - y0, 2))

    for i in kp:
        if i['x'] is None or i['y'] is None: continue
        i['x'] = (i['x'] - x0) / neck_dist
        i['y'] = (i['y'] - y0) / neck_dist

    # create image PIL 
    image = Image.new('1', (512, 512), 0) # binary, size, background
    
    # draw object from PIL
    draw = ImageDraw.Draw(image)
    
    # white line
    line_color = 1
    line_thickness = 4
    
    # stickman scale
    scale = 30

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
    folder_path = f'./images/binary_pose/{folder_name.split("/")[-1]}'
    file_name = f'{folder_path}/pose_{frame_number}.png'

    # Check Directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save Image
    image.save(file_name)

    # Print Log
    print(f'Binary Pose Image Save in: {file_name}')