import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

debug = False

def openfr(paff):
    '''le opener'''
    return cv2.cvtColor(cv2.imread(paff), cv2.COLOR_BGR2RGB)

colors = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (128, 0, 0),       # Maroon
    (0, 128, 0),       # Dark Green
    (0, 0, 128),       # Navy
    (128, 128, 0),     # Olive
    (128, 0, 128),     # Purple
    (0, 128, 128),     # Teal
    (192, 192, 192),   # Silver
    (255, 165, 0),     # Orange
    (0, 100, 0),       # Dark Green (darker)
    (75, 0, 130),      # Indigo
    (255, 20, 147),    # Deep Pink
    (105, 105, 105),   # Dim Gray
    (0, 191, 255),     # Deep Sky Blue
    (160, 82, 45)      # Sienna
]


def pil_show(image):
    '''meant for displaying the image in jupiter notebook'''
    image = Image.fromarray(image)
    display(image)


def get_points(corners):
    if(len(corners) != 4):
        print("ERROR: passed countor doesn't match a rectangular type")
        
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
    return top_l, top_r, bottom_r, bottom_l


def biggest_contour(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    grey = cv2.GaussianBlur(grey, (3, 3), 0)

    ret, thresh = cv2.threshold(grey, 127, 255, 0)
    #edged = cv2.Canny(grey, 30, 200)
    
    inv = cv2.bitwise_not(thresh)
    
    contours, hierarchy = cv2.findContours(inv,
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    neo = img.copy()
    cv2.drawContours(neo, contours, -1, (255, 0, 0), 3)
    

    very_neo = img.copy()
    largest_contour = max(contours, key=cv2.contourArea)


    #test
    eps = 0.1*cv2.arcLength(largest_contour, True)
    largest_contour = cv2.approxPolyDP(largest_contour,eps,True)

    cv2.drawContours(very_neo, [largest_contour], 0, (255, 0, 0), 3)
   
   
    if(debug):
        pil_show(thresh)
        pil_show(neo)
        pil_show(inv)
        pil_show(very_neo)
   
   
    return largest_contour


def crop_down(img, contour):


    image = img.copy()

    top_l, top_r, bottom_r, bottom_l = get_points(contour)


    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))


    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                       [0, height - 1]], dtype="float32")
    # Convert to Numpy format
    ordered_corners = [top_l, top_r, bottom_r, bottom_l]
    ordered_corners = np.array(ordered_corners, dtype="float32")# calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    cropped = cv2.warpPerspective(image, grid, (width, height))

    if(debug):
        pil_show(cropped)
    
    return cropped

def boldify_lines(image):
    '''unused, gonna leave it here just in case'''
    # Copy image to draw on
    output = image.copy()


    # Edge detection
    edges = cv2.Canny(output, 0, 100, apertureSize=7)

    # Get image dimensions
    h, w = image.shape[:2]
    avg_dim = (h + w) / 2
    min_length = avg_dim / 2  # threshold for bolding lines

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=int(min_length), maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= min_length:
                # Draw bolder line (thickness=3)
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

    return output


def split_grid(image):
    '''literally just dividing the image into squares
    I will work on a better approach if I get the time (and will)'''

    # Ensure grayscale for consistent slicing
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    cell_h, cell_w = height // 9, width // 9

    cells = []

    for row in range(9):
        row_cells = []
        for col in range(9):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            cell = image[y1:y2, x1:x2]
            row_cells.append(cell)
        cells.append(row_cells)

    return cells


def preprocess(image):
    img = image.copy()
    table_edge = biggest_contour(img)
    cropped = crop_down(img, table_edge)
    pil_show(cropped)

    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)


    thresh = cv2.adaptiveThreshold(gray, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 20)

    pil_show(thresh)
    neg = cv2.bitwise_not(thresh)
    pil_show(neg)

    arr = split_grid(neg)
    return arr

image = openfr("image.jpg")
grid = preprocess(image)

for i in range(len(grid)):
    for j in range(len(grid[i])):
        cv2.imwrite("cell" + str(i) + str(j) + ".jpg", grid[i][j])

#denoised = cv2.morphologyEx(neg, cv2.MORPH_OPEN, kernel)
#pil_show(denoised)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#ran = cv2.morphologyEx(neg, cv2.MORPH_DILATE, kernel)
#pil_show(ran)

