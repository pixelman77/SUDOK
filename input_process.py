import cv2
import numpy as np
from PIL import Image
from IPython.display import display

def openfr(paff):
    return cv2.cvtColor(cv2.imread(paff), cv2.COLOR_BGR2RGB)

def pil_show(image):
    image = Image.fromarray(image)
    display(image)

def order_corners(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    top_left     = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right    = pts[np.argmin(diff)]
    bottom_left  = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def biggest_contour(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    _, thresh = cv2.threshold(grey, 127, 255, 0)
    inv = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    eps = 0.1 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, eps, True)
    return approx

def crop_down(img, contour):
    pts = contour.reshape(4, 2)
    ordered_corners = order_corners(pts)
    
    width_A = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    width_B = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    width = int(max(width_A, width_B))

    height_A = np.linalg.norm(ordered_corners[1] - ordered_corners[2])
    height_B = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
    height = int(max(height_A, height_B))

    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    cropped = cv2.warpPerspective(img, M, (width, height))
    return cropped

def split_grid(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    cell_h, cell_w = height // 9, width // 9
    cells = []
    for row in range(9):
        row_cells = []
        for col in range(9):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            row_cells.append(image[y1:y2, x1:x2])
        cells.append(row_cells)
    return cells

def preprocess(image):
    img = image.copy()
    table_edge = biggest_contour(img)
    cropped = crop_down(img, table_edge)
    pil_show(cropped)
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 20)
    pil_show(thresh)
    cell = cv2.bitwise_not(thresh)
    pil_show(cell)
    return split_grid(cell)


def check_for_digit_in_cell_image(img, area_threshold=5, apply_border=False):
    '''
    Determine whether or not a digit is present in an image of a single
    sudoku cell based on contour area.
    
    Args:
        img (np.ndarray): An image of a single sudoku cell (expected grayscale, inverted).
        area_threshold (int/float): Threshold value (percentage of image area).
                                    A contour is considered a digit if its area exceeds this.
        apply_border (bool): Whether to apply a mask to remove non-digit pixels
                             around the edges of the cell image.
            
    Returns:
        tuple: (image_contains_digit (bool), cell_img (np.ndarray))
               - `image_contains_digit`: True if a significant contour (digit) is found.
               - `cell_img`: The processed cell image (with optional border masked out).
    '''
    cell_img = img.copy()
  
    if apply_border:
        # Crude way to eliminate the unwanted pixels around the borders
        border_fraction = 0.07
        replacement_val = 0 # Assuming black background for digits

        y_border_px = int(border_fraction * cell_img.shape[0])
        x_border_px = int(border_fraction * cell_img.shape[1])
        
        # Set border pixels to replacement_val (0 for black)
        cell_img[:, 0:x_border_px] = replacement_val
        cell_img[:, -x_border_px:] = replacement_val
        cell_img[0:y_border_px, :] = replacement_val
        cell_img[-y_border_px:, :] = replacement_val
    
    # Get the contours for the image
    
    contours_found = cv2.findContours(image=cell_img,
                                mode=cv2.RETR_TREE, # Retrieve all contours and hierarchy
                                method=cv2.CHAIN_APPROX_SIMPLE) # Compress horizontal/vertical/diagonal segments
    
    # Handle different OpenCV versions returning contours differently
    
    # Handle varying return formats of cv2.findContours across OpenCV versions
    contours = contours_found[0] if len(contours_found) == 2 else contours_found[1]
    
    
    if len(contours) > 0:
        # Sort the contours according to contour area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)        
        largest_contour_area = cv2.contourArea(contours[0])
        image_area = cell_img.shape[0] * cell_img.shape[1]
        contour_percentage_area = 100 * largest_contour_area / image_area
        
        # If the largest contour's area exceeds the threshold, it's considered a digit
        if contour_percentage_area > area_threshold:
            image_contains_digit = True
        else:
            image_contains_digit = False
        
    else:
        # No contours found, so no digit is present
        image_contains_digit = False
        
    return image_contains_digit, cell_img
   # return image_contains_digit


def get_input(path):
    image=openfr(path)
    grid=preprocess(image)
    check_digit = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(9):
       for j in range(9):
           cell=grid[i][j]
           check,_=check_for_digit_in_cell_image(cell.copy(),apply_border=True)
           check_digit[i][j]=check
        
        #    grid[i][j]=cv2.resize(grid[i][j],(28,28),interpolation=cv2.INTER_CUBIC)
        #    grid[i][j]=cv2.bitwise_not(grid[i][j])
        #    grid[i][j]=grid[i][j].astype('float32')/255.0
        #    grid[i][j]=grid[i][j].reshape(1,28,28,1)
           
 

    input=(grid,check_digit)
    return input

# # Example usage:
# image = openfr("soduku_tables/1.jpg")
# grid = preprocess(image)
# for i in range(len(grid)):
#     for j in range(len(grid[i])):
#         cv2.imwrite(f"ext/cell{i}{j}.jpg", grid[i][j])

