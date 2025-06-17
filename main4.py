import input_process4 as ips
import cv2
import numpy as np
import keras
import os 
import numpy as np
import matplotlib.pyplot as plt
import sudoku_solver as sv


def plot_table(array_2d):
    if not array_2d:  # Check if array_2d is False or invalid
        print("No solution exists")
        return
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=array_2d, loc='center', cellLoc='center')
    plt.show()





def save_table(array_2d, save_dir, file_name):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=array_2d, loc='center', cellLoc='center')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    plt.close(fig)

# Enhanced matrix formation with confidence checking
def form_sudoku_matrix(input_data, check, model, confidence_threshold=0.7):
    """Form Sudoku matrix with confidence checking."""
    sudoku_table = np.zeros((9, 9), dtype=int)
    confidence_matrix = np.zeros((9, 9))
    
    for i in range(9):
        for j in range(9):
            if check[i][j]:
                predictions = model.predict(input_data[i][j], verbose=0)
                digit = np.argmax(predictions) + 1
                confidence = np.max(predictions)
                sudoku_table[i][j]=digit
                # # Only accept high-confidence predictions
                # if confidence > confidence_threshold:
                #     sudoku_table[i][j] = digit
                #     confidence_matrix[i][j] = confidence
                # else:
                #     sudoku_table[i][j] = 0  # Treat as empty if low confidence
                #     print(f"Low confidence ({confidence:.2f}) at position ({i},{j})")
            else:
                sudoku_table[i][j] = 0
    
    return sudoku_table, confidence_matrix


root_dir=os.path.dirname(os.path.abspath(__file__))
print("local path:",root_dir)

model_path = os.path.join(root_dir, "digit_recognizer", "digit_recognizer.keras")
# model_path = os.path.join(root_dir, "model", "model_one.keras")

model = keras.models.load_model(model_path)
# model.summary()

input_image_path = os.path.join(root_dir, "soduku_tables", "20.jpg")
save_path=os.path.join(root_dir, "extracted_table")

img=cv2.imread(input_image_path)
# cv2.imshow("input",img)
# cv2.waitKey()
a,b,c=img.shape
# print(a,b,c)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = ips.resize_and_maintain_aspect_ratio(input_image=img, new_width=1000)

# cv2.imshow("input",img)
# cv2.waitKey()


cells, _,_ = ips.get_valid_cells_from_image(img)

# print(len(cells[0])*len(cells))

# # ips.plot_cell_images_in_grid(cells)

cells_arr = []
check = []

for i, cell in enumerate(cells):
    row_idx = i // 9
    if len(cells_arr) <= row_idx:
        cells_arr.append([])  # start a new row
        check.append([])

    cells_arr[row_idx].append(cell['img'])
    check[row_idx].append(cell['contains_digit'])

print(len(cells_arr), len(cells_arr[0]))  # should be 9, 9
print(len(check), len(check[0]))          # should be 9, 9

for i in range(9):
    for j in range(9):
        cell=cells_arr[i][j]
        cell=cv2.bitwise_not(cell)
        cell = cell.astype('float32') / 255.0
        cell= cell.reshape(1,28, 28, 1)  # Shape for model input
        cells_arr[i][j]=cell
        
# debug_dir = "debug_cells2"

# if not os.path.exists(debug_dir):
#     os.makedirs(debug_dir)

# for i in range(9):

#     for j in range(9):
#         # Save cell for debugging
#         cv2.imwrite(f"{debug_dir}/cell_{i}_{j}_digit_{check[i][j]}.jpg", cells_arr[i][j])





sudoku_table,conf=form_sudoku_matrix(cells_arr,check,model)
# save_table(sudoku_table,save_path,'table24')
print(sudoku_table)


solved=sv.solve(sudoku_table)
plot_table(solved)