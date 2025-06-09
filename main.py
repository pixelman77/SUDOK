

from input_process import get_input
import numpy as np
import keras
import os 
import cv2 
import numpy as np
import matplotlib.pyplot as plt

def plot_table(array_2d):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=array_2d, loc='center', cellLoc='center')
    plt.show()

def form_soduku_matrix(input,check,model):
    soduku_table=np.zeros((9,9),dtype=int)
    for i in range(9):
        for j in range(9):
            if check[i][j]==True:

                # predict=model.predict(input[i][j],verbose=0)
                # digit=np.argmax(predict)+1
                soduku_table[i][j]=1
            else:
                soduku_table[i][j]=0
    return soduku_table


    

root_dir=os.path.dirname(os.path.abspath(__file__))
print("local path:",root_dir)

model_path = os.path.join(root_dir, "model", "model_one.keras")
model = keras.models.load_model(model_path)
model.summary()

input_image_path = os.path.join(root_dir, "soduku_tables", "5.jpg")

input_arr,check_digit = get_input(input_image_path)

# print(check_digit[0][0])
# table=form_soduku_matrix(input_arr,check_digit,model)
# print(table)


# Display the original and preprocessed cell
cell = input_arr[0][0]  # Adjust indices as needed
cv2.imshow("Original Cell", cell)
cell_resized = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imshow("Preprocessed Cell (28x28)", cell_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cell=cv2.bitwise_not(cell)
# cv2.imshow("not",cell)
# cv2.waitKey()
cell=cell_resized.astype('float32')/255.0
cell=cell.reshape(1,28,28,1)
print(check_digit[5][2])
predict=model.predict(cell)
print(np.argmax(predict)+1)
def print_2d_list(grid):
    for row in grid:
        print(" ".join(f"{x:2}" for x in row))

print_2d_list(check_digit)
# print(check_digit)
plot_table(form_soduku_matrix(input_arr,check_digit,model))


