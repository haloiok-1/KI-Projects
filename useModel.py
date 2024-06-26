import trainModel
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load the model
try:
    model = trainModel.load_model()
except:
    print("The model could not be loaded")
    print("Create a model first by running the trainModel.py file")
    exit()

# Example input
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# result
output = trainModel.predict([1, 0])
result_list = list(output[0])

if result_list[0] > 0.8:
    print("The output is 1")
elif result_list[1] > 0.8:
    print("The output is 0")
else:
    print("The output is not clear")

# create simpe gui with tkinter


# create the main window
root = tk.Tk()
root.title("XOR Neural Network")
root.geometry("400x200")

# add title label
title = tk.Label(root, text="XOR Neural Network")

#add explenation label
explanation = tk.Label(root, text="This is a simple XOR neural network.\n"
                                  " Enter two values (0 or 1) and click predict")
explanation.pack()

#create space
space = tk.Label(root, text="")
space.pack()

# create the label
label = tk.Label(root, text="Enter the input values:")
label.pack()

# create two entry fields
entry1 = tk.Entry(root)
entry1.pack()
entry2 = tk.Entry(root)
entry2.pack()


# create the predict button
def predict():
    try:
        input_values = [int(entry1.get()), int(entry2.get())]

    except:  # if the input is not a number
        messagebox.showerror("Error", "The input values must be numbers")
        return

    # check if the input is valid
    if input_values[0] not in [0, 1] or input_values[1] not in [0, 1]:
        messagebox.showerror("Error", "The input values must be 0 or 1")
        return

    output = trainModel.predict([input_values])
    result_list = list(output[0])

    if result_list[0] > 0.8:
        result_label.config(text="The output is 1 with a probability of " + str(result_list[0] * 100) + "%")
    elif result_list[1] > 0.8:
        result_label.config(text="The output is 0 with a probability of " + str(result_list[1] * 100) + "%")
    else:
        result_label.config(text="The output is not clear")


button = tk.Button(root, text="Predict", command=predict)
button.pack()

# add result label
result_label = tk.Label(root, text="")
result_label.pack()

# add space
space = tk.Label(root, text="")
space.pack()

#create credits label
credits = tk.Label(root, text="Made by: Czylonio")
credits.pack()

# run the main loop
root.mainloop()
