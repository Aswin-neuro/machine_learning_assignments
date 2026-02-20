import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import torch
    import os
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import math
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        DataLoader,
        ImageFolder,
        compute_device,
        math,
        mo,
        os,
        plt,
        torch,
        transforms,
    )


@app.cell
def _(DataLoader, ImageFolder, transforms):
    t_folders=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(), #conversion to tensor (0 -1)
    ])

    train_dset = ImageFolder(root="/home/aswin/ml_proj/a6/training_data/", transform=t_folders)

    train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
    ## batchsize 64 helps in good gppu
    return t_folders, train_dset, train_loader


@app.cell
def _():
    # Check if the model is working
    # images, labels = next(iter(train_loader))
    ### print(f"Image Shape: {images[0].shape}")
    return


@app.cell
def _(torch):
    ## nn model - l1, l2 with relu and output with softmax
    def forward_prop(w1, b1, w2, b2, w3, b3, X):
        # Layer 1
        z1 = torch.matmul(w1, X) + b1
        a1 = torch.relu(z1)

        # Layer 2
        z2 = torch.matmul(w2, a1) + b2
        a2 = torch.relu(z2)

        # Layer3 (output)
        z3 = torch.matmul(w3, a2) + b3
        exp_z3 = torch.exp(z3 - torch.max(z3, dim=0, keepdim=True)[0])
        a3 = exp_z3 / torch.sum(exp_z3, dim=0, keepdim=True)

        return z1, a1, z2, a2, z3, a3
    return (forward_prop,)


@app.cell
def _(torch):
    def backward_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y):
        m = Y.size(1)

        # layer3
        dz3 = a3 - Y
        dw3 = (1/m) * torch.matmul(dz3, a2.T)
        db3 = (1/m) * torch.sum(dz3, dim=1, keepdim=True)

        # layer2
        da2 = torch.matmul(w3.T, dz3)
        dz2 = da2 * (z2 > 0).float() # ReLU derivative
        dw2 = (1/m) * torch.matmul(dz2, a1.T)
        db2 = (1/m) * torch.sum(dz2, dim=1, keepdim=True)

        # layer1
        da1 = torch.matmul(w2.T, dz2)
        dz1 = da1 * (z1 > 0).float() # ReLU derivative
        dw1 = (1/m) * torch.matmul(dz1, X.T)
        db1 = (1/m) * torch.sum(dz1, dim=1, keepdim=True)

        return dw1, db1, dw2, db2, dw3, db3
    return (backward_prop,)


@app.function
def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3
    return w1, b1, w2, b2, w3, b3


@app.cell
def _(torch):
    def main(Y, num_classes=10):
        main_y=torch.zeros((num_classes, Y.size(0)), device=Y.device)

        ## scatter 1s into the correct indices
        main_y.scatter_(0, Y.unsqueeze(0),1)
        return main_y
    return (main,)


@app.cell
def _(compute_device, math, torch):
    def init_params(num_classes):
        n0 = 784   # Input
        n1 = 64   # Hidden 1
        n2 = 16    # Hidden 2
        n3 = num_classes # Output

        # layer1
        w1 = torch.randn(n1, n0, device=compute_device) * math.sqrt(2.0 / n0)
        b1 = torch.zeros(n1, 1, device=compute_device)
        # layer2
        w2 = torch.randn(n2, n1, device=compute_device) * math.sqrt(2.0 / n1)
        b2 = torch.zeros(n2, 1, device=compute_device)
        # layer3 (output)
        w3 = torch.randn(n3, n2, device=compute_device) * math.sqrt(2.0 / n2)
        b3 = torch.zeros(n3, 1, device=compute_device)

        return w1, b1, w2, b2, w3, b3
    return


@app.cell
def _(
    backward_prop,
    compute_device,
    forward_prop,
    init_params_2,
    main,
    torch,
    train_dset,
    train_loader,
):
    num_classes = len(train_dset.classes)
    w1, b1, w2, b2, w3, b3 = init_params_2(num_classes)

    ## hyperparameters
    epochs = 100
    alpha = 0.009

    ## Start Training
    print("Starting fresh 3-layer training loop...")
    for i in range(epochs):
        corr_predictions = 0
        tot_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # move data to gpu
            images, labels = images.to(compute_device), labels.to(compute_device)

            # reshape images: (batch,1,28,28) -> (784,batch)
            X = images.view(images.shape[0], -1).T
            # encode labels
            Y_main = main(labels, num_classes=num_classes)

    ## main modules
            # forward Pass
            z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
            # backward pass
            dw1, db1, dw2, db2, dw3, db3 = backward_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y_main)
            # weights update
            w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)

            ## tracking accuracy using a3 (the output of the final layer)
            predictions = torch.argmax(a3, dim=0)
            corr_predictions += (predictions == labels).sum().item()
            tot_samples += labels.size(0)

        accuracy = (corr_predictions / tot_samples) * 100
        print(f"epoch {i+1}/{epochs} accuracy: {accuracy:.2f}%")
    return b1, b2, b3, w1, w2, w3


@app.cell
def _(
    b1,
    b2,
    b3,
    compute_device,
    forward_prop,
    plt,
    torch,
    train_dset,
    w1,
    w2,
    w3,
):
    import random

    ## predict_image helper to get max acc a3
    def predict_image(image, w1, b1, w2, b2, w3, b3):
        image = image.to(compute_device)
        # Reshape to a single column vector: (784, 1)
        X = image.view(-1, 1)
        # Run fwd Prop, extracting a3 and ignore rest
        _, _, _, _, _, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
        # index with the highest probability
        prediction = torch.argmax(a3, dim=0).item()
        return prediction

    ## grab a random image
    random_idx = random.randint(0, len(train_dset) - 1)
    image, true_label = train_dset[random_idx]

    # get the prediction (6 params)
    predicted_label = predict_image(image, w1, b1, w2, b2, w3, b3)

    #output
    class_names = train_dset.classes
    true_class = class_names[true_label]
    predicted_class = class_names[predicted_label]

    print(f"True Class: {true_class}")
    print(f"Predicted Class: {predicted_class}")

    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"Predicted: {predicted_class} | True: {true_class}")
    plt.axis()
    plt.show()
    return class_names, random


@app.cell
def _(mo):
    mo.md(r"""
    ## Network architecture
    - There is 4 layers in total
    - Layer 1, input layer = 28x28 =784 neurons
    - Layer 2, hidden layer 1= 64 neurons
    - Layer 3, hidden layer 2= 16 neurons
    - Hyperparameters include
    - epochs and alpha = 100 epochs and 0.005 learning rate gave a 78.47% accuracy
    - I could have taken some part of the data and removed the undetectable ones
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    scikit - for performance analystics
    disabled gradient tracking for evaluation
    flatten each image into 1D vector
    final layer outputs, logits or probabilities are used to get predictions , then takes the maximum value

    ## outputs

    Accuracy → overall correctness

    Precision → how many predicted positives were correct

    Recall → how many actual positives were detected

    F1-score → harmonic mean of precision and recall
    """)
    return


@app.cell
def _(
    DataLoader,
    ImageFolder,
    b1,
    b2,
    b3,
    class_names,
    compute_device,
    forward_prop,
    plt,
    t_folders,
    torch,
    train_loader,
    w1,
    w2,
    w3,
):
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    def evaluate_full_performance(loader, w1, b1, w2, b2, w3, b3, class_names):
        print("Gathering predictions on the test dataset...\n")
    
        all_true_labels = []
        all_predictions = []
    
        # 1. Run the entire test set through the model
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(compute_device)
                X = images.view(images.shape[0], -1).T
            
                # Forward pass
                _, _, _, _, _, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
                # Get predicted class
                preds = torch.argmax(a3, dim=0)
            
                # Store true and predicted labels as lists
                all_true_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        # Calculate Metrics using Scikit-Learn
        acc = accuracy_score(all_true_labels, all_predictions)
        prec = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        rec = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    
        # Print results
        print("--- Model Performance Metrics ---")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")
    
        # generate the confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
    
        # plot the Confusion Matrix, Seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    
        plt.title('Confusion Matrix - Digit Recognition', fontsize=16)
        plt.xlabel('Predicted Digit', fontsize=14)
        plt.ylabel('Actual Digit', fontsize=14)
        # plt.savefig('confusion_matrix.png') 
        plt.show()
    #output
    test_dset = ImageFolder(root="/home/aswin/ml_proj/a6/training_data/", transform=t_folders)
    test_loader = DataLoader(test_dset, batch_size=64, shuffle=False)

    # class_names = train_dset.classes

    evaluate_full_performance(train_loader, w1, b1, w2, b2, w3, b3, class_names)
    return


@app.cell
def _(b1, b2, b3, w1, w2, w3):
    def save_weights_and_biases(w1, b1, w2, b2, w3, b3):
        print("start writing..")

        # converting to numpy and load in cpu
        weights = {
            "W01": w1.detach().cpu().numpy(),
            "W12": w2.detach().cpu().numpy(),
            "W23": w3.detach().cpu().numpy()
        }
    
        biases = {
            "B1": b1.detach().cpu().numpy(),
            "B2": b2.detach().cpu().numpy(),
            "B3": b3.detach().cpu().numpy()
        }
    # saving weights
        with open("network_weights.txt", "w") as f:
            for layer_name, w_matrix in weights.items():
                f.write(f"___{layer_name} weights___\n")
                num_neurons, num_inputs = w_matrix.shape
            
                for neuron_idx in range(num_neurons):
                    # Identifies the layer, the specific neuron, and its inputs
                    prefix = f"{layer_name}_neuron_{neuron_idx}: "
                    #each single weight to 3 decimal places
                    formatted_weights = ", ".join([f"{val:.3f}" for val in w_matrix[neuron_idx]])
                    f.write(prefix + formatted_weights + "\n")
                f.write("\n") #blank
            
        print("- Created 'network_weights.txt' successfully.")

        # saving biases
        with open("network_biases.txt", "w") as f:
            for layer_name, b_matrix in biases.items():
                f.write(f"___{layer_name} biases___\n")
                num_neurons = b_matrix.shape[0]
            
                for neuron_idx in range(num_neurons):
                    # Nomenclature: Identifies the layer and the specific neuron's bias
                    # b_matrix[neuron_idx][0] grabs the single scalar value
                    bias_val = b_matrix[neuron_idx][0]
                    f.write(f"{layer_name}_neuron_{neuron_idx}: {bias_val:.3f}\n")
                
                f.write("\n") #blank
            
        print("- Created 'network_biases.txt' successfully.")

    # Execute the function after your training loop finishes
    save_weights_and_biases(w1, b1, w2, b2, w3, b3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2
    2.Repeat the same exercise for a diﬀerent design and report each step as deﬁned above.
    Also make the comparative evaluation of both the design of your both the networks
    """)
    return


@app.cell
def _(compute_device, math, torch):
    def init_params_2(num_classes):
        n0 = 784   # Input
        n1 = 128   # Hidden 1
        n2 = 64   # Hidden 2
        n3 = num_classes # Output

        # layer1
        w1 = torch.randn(n1, n0, device=compute_device) * math.sqrt(2.0 / n0)
        b1 = torch.zeros(n1, 1, device=compute_device)
        # layer2
        w2 = torch.randn(n2, n1, device=compute_device) * math.sqrt(2.0 / n1)
        b2 = torch.zeros(n2, 1, device=compute_device)
        # layer3 (output)
        w3 = torch.randn(n3, n2, device=compute_device) * math.sqrt(2.0 / n2)
        b3 = torch.zeros(n3, 1, device=compute_device)

        return w1, b1, w2, b2, w3, b3
    return (init_params_2,)


@app.function
def bw_convert1(img,T=90):
    bw = img.copy()
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i,j]>=T:
                bw[i,j] = 255
            else:
                bw[i,j] = 0
    return bw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I am using the network 1 with updated parameters for model 2
    """)
    return


@app.cell
def _():
    ## Random dot generation over fence
    return


@app.cell
def _(random):
    import cv2

    img1 = cv2.imread("fence.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape
    def bw_convert1(img,T=90):
        bw = img.copy()
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                if img[i,j]>=T:
                    bw[i,j] = 255
                else:
                    bw[i,j] = 0
        return bw
    radius = 5
    def random_circle_generator(img):
        for i in range(100):
            img_circle = img.copy()
            x = random.randint(radius, w - radius - 1)
            y = random.randint(radius, h - radius - 1)
            cv2.circle(img_circle, (x, y), radius, 0, -1)
            cv2.imwrite(f"with_circle/img_{i+1}.jpg", img_circle)
        print('done')

    img_fence = bw_convert1(img1)
    random_circle_generator(img_fence)
    return (cv2,)


@app.cell
def _(cv2, os):
    import shutil

    roll_no = "ms22276"
    source_folder = ""

    # destination folders
    in_folder = f"inside_{roll_no}"
    out_folder = f"outside_{roll_no}"
    os.makedirs(in_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    # counters
    in_count = 0
    out_count = 0
    target = 25

    print("sort manually by yourelf")
    print(f"Press 'i' for INSIDE.")
    print(f"Press 'o' for OUTSIDE.")
    print(f"Press 's' to SKIP (if circle is on the line).")
    print(f"Press 'q' to QUIT.\n")

    #list of generated images
    files = sorted(os.listdir(source_folder))

    for filename in files:
        if in_count >= target and out_count >= target:
            print("\nSuccess! You have collected 25 images for both categories.")
            break

        if not filename.endswith((".jpg", ".png")):
            continue
        # read image
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # show image
        cv2.imshow("Sort Images: 'i'=Inside, 'o'=Outside", img)
        key = cv2.waitKey(0) & 0xFF  #wait indefinitely for key

        if key == ord('i'):  #inside
            if in_count < target:
                in_count += 1
                # Rename format: in_rollno_1.jpg
                new_name = f"in_{roll_no}_{in_count}.jpg"
                shutil.copy(img_path, os.path.join(in_folder, new_name))
                print(f"Saved to inside ({in_count}/25)")
            else:
                print("Inside folder is full! Skipping...")

        elif key == ord('o'):  # outside
            if out_count < target:
                out_count += 1
                # format: out_rollno_1.jpg
                new_name = f"out_{roll_no}_{out_count}.jpg"
                shutil.copy(img_path, os.path.join(out_folder, new_name))
                print(f"Saved to outside({out_count}/25)")
            else:
                print("Outside folder is full")
        elif key == ord('s'):  #skip
            print("Skipped.")
        elif key == ord('q'):  #quit
            break

    cv2.destroyAllWindows()
    print("\nSorting complete.")
    return


if __name__ == "__main__":
    app.run()
