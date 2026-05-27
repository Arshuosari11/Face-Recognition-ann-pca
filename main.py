import os
import cv2
import numpy as np
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 


dataset_path = "dataset"

faces = []
labels = []

label = 0
label_names = {}  # Dictionary to map label numbers to person names

# loop through person folders
for person_name in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_name)

    # skip non-folders
    if not os.path.isdir(person_path):

        continue
    label_names[label] = person_name        # Map label number to person name

    print("Loading images from:", person_name)
    
    # loop through images inside folder
    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, 0)

        # skip invalid images
        if img is None:

            print("Could not read:", image_path)

            continue

        img = cv2.resize(img, (50, 50))

        img = img.flatten()

        faces.append(img)

        labels.append(label)

    # next person gets next label
    label += 1

# convert to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

print("Faces shape:", faces.shape)

# calculate mean face
mean_face = np.mean(faces, axis=0)

print("Mean face shape:", mean_face.shape)

# mean subtraction
A = faces - mean_face

print("A shape:", A.shape)    

#calculate covariance matrix
cov_matrix = np.dot(A, A.T)   
print("covariance matrix shape:", cov_matrix.shape) 

#calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(cov_matrix)
print("Eigenvalues shape:", eigenvalues.shape)
print("Eigenvectors shape:", eigenvectors.shape)

#sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

#select top k eigenvectors
k = 25
eigenvectors = eigenvectors[:, :k]
print("Selected eigenvectors shape:", eigenvectors.shape)

#generate eigenfaces
eigenfaces = np.dot(A.T, eigenvectors)
print("Eigenfaces shape:", eigenfaces.shape)

#normalize eigenfaces
for i in range(k):
    norm = np.linalg.norm(eigenfaces[:, i])
    if norm != 0:
       eigenfaces[:, i] = eigenfaces[:, i] /norm 

#project faces into PCA space
projected_data = np.dot(A, eigenfaces)

# standardize projected data
mean_pca = np.mean(projected_data, axis=0)

std_pca = np.std(projected_data, axis=0) + 1e-8

projected_data = (projected_data - mean_pca) / std_pca
print("Projected data shape:", projected_data.shape)

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(projected_data, labels, test_size=0.4, random_state=42)
print("Training data :", X_train.shape)
print("Testing data:", X_test.shape)

#network size
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(labels))
print("Input neurons:", input_size)
print("Hidden neurons:", hidden_size)
print("Output neurons:", output_size)

#Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

#Activation function
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#one-hot encoding
def one_hot(y, num_classes):
    onehot = np.zeros((len(y), num_classes))
    onehot[np.arange(len(y)), y] = 1
    return onehot
Y_train = one_hot(y_train, output_size)

#FORWARD PROPAGATION
def forward(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#loss function
def compute_loss(A2, Y):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(A2 + 1e-8)) / m
    return loss

print(label_names)
#train ANN
learning_rate = 0.001
epochs = 5000
for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward(X_train)    #forward pass
    loss = compute_loss(A2, Y_train)   #compute loss
    m = X_train.shape[0]
    
    #output layer gradients
    dZ2 = A2 - Y_train     
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    #hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X_train.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    #update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if epoch % 50 ==0:
        print(f"Epoch {epoch}, Loss: {loss}")

 #create function to predict
def predict(X):
    _, _, _, output = forward(X)
    predictions = np.argmax(output, axis=1)
    confidence = np.max(output, axis=1)
    return predictions, confidence
 
 #test model
predictions = predict(X_test)
print("Predictions:", predictions)
print("Actual Labels:", y_test)
 
#calculate accuracy
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")    

#Test single image
def recognize_face(image_path):
    print("Reading image from:", image_path)
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Image not found!")
        return
    
    img = cv2.resize(img, (50, 50)) 
    img = img.flatten()
    img = img - mean_face   # mean subtraction
    img_pca = np.dot(img, eigenfaces)   # project into PCA space
    img_pca = img_pca.reshape(1, -1)   # reshape for ANN input
    prediction, confidence = predict(img_pca)
    if confidence[0] < 0.3: # confidence threshold
        return "Not Enrolled"
    else:
        return label_names[prediction[0]]

#test on a new image 
if __name__ == "__main__":
    result = recognize_face(r"C:\Users\arshi\OneDrive\Desktop\face-recognization\dataset\jichangwook\img1.jpg")
    print(f"predicted person: {result}")

#Graph code
k_values = [10, 20, 30, 40, 50]
accuracies = [72, 81, 87, 90, 92]       # Example accuracies for different k values
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k values")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs k values")
plt.savefig("graph.png")
plt.show()


