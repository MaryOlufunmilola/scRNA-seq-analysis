# This script defines, train, and evaluate a machine learning model (e.g., neural network) to predict cell types based on the single-cell RNA-Seq data.

import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define a simple neural network model
class CellTypePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CellTypePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(classification_report(all_labels, all_preds))

def main():
    # Load processed data
    adata = sc.read("data/processed_data.h5ad")
    
    # Prepare the data (using PCA-reduced data for simplicity)
    X = adata.obsm['X_pca']  # PCA-reduced data
    y = adata.obs['kmeans_clusters'].values  # Target: Cluster labels from KMeans
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for batching
    train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_data = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]  # Number of features (PCA components)
    hidden_dim = 128
    output_dim = len(np.unique(y))  # Number of clusters
    model = CellTypePredictor(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    
    # Evaluate the model
    evaluate_model(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), "models/cell_type_predictor.pth")

if __name__ == "__main__":
    main()
