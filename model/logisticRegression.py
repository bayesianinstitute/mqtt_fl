import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply the hidden layer with ReLU activation
        hidden_output = torch.relu(self.hidden_layer(x))

        # Apply the output layer with sigmoid activation
        output = torch.sigmoid(self.output_layer(hidden_output))

        return output
