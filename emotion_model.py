import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np

class CNNEmotionModel(nn.Module):
    """
    CNN model for emotion classification from spectrograms
    """
    def __init__(self, num_classes=7, input_channels=1):
        """
        Initialize the CNN model
        
        Parameters:
        ----------
        num_classes : int
            Number of emotion classes
        input_channels : int
            Number of input channels (1 for mono spectrogram)
        """
        super(CNNEmotionModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # The output size will depend on the input spectrogram size
        # For a typical 128x128 spectrogram, after 4 pooling layers (2^4 = 16 reduction)
        # the output size would be 128/16 = 8, so 8x8 feature maps
        self.flatten_size = 128 * 8 * 8  # This needs to be adjusted based on input size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor containing spectrograms
            
        Returns:
        -------
        torch.Tensor
            Class logits
        """
        # Apply convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class LSTMEmotionModel(nn.Module):
    """
    LSTM model for emotion classification from audio features
    """
    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=7, bidirectional=True):
        """
        Initialize the LSTM model
        
        Parameters:
        ----------
        input_size : int
            Size of input features
        hidden_size : int
            Size of LSTM hidden state
        num_layers : int
            Number of LSTM layers
        num_classes : int
            Number of emotion classes
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(LSTMEmotionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * self.directions, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.directions, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
    
    def attention_net(self, lstm_output):
        """
        Attention mechanism to focus on important parts of the sequence
        
        Parameters:
        ----------
        lstm_output : torch.Tensor
            Output from LSTM layer
            
        Returns:
        -------
        torch.Tensor
            Context vector
        """
        # lstm_output shape: (batch_size, seq_len, hidden_size * directions)
        
        # Calculate attention scores
        attn_weights = self.attention(lstm_output)  # shape: (batch_size, seq_len, 1)
        
        # Apply softmax to get probabilities
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape: (batch_size, seq_len, 1)
        
        # Apply attention weights to LSTM output
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)  # shape: (batch_size, hidden_size * directions, 1)
        
        # Reshape context vector
        context = context.squeeze(2)  # shape: (batch_size, hidden_size * directions)
        
        return context
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor containing audio features
            Shape: (batch_size, seq_len, input_size)
            
        Returns:
        -------
        torch.Tensor
            Class logits
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention
        attn_output = self.attention_net(output)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(attn_output))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class CNNLSTMEmotionModel(nn.Module):
    """
    CNN-LSTM hybrid model for emotion classification
    """
    def __init__(self, input_channels=1, hidden_size=256, num_layers=2, num_classes=7):
        """
        Initialize the CNN-LSTM model
        
        Parameters:
        ----------
        input_channels : int
            Number of input channels (1 for mono spectrogram)
        hidden_size : int
            Size of LSTM hidden state
        num_layers : int
            Number of LSTM layers
        num_classes : int
            Number of emotion classes
        """
        super(CNNLSTMEmotionModel, self).__init__()
        
        # CNN part - extract features from spectrogram
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # LSTM part - model temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128 * 8,  # Assuming 8 frequency bins after pooling
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)  # 2 for bidirectional
        
        # Fully connected part - classification
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Save parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def attention_net(self, lstm_output):
        """Attention mechanism for LSTM output"""
        attn_weights = self.attention(lstm_output)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        return context.squeeze(2)
    
    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Reshape for LSTM
        # Assuming input is (batch_size, channels, freq_bins, time_steps)
        # Convert to (batch_size, time_steps, channels * freq_bins)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # LSTM
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Attention
        attn_output = self.attention_net(output)
        
        # Fully connected
        x = F.relu(self.fc1(attn_output))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class EmotionClassifier:
    """
    Wrapper class for emotion classification models
    """
    def __init__(self, model_type='cnn_lstm', model_path=None, num_classes=7, device=None):
        """
        Initialize the emotion classifier
        
        Parameters:
        ----------
        model_type : str
            Type of model to use ('cnn', 'lstm', 'cnn_lstm')
        model_path : str or None
            Path to pretrained model, if None, will initialize a new model
        num_classes : int
            Number of emotion classes
        device : str or None
            Device to use ('cuda', 'cpu'), if None, will use cuda if available
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_type = model_type
        self.num_classes = num_classes
        
        # Initialize model based on type
        if model_type == 'cnn':
            self.model = CNNEmotionModel(num_classes=num_classes)
        elif model_type == 'lstm':
            # Assuming a standard feature size of 193 (40 MFCC * 3 + 73 other features)
            self.model = LSTMEmotionModel(input_size=193, num_classes=num_classes)
        elif model_type == 'cnn_lstm':
            self.model = CNNLSTMEmotionModel(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model if path is provided
        if model_path:
            self.load_model(model_path)
        
        self.model = self.model.to(self.device)
        
        # Class names (map indices to emotion names)
        self.class_names = {
            0: 'angry',
            1: 'disgusted',
            2: 'fearful',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprised'
        }
    
    def train(self, train_loader, valid_loader=None, epochs=20, lr=0.001, weight_decay=1e-5):
        """
        Train the model
        
        Parameters:
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data
        valid_loader : torch.utils.data.DataLoader or None
            DataLoader for validation data, if None, no validation is performed
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        weight_decay : float
            Weight decay for optimizer
            
        Returns:
        -------
        dict
            Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': []
        }
        
        # Train for a number of epochs
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss and accuracy
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # Calculate epoch metrics
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if valid_loader is not None:
                valid_loss, valid_acc = self.evaluate(valid_loader)
                history['valid_loss'].append(valid_loss)
                history['valid_acc'].append(valid_acc)
                
                # Update learning rate
                scheduler.step(valid_loss)
                
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        return history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model
        
        Parameters:
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for evaluation data
            
        Returns:
        -------
        float
            Average loss
        float
            Accuracy
        """
        # Set model to evaluation mode
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        eval_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Track loss and accuracy
                eval_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        eval_loss = eval_loss / len(data_loader.dataset)
        accuracy = correct / total
        
        # Set model back to training mode
        self.model.train()
        
        return eval_loss, accuracy
    
    def predict(self, inputs):
        """
        Make predictions
        
        Parameters:
        ----------
        inputs : torch.Tensor or np.ndarray
            Input data
            
        Returns:
        -------
        np.ndarray
            Predicted class indices
        np.ndarray
            Predicted probabilities
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert numpy array to tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        
        # Add batch dimension if needed
        if len(inputs.shape) == 3:  # For spectrograms (channel, height, width)
            inputs = inputs.unsqueeze(0)
        elif len(inputs.shape) == 2:  # For features (seq_len, input_size)
            inputs = inputs.unsqueeze(0)
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the predicted class indices
            _, predicted = torch.max(outputs, 1)
        
        # Convert to numpy
        predicted = predicted.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        return predicted, probabilities
    
    def predict_emotion(self, inputs):
        """
        Predict emotion class names
        
        Parameters:
        ----------
        inputs : torch.Tensor or np.ndarray
            Input data
            
        Returns:
        -------
        list
            Predicted emotion names
        np.ndarray
            Predicted probabilities
        """
        # Get predictions
        predictions, probabilities = self.predict(inputs)
        
        # Convert indices to emotion names
        emotion_names = [self.class_names[pred] for pred in predictions]
        
        return emotion_names, probabilities
    
    def save_model(self, path):
        """
        Save model to file
        
        Parameters:
        ----------
        path : str
            Path to save the model
        """
        # Save model state dict
        torch.save({
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'state_dict': self.model.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from file
        
        Parameters:
        ----------
        path : str
            Path to the saved model
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Check if model type matches
            if checkpoint['model_type'] != self.model_type:
                print(f"Warning: Loaded model type ({checkpoint['model_type']}) doesn't match initialized model type ({self.model_type})")
            
            # Load state dict
            self.model.load_state_dict(checkpoint['state_dict'])
            self.num_classes = checkpoint['num_classes']
            
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False 
