# ML Scratch Algorithms

## Overview

This directory contains implementations of fundamental Machine Learning algorithms built entirely from scratch using NumPy and Python. No high-level ML libraries like scikit-learn are used for algorithm implementation—they are only used for data generation and metrics.

The goal is to provide a deep understanding of how these algorithms work mathematically and computationally.

## 📚 Algorithm Directory

### 1. **Linear Regression.ipynb**
**Purpose**: Predicting continuous values using linear relationships

**Topics Covered**:
- Dataset creation with linear relationships
- Mathematical background (residuals, error minimization)
- Implementation of slope and intercept calculation
- Model training and prediction
- Performance evaluation (MSE, R² score)
- Visualization of data and regression line

**Key Formula**: 
$$\text{slope} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

**Output**: Regression line visualization and accuracy metrics

---

### 2. **Logistic Regression.ipynb**
**Purpose**: Binary classification with probability estimation

**Topics Covered**:
- Creating linearly separable binary datasets
- Sigmoid activation function for probability mapping
- Stochastic Gradient Descent (SGD) algorithm
- Cost function derivation and implementation
- Parameter updates using gradient descent
- Decision boundary visualization
- Model accuracy and performance metrics

**Key Components**:
- **Sigmoid Function**: Maps output to probability (0-1)
- **Cost Function**: Measures prediction error
- **Gradient Descent**: Optimizes parameters

**Output**: Decision boundaries, accuracy scores, training progression

---

### 3. **K-Nearest-Neighbors (KNN).ipynb**
**Purpose**: Non-parametric classification using local similarity

**Topics Covered**:
- Dataset generation with Gaussian distributions
- Euclidian distance calculation between data points
- KNN algorithm implementation
- Distance-based classification
- Mode calculation for class prediction
- Performance analysis and visualization

**Key Concept**:
- Algorithm searches for k nearest neighbors
- Uses majority voting for classification
- No formal training phase (lazy learner)

**Mathematical Foundation**:
$$d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$

**Limitations**:
- Curse of dimensionality
- Requires feature normalization
- Computational complexity scales with dataset size

**Output**: Classification boundaries and accuracy metrics

---

### 4. **Artificial Neural Network.ipynb**
**Purpose**: Complex non-linear pattern recognition using neural networks

**Topics Covered**:
- Neural network architecture design
- Layer initialization with random weights
- Forward propagation through multiple layers
- Backpropagation algorithm for weight updates
- Activation functions (ReLU, Sigmoid, Tanh)
- Cost function calculation
- Training loop with convergence analysis
- Decision boundary visualization for classification tasks
- Accuracy tracking and performance metrics

**Key Components**:
1. **Layer Initialization**: Initialize weight matrices and biases
2. **Forward Pass**: Compute predictions through hidden and output layers
3. **Cost Calculation**: Measure prediction error
4. **Backward Pass**: Compute gradients using chain rule
5. **Parameter Update**: Update weights using gradient descent

**Architecture Example**:
```
Input Layer (n features) 
    ↓
Hidden Layer (h neurons with activation)
    ↓
Output Layer (output classes)
```

**Learning Process**:
- Initialize weights randomly
- Forward propagate: compute output
- Calculate loss
- Backpropagate: compute gradients
- Update weights: gradient descent
- Repeat until convergence

**Output**: 
- Training/validation curves
- 2D decision boundaries
- 3D visualization of learning progress
- Model accuracy and loss metrics

---

### 5. **RNN from scratch.ipynb**
**Purpose**: Sequential and temporal data processing using recurrent connections

**Topics Covered**:
- Recurrent Neural Network architecture
- Forward propagation through time
- Backpropagation Through Time (BPTT)
- Hidden state propagation and memory
- Weight matrices (U, W, V) for different connections
- Gradient clipping to prevent exploding gradients
- BPTT truncation for computational efficiency
- Time series prediction and sequence modeling

**Key Weight Matrices**:
- **U**: Input to Hidden layer weights
- **W**: Hidden to Hidden layer weights (recurrent)
- **V**: Hidden to Output layer weights

**RNN Process**:
1. Initialize weights and hidden state
2. For each time step:
   - Compute hidden state using previous hidden state and current input
   - Compute output using hidden state
   - Track prediction errors
3. Backpropagate through time:
   - Calculate gradients across all time steps
   - Apply gradient clipping for stability
   - Update weights
4. Repeat for multiple epochs

**Special Techniques**:
- **BPTT Truncation**: Limit backpropagation depth for computational efficiency
- **Gradient Clipping**: Prevent exploding gradients by clipping to [-10, 10] range
- **Hidden State**: Maintains context information across sequences

**Output**:
- Training/validation loss progression
- Model predictions vs actual values
- Visualization of learned patterns
- Loss curves showing convergence

---

## 🔧 Common Implementation Patterns

### Dataset Creation
All notebooks start with synthetic dataset generation to ensure:
- Full control over data characteristics
- Reproducible experiments
- Clear understanding of what the model should learn

### Training Loop
```
for epoch in range(num_epochs):
    1. Forward pass (compute predictions)
    2. Calculate loss
    3. Backward pass (compute gradients)
    4. Update parameters
    5. Track metrics
```

### Visualization
All notebooks include visualization of:
- Data distribution
- Model behavior (decision boundaries)
- Training progress (loss curves)
- Performance metrics

## 📊 Metrics Used

- **Accuracy**: Percentage of correct predictions
- **Mean Squared Error (MSE)**: Average squared prediction error
- **R² Score**: Coefficient of determination for regression
- **Loss Function**: Algorithm-specific error measure

## 🎯 Learning Objectives

By studying these implementations, you will understand:

1. ✅ Mathematical foundations of ML algorithms
2. ✅ How to implement algorithms from first principles
3. ✅ Gradient descent optimization in practice
4. ✅ Backpropagation and gradient computation
5. ✅ Neural network training dynamics
6. ✅ Sequential data processing with RNNs
7. ✅ Proper visualization and evaluation techniques

## 📖 How to Use

### Running a Notebook
```bash
jupyter notebook "Linear Regression.ipynb"
```

### Understanding the Flow
1. **Import Libraries**: See what tools are being used
2. **Data Creation**: Generate or load data
3. **Theory Section**: Read mathematical background
4. **Implementation**: Study the algorithm code
5. **Training**: See the model learning
6. **Visualization**: Understand results visually
7. **Evaluation**: Check performance metrics

### Experimenting
- Modify hyperparameters (learning rate, number of epochs, etc.)
- Try different data distributions
- Change dataset sizes
- Adjust network architectures (for ANN/RNN)
- Observe how changes affect learning

## 🚀 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn (for data generation and metrics only)
imageio (for animated visualizations)
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imageio
```

## 📝 Notes

- All algorithms are implemented to be educational—not production-optimized
- Code is heavily commented for clarity
- Each notebook is independent and self-contained
- Visualizations help build intuition about how algorithms work
- Hyperparameters are chosen for educational clarity, not optimal performance

## 🔗 References

See main README.md for detailed references to each algorithm's sources.

## 💡 Tips for Learning

1. **Code Along**: Don't just read, type and modify the code
2. **Experiment**: Change parameters and observe results
3. **Visualize**: Pay attention to the plots and animations
4. **Understand Math**: Read the mathematical background sections
5. **Connect Ideas**: Notice similarities between algorithms
6. **Ask Questions**: When something isn't clear, look deeper

---

**Created**: January 2026  
**Purpose**: Educational Machine Learning from Scratch
