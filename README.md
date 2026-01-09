# Axioms of Machine Learning

A comprehensive repository implementing fundamental Machine Learning algorithms from scratch using NumPy and Python, without relying on high-level libraries like scikit-learn.

## 📋 Project Overview

This project demonstrates the mathematical foundations and implementation details of core machine learning algorithms. Each algorithm is implemented from scratch to provide deep understanding of how these methods work at their core.

## 📁 Repository Structure

```
Axioms-of-Machine-Learning/
├── README.md
├── ML Scratch Algorithms/
│   ├── Linear Regression.ipynb
│   ├── Logistic Regression.ipynb
│   ├── K-Nearest-Neighbors (KNN).ipynb
│   ├── Artificial Neural Network.ipynb
│   ├── RNN from scratch.ipynb
│   └── README.md
└── scratch_mlp/
    └── plots/
        ├── accuracy/
        ├── loss/
        ├── boundary/
        ├── gif/
        └── all/
```

## 🧠 Implemented Algorithms

### 1. **Linear Regression**
- **File**: [ML Scratch Algorithms/Linear Regression.ipynb](ML%20Scratch%20Algorithms/Linear%20Regression.ipynb)
- **Description**: Implementation of simple linear regression to find the best-fit line through data points
- **Mathematical Concepts**: 
  - Residuals equation
  - Minimization of sum of squared errors
  - Slope and intercept calculation
- **Key Methods**: Normal equation approach
- **Applications**: Predictive modeling for continuous values

### 2. **Logistic Regression**
- **File**: [ML Scratch Algorithms/Logistic Regression.ipynb](ML%20Scratch%20Algorithms/Logistic%20Regression.ipynb)
- **Description**: Binary classification using sigmoid activation and stochastic gradient descent
- **Mathematical Concepts**:
  - Sigmoid function for probability estimation
  - Cost function derivation
  - Stochastic Gradient Descent (SGD) optimization
  - Parameter updates via gradient descent
- **Activation Function**: Sigmoid
- **Optimization**: Stochastic Gradient Descent
- **Applications**: Binary classification problems

### 3. **K-Nearest-Neighbors (KNN)**
- **File**: [ML Scratch Algorithms/K-Nearest-Neighbors (KNN).ipynb](ML%20Scratch%20Algorithms/K-Nearest-Neighbors%20(KNN).ipynb)
- **Description**: Non-parametric algorithm for classification and regression
- **Mathematical Concepts**:
  - Euclidian distance metric
  - Voting mechanism for classification
  - Mode calculation for class prediction
- **Distance Metric**: Euclidian Distance
- **Key Features**:
  - Simple and intuitive
  - No training phase required
  - Sensitive to feature scaling
- **Limitations**: Curse of dimensionality, computational complexity for large datasets
- **Applications**: Classification and regression with local decision boundaries

### 4. **Artificial Neural Network (ANN)**
- **File**: [ML Scratch Algorithms/Artificial Neural Network.ipynb](ML%20Scratch%20Algorithms/Artificial%20Neural%20Network.ipynb)
- **Description**: Multi-layer feedforward neural network with backpropagation
- **Mathematical Concepts**:
  - Layer initialization
  - Forward propagation through multiple layers
  - Backpropagation algorithm
  - Weight updates and optimization
  - Visualization of decision boundaries
- **Architecture**: Input layer → Hidden layer(s) → Output layer
- **Key Components**:
  - Activation functions (ReLU, Sigmoid)
  - Cost function calculation
  - Parameter optimization via gradient descent
  - Accuracy metrics and visualization
- **Applications**: Complex non-linear pattern recognition and classification tasks

### 5. **Recurrent Neural Network (RNN)**
- **File**: [ML Scratch Algorithms/RNN from scratch.ipynb](ML%20Scratch%20Algorithms/RNN%20from%20scratch.ipynb)
- **Description**: Sequence-to-sequence learning with recurrent connections
- **Mathematical Concepts**:
  - Recurrent connections and hidden state propagation
  - Backpropagation Through Time (BPTT)
  - Gradient clipping
  - Weight matrices for input-hidden, hidden-hidden, and hidden-output connections
- **Weight Matrices**:
  - U: Input to Hidden layer
  - W: Hidden to Hidden layer (recurrent connections)
  - V: Hidden to Output layer
- **Key Features**:
  - Handles sequential/temporal data
  - Memory of previous inputs
  - BPTT truncation for stability
  - Gradient clipping to prevent exploding gradients
- **Applications**: Time series prediction, sequence modeling, language processing

## 📊 Visualization & Output

The project includes extensive visualization capabilities:
- **Accuracy Plots**: Training and validation accuracy curves
- **Loss Curves**: Training and validation loss progression
- **Decision Boundaries**: 2D visualization of classifier boundaries
- **GIF Animations**: Animated visualizations of learning progression
- **All Metrics**: Comprehensive performance metrics

All plots are stored in the `scratch_mlp/plots/` directory organized by type.

## 🛠️ Technologies Used

- **NumPy**: Numerical computations and matrix operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **scikit-learn**: For datasets and metrics (only used for data generation, not algorithm implementation)
- **Python 3**: Core programming language

## 📚 Key Learning Outcomes

1. **Mathematical Foundation**: Understanding the mathematical principles behind ML algorithms
2. **Implementation Skills**: Practical implementation of complex algorithms from first principles
3. **Optimization Techniques**: Gradient descent, SGD, and backpropagation
4. **Neural Networks**: Deep understanding of how neural networks learn and propagate information
5. **Visualization**: Effective visualization of model behavior and decision-making processes

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imageio
```

### Running Notebooks
```bash
jupyter notebook
```

Then navigate to any notebook in the `ML Scratch Algorithms/` directory.

## 📖 References & Credits

The implementations are based on guidance from renowned sources:

**Linear Regression & Neural Networks**:
- Sanjay.M: [Neural Net from Scratch using NumPy](https://towardsdatascience.com/neural-net-from-scratch-using-numpy-71a31f6e3675)
- Omar U. Florez: [One Lego at a Time - Explaining the Math of How Neural Networks Learn](https://medium.com/towards-artificial-intelligence/one-lego-at-a-time-explaining-the-math-of-how-neural-networks-learn-with-implementation-from-scratch-39144a1cf80)

**RNNs**:
- Josh Varty: [Visualizing RNNs](http://joshvarty.github.io/VisualizingRNNs/)
- Javaid Nabi: [Recurrent Neural Networks (RNNs)](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85)
- Faizan Shaikh: [Fundamentals of Deep Learning - RNNs](https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/)

## 💡 Key Concepts Covered

### Regression
- Linear regression with normal equations
- Cost functions and error metrics

### Classification
- Logistic regression with sigmoid activation
- KNN distance-based classification
- Neural network classification with decision boundaries

### Deep Learning
- Feedforward neural networks
- Backpropagation algorithm
- Recurrent neural networks and sequential processing

### Optimization
- Gradient descent variants
- Learning rate effects
- Convergence analysis
- Gradient clipping and numerical stability

## ✨ Highlights

- **From Scratch**: All algorithms implemented using only NumPy
- **Educational**: Well-commented code with mathematical explanations
- **Visual Learning**: Extensive plots and animations
- **Practical Examples**: Real datasets and realistic scenarios
- **Complete Pipeline**: Data generation → Model training → Evaluation → Visualization

## 📝 Notes

- Each notebook is self-contained and can be run independently
- Visualizations are automatically saved to the `scratch_mlp/plots/` directory
- Results may vary slightly due to random initialization (see notebooks for seed information)
- Some notebooks generate animated GIFs showing learning progression

## 🤝 Contributing

This is an educational repository. Feel free to fork, modify, and learn from the implementations.

## 📄 License

This project is for educational purposes.

---

**Last Updated**: January 2026