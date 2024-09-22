### README: Bug Attraction Prediction Analysis

#### **Project Overview**
This project involves building a predictive model using neural networks and ML Models including MLP, Extra Tree Regressor, KNN Regressor etc., to estimate the number of bugs attracted to various types of light sources. The analysis is performed using a dataset of light conditions and corresponding bug counts. To prevent overfitting, techniques such as dropout, L2 regularization (weight decay), and early stopping were applied.

#### **Objective**
The primary goal of this analysis is to train a machine learning model that can accurately predict bug attraction based on input features (related to light type and other conditions) and a single output variable (the total number of bugs attracted). 

#### **Dataset**
The dataset contains:
- **Input Features**: 32 variables related to the type of light and environmental conditions.
- **Output Variable**: Total bug attraction count for different light setups.

#### **File Structure**
- **Bug_attraction_data_modified.csv**: The dataset used for training and testing the model.
- **Bug_attraction.pbix**: Power BI file (for visualization purposes, not covered in this README).
- **Python Script**: Contains the neural network model, loss curve generation, and implementation of overfitting prevention techniques.

#### **Neural Network Architecture**
The neural network used in the analysis consists of 1 input layer, 4 hidden layers and an output layer, structured as follows:
- **Layer 1**: 256 neurons, ReLU activation, Batch Normalization.
- **Layer 2**: 128 neurons, ReLU activation.
- **Layer 3**: 64 neurons, ReLU activation.
- **Layer 4**: 32 neurons, ReLU activation.
- **Layer 5**: 16 neurons, ReLU activation.
- **Output Layer**: 1 neuron, linear output for predicting total bug attraction.

#### **Techniques to Reduce Overfitting**
1. **L2 Regularization (Weight Decay)**: Added to the Adam optimizer to penalize large weight values and improve model generalization.

#### **Model Training**
- The dataset is split into training, validation, and test sets. 
- During training, the model optimizes using the **Mean Squared Error (MSE)** loss function.
- The model performance is evaluated using the **R-squared (R2)** score on the test set.
- The training process monitors both training and validation losses to ensure the model doesn’t overfit.

#### **Loss Curve**
A loss curve is generated to visualize the reduction in training and validation losses over epochs. This provides a clear view of how well the model is learning and whether early stopping occurs when the validation loss stops improving.

#### **Model Performance**
The final model is evaluated using the test set, and its accuracy is measured using the **R2 Score**.


#### **How to Run the Code**
1. Load the `bug-attraction.csv` file into a Pandas DataFrame.
2. Preprocess the data using scaling techniques (StandardScaler in this case).
3. Run the neural network model in PyTorch.
4. Visualize the loss curve and evaluate the test set performance.

#### **Further Improvements**
- Experiment with different architectures (number of layers/neurons) to see if a more optimized structure improves performance.
- Test additional regularization techniques (e.g., L1 regularization) or different optimizers (e.g., RMSProp).
- Cross-validation techniques could be used for better model performance evaluation.

---

This README provides an overview of the dataset, modeling approach, and techniques used to mitigate overfitting.
