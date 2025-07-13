# Model Comparison - Function Reference

## Core Functions

### `pandas.read_csv()`
- **Purpose**: Load data from a CSV file into a pandas DataFrame
- **Parameters**:
  - `filepath_or_buffer`: Path to the CSV file or URL
  - `header`: Row number to use as column names
  - `names`: List of column names to use
- **Returns**: DataFrame containing the CSV data

### `DataFrame.values.ravel()`
- **Purpose**: Convert a DataFrame column to a 1D numpy array
- **Use Case**: Required by scikit-learn for target variables
- **Returns**: Flattened array of values

## Model Initialization

### `LogisticRegression()`
- **Purpose**: Implements logistic regression for binary classification
- **Key Parameters**:
  - `max_iter`: Maximum iterations for optimization
  - `random_state`: Seed for random number generation
- **Returns**: Untrained logistic regression model

### `RandomForestClassifier()`
- **Purpose**: Implements a random forest classifier
- **Key Parameters**:
  - `n_estimators`: Number of trees in the forest
  - `random_state`: Seed for reproducibility
- **Returns**: Untrained random forest model

## Model Training & Evaluation

### `model.fit(X, y)`
- **Purpose**: Train the model on input features and target
- **Parameters**:
  - `X`: Feature matrix (n_samples, n_features)
  - `y`: Target vector (n_samples,)
- **Returns**: Trained model (in-place modification)

### `model.predict(X)`
- **Purpose**: Make class predictions
- **Parameters**:
  - `X`: Feature matrix to predict on
- **Returns**: Array of predicted class labels

### `model.predict_proba(X)`
- **Purpose**: Predict class probabilities
- **Parameters**:
  - `X`: Feature matrix to predict on
- **Returns**: Array of probability estimates for each class

## Evaluation Metrics

### `accuracy_score(y_true, y_pred)`
- **Purpose**: Calculate accuracy classification score
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Range**: 0 to 1 (higher is better)

### `precision_score(y_true, y_pred)`
- **Purpose**: Calculate precision
- **Formula**: TP / (TP + FP)
- **Interpretation**: Ability to not label negative samples as positive

### `recall_score(y_true, y_pred)`
- **Purpose**: Calculate recall (sensitivity)
- **Formula**: TP / (TP + FN)
- **Interpretation**: Ability to find all positive samples

### `f1_score(y_true, y_pred)`
- **Purpose**: Calculate F1 score (harmonic mean of precision and recall)
- **Formula**: 2 * (precision * recall) / (precision + recall)
- **Use Case**: Balance between precision and recall

### `roc_auc_score(y_true, y_score)`
- **Purpose**: Compute Area Under the Receiver Operating Characteristic Curve
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Interpretation**: Probability that classifier ranks a random positive instance higher than a random negative one

## Cross-Validation

### `StratifiedKFold()`
- **Purpose**: K-Folds cross-validator with stratification
- **Key Parameters**:
  - `n_splits`: Number of folds (default=5)
  - `shuffle`: Whether to shuffle data before splitting
  - `random_state`: Random seed for reproducibility

### `cross_val_score()`
- **Purpose**: Evaluate a score by cross-validation
- **Key Parameters**:
  - `estimator`: The model to evaluate
  - `X`: Data to fit
  - `y`: Target variable
  - `cv`: Cross-validation strategy
  - `scoring`: Scoring metric ('accuracy', 'f1', 'roc_auc', etc.)

## Model Persistence

### `joblib.dump(model, filename)`
- **Purpose**: Save a Python object to disk
- **Parameters**:
  - `model`: The model to save
  - `filename`: Path where to save the model
- **Use Case**: Save trained models for later use

### `joblib.load(filename)`
- **Purpose**: Load a Python object from disk
- **Returns**: The loaded Python object
- **Use Case**: Load a previously saved model

## Data Structures

### `pandas.DataFrame`
- **Purpose**: 2D labeled data structure
- **Key Methods**:
  - `loc[]`: Access group of rows/columns by label
  - `iloc[]`: Access by integer position
  - `to_csv()`: Write to CSV file

### `numpy.array`
- **Purpose**: N-dimensional array object
- **Key Attributes**:
  - `shape`: Dimensions of the array
  - `dtype`: Data type of elements