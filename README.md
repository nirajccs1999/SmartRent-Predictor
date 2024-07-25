# SmartRent Predictor

**SmartRent Predictor** is a machine learning-based project designed to accurately predict house rents using advanced regression models. Leveraging algorithms such as Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regressor, and XGBoost Regressor, this tool analyzes various features of a property to provide reliable rent estimates. By integrating multiple regression techniques, SmartRent Predictor ensures high accuracy and robust performance, making it a valuable asset for property management, real estate investments, and rental market analysis.

## Project Structure

- `data/`: Directory containing the dataset used for training and testing the models.
- `notebooks/`: Jupyter notebooks containing the code for each regression model.
- `results/`: Directory to store the results of the model predictions and evaluation metrics.
- `README.md`: This file.

## Dependencies

To run the code in this repository, you need the following dependencies:

- Python 3.x
- scikit-learn
- pandas
- numpy
- xgboost

You can install the required packages using pip:


pip install scikit-learn pandas numpy xgboost


## Models Implemented

1. Multiple Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor


## Usage

1. Clone this repository:

  git clone https://github.com/nirajccs1999/smartrent-predictor.git
  cd smartrent-predictor

2. Ensure you have the necessary dataset.
3. Run the Jupyter notebooks in the notebooks/ directory to train and evaluate each model.


## Result

## Example Code
Here's a brief overview of how to apply each regression model:

1. Decision Tree Regressor

      from sklearn.tree import DecisionTreeRegressor
      RegModel = DecisionTreeRegressor(max_depth=9, criterion='squared_error')
      DT = RegModel.fit(X_train, y_train)
      prediction = DT.predict(X_test)

Results:

R2 Value: 0.7271952880631326
Mean Accuracy on test data: 65.15666037621547
Median Accuracy on test data: 75.13636363636364
Accuracy values for 10-fold Cross Validation: [58.88099287, 59.62136038, 69.28826287, 71.60759692, 66.65153638, 58.21023951, 71.68405616, 72.49812822, 69.31749655, 69.12144499]
Final Average Accuracy of the model: 66.69





2. Random Forest Regressor

      from sklearn.ensemble import RandomForestRegressor
      RFModel = RandomForestRegressor(n_estimators=100, max_depth=9, criterion='squared_error', random_state=42)
      RF = RFModel.fit(X_train, y_train)
      prediction = RF.predict(X_test)


Results:

R2 Value: 0.7329273355079893
Mean Accuracy on test data: 65.76865824038788
Median Accuracy on test data: 74.575
Accuracy values for 10-fold Cross Validation: [60.53010159, 61.65438009, 69.68194371, 71.83717053, 67.48977392, 58.7514674, 70.37206035, 72.37114811, 69.00969762, 69.31832208]
Final Average Accuracy of the model: 67.1


## Evaluation Metrics
The models are evaluated using the following metrics:

    R2 Score
    Mean Absolute Percentage Error (MAPE)
    Median Absolute Percentage Error (Median APE)
    Cross-Validation Accuracy


## Contact
If you have any questions or need further assistance, feel free to contact me at:

Email: niraj919953@gmail.com

## Acknowledgments
Thanks to the developers of scikit-learn and XGBoost for providing excellent machine learning libraries.





