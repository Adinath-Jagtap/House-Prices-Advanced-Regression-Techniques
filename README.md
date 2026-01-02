# 🏡 House Prices - Advanced Regression Techniques

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Predicting residential home prices in Ames, Iowa using advanced regression techniques**

[Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) • [Kaggle Notebook](https://www.kaggle.com/code/adinathjagtap777/house-prices-advanced-regression-techniques) • [Documentation](#-features)

</div>

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Public Leaderboard** | **0.12029** |
| Evaluation Metric | RMSE (Log Scale) |
| Model Architecture | Stacking Ensemble |

---

## 🎯 Project Overview

This repository contains a comprehensive solution for the **House Prices: Advanced Regression Techniques** competition on Kaggle. The project implements sophisticated feature engineering techniques and ensemble modeling to predict residential home prices with high accuracy.

### Competition Context

With **79 explanatory variables** describing various aspects of residential homes in Ames, Iowa, this competition challenges participants to predict the final sale price of each property. The dataset offers a modernized alternative to the classic Boston Housing dataset, providing rich opportunities for creative feature engineering and advanced modeling techniques.

---

## 🚀 Key Features

### Feature Engineering
- **YrBltAndRemod**: Combined year built and remodel information
- **TotalSF**: Total square footage across all floors
- **Total_sqr_footage**: Comprehensive basement and floor area calculation
- **Total_Bathrooms**: Weighted bathroom count (full + 0.5 × half baths)
- **Total_porch_sf**: Aggregate porch and deck square footage

### Data Processing
- **Intelligent Missing Value Handling**: Neighborhood-based imputation for LotFrontage
- **Categorical Encoding**: OneHotEncoder with drop='first' to avoid multicollinearity
- **Comprehensive Preprocessing**: Separate handling for numeric and categorical features

### Model Architecture

```
┌─────────────────────────────────────┐
│       Stacking Regressor            │
├─────────────────────────────────────┤
│  Base Models:                       │
│  • Ridge Regression (α=15)          │
│  • XGBoost Regressor (tuned)        │
├─────────────────────────────────────┤
│  Meta-Model:                        │
│  • Linear Regression                │
└─────────────────────────────────────┘
```

#### XGBoost Hyperparameters
```python
{
    'max_depth': 4,
    'learning_rate': 0.00875,
    'n_estimators': 3515,
    'min_child_weight': 2,
    'colsample_bytree': 0.205,
    'subsample': 0.404,
    'reg_alpha': 0.330,
    'reg_lambda': 0.046
}
```

---

## 📁 Repository Structure

```
House-Prices-Advanced-Regression-Techniques/
│
├── house-prices-advanced-regression-techniques.ipynb
│   └── Complete analysis and model training notebook
│
├── submission.csv
│   └── Best scoring predictions (Public Score: 0.12029)
│
├── README.md
│   └── Project documentation
│
└── data/
    ├── train.csv (provided by Kaggle)
    ├── test.csv (provided by Kaggle)
    ├── sample_submission.csv (provided by Kaggle)
    └── data_description.txt (provided by Kaggle)
```

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Feature Engineering** | OneHotEncoder, Custom transformations |
| **Model Ensemble** | StackingRegressor |

---

## 📈 Methodology

### 1. Data Preprocessing
- Load training and test datasets
- Combine datasets for consistent preprocessing
- Handle missing values with domain-specific strategies
- Create engineered features

### 2. Feature Engineering
```python
# Example: Total square footage
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Example: Total bathrooms (weighted)
df['Total_Bathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                          df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
```

### 3. Encoding
- OneHotEncoder for categorical variables
- Drop first category to prevent multicollinearity
- Preserve numeric features as-is

### 4. Model Training
- Ridge Regression with L2 regularization (α=15)
- XGBoost with optimized hyperparameters
- Stacking ensemble with Linear Regression as meta-model
- Target transformation using log1p for better prediction

### 5. Prediction
- Generate predictions on test set
- Apply exponential transformation to reverse log scale
- Create submission file in required format

---

## 🔧 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Adinath-Jagtap/House-Prices-Advanced-Regression-Techniques.git
cd House-Prices-Advanced-Regression-Techniques
```

2. **Download the dataset**
- Visit the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Download `train.csv`, `test.csv`, `sample_submission.csv`, and `data_description.txt`
- Place files in the `data/` directory

3. **Run the notebook**
```bash
jupyter notebook house-prices-advanced-regression-techniques.ipynb
```

4. **Generate predictions**
The notebook will automatically create `submission.csv` with predictions

---

## 📊 Model Performance Analysis

### Evaluation Metric
**Root Mean Squared Error (RMSE)** on logarithmic scale:

```
RMSE = sqrt(mean((log(predicted) - log(actual))²))
```

This metric ensures errors in predicting expensive and inexpensive houses are weighted equally.

### Performance Breakdown
- **Ridge Regression**: Provides stable baseline with regularization
- **XGBoost**: Captures non-linear relationships and interactions
- **Stacking**: Combines strengths of both models for optimal performance

---

## 🎓 Key Learnings

### Feature Engineering
- Domain knowledge significantly improves prediction accuracy
- Combining related features often creates more predictive variables
- Proper handling of missing values is crucial for model performance

### Model Selection
- Ensemble methods consistently outperform individual models
- Regularization prevents overfitting on high-dimensional data
- Hyperparameter tuning is essential for XGBoost performance

### Competition Strategy
- Log transformation of target variable improves RMSE
- Stacking different model types captures diverse patterns
- Careful preprocessing ensures consistent train/test predictions

---

## 🔍 Future Improvements

- [ ] Implement cross-validation for robust performance estimation
- [ ] Explore additional feature interactions
- [ ] Test alternative ensemble methods (e.g., Voting, Blending)
- [ ] Incorporate neural network models
- [ ] Add feature selection techniques
- [ ] Implement automated hyperparameter tuning (Optuna, GridSearchCV)

---

## 📚 References

### Competition Resources
- [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [Kaggle Notebook](https://www.kaggle.com/code/adinathjagtap777/house-prices-advanced-regression-techniques)

### Documentation
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

### Dataset
The Ames Housing dataset was compiled by Dean De Cock for data science education as a modernized alternative to the Boston Housing dataset.

---

## 👤 Author

**Adinath Jagtap**

- GitHub: [@Adinath-Jagtap](https://github.com/Adinath-Jagtap)
- Kaggle: [@adinathjagtap777](https://www.kaggle.com/adinathjagtap777)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the competition and providing the platform
- **Dean De Cock** for compiling the Ames Housing dataset
- **DataCanary** and **Anna Montoya** for competition organization
- The Kaggle community for valuable discussions and insights

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- Open an [issue](https://github.com/Adinath-Jagtap/House-Prices-Advanced-Regression-Techniques/issues)
- Connect on [Kaggle](https://www.kaggle.com/adinathjagtap777)

---

<div align="center">

**If you find this project helpful, please consider giving it a ⭐!**

Made with ❤️ for the Data Science Community

</div>
