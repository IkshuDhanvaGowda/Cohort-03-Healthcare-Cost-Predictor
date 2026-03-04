
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel


def analyze_correlation_impact(df, feature, target="charges"):
    """
    Returns correlation between a feature and target variable.
    """
    return df[feature].corr(df[target])


def correlation_feature_selection(df, target="charges", threshold=0.1):
    """
    Select features based on absolute correlation with target.
    """
    corr_matrix = df.corr(numeric_only=True)
    target_corr = corr_matrix[target].abs()

    selected = target_corr[target_corr > threshold].index.tolist()

    if target in selected:
        selected.remove(target)

    return selected


def model_based_feature_selection(X, y):
    """
    Select features using Linear Regression coefficients.
    """
    model = LinearRegression()
    model.fit(X, y)

    selector = SelectFromModel(model, prefit=True, threshold="mean")
    selected_features = X.columns[selector.get_support()].tolist()

    return selected_features


def get_bmi_age_impact(df):
    """
    Returns correlation impact of BMI and Age on charges.
    """
    bmi_corr = analyze_correlation_impact(df, "bmi")
    age_corr = analyze_correlation_impact(df, "age")

    return {
        "bmi_correlation": bmi_corr,
        "age_correlation": age_corr
    }


def final_feature_selection(df, target="charges", corr_threshold=0.1):
    """
    Final feature selection using intersection of correlation
    and model-based methods.
    """
    X = df.drop(columns=[target])
    y = df[target]

    corr_features = correlation_feature_selection(df, target, corr_threshold)
    model_features = model_based_feature_selection(X, y)

    final_features = list(set(corr_features).intersection(model_features))

    return final_features
