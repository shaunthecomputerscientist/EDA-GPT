import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.preprocessed_data = None

    def clean_data(self, num_columns_to_impute, cat_columns_to_impute, num_imputation_strategy, cat_imputation_strategy, columns_to_encode):
        # Impute missing values for numerical columns
        if num_columns_to_impute:
            num_imputer = SimpleImputer(strategy=num_imputation_strategy)
            self.data[num_columns_to_impute] = num_imputer.fit_transform(self.data[num_columns_to_impute])
        
        # Impute missing values for categorical columns
        if cat_columns_to_impute:
            cat_imputer = SimpleImputer(strategy=cat_imputation_strategy)
            self.data[cat_columns_to_impute] = cat_imputer.fit_transform(self.data[cat_columns_to_impute])
        
        # Encode categorical data
        if columns_to_encode:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(self.data[columns_to_encode])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))
            self.data = self.data.drop(columns_to_encode, axis=1)
            self.data = pd.concat([self.data, encoded_df], axis=1)
        
        self.preprocessed_data = self.data

    def preprocess_data(self, columns_to_normalize, columns_to_standardize, columns_to_robust_scale):
        # Normalize data
        if columns_to_normalize:
            normalizer = MinMaxScaler()
            self.data[columns_to_normalize] = normalizer.fit_transform(self.data[columns_to_normalize])
        
        # Standardize data
        if columns_to_standardize:
            scaler = StandardScaler()
            self.data[columns_to_standardize] = scaler.fit_transform(self.data[columns_to_standardize])
        
        # Robust scale data
        if columns_to_robust_scale:
            robust_scaler = RobustScaler()
            self.data[columns_to_robust_scale] = robust_scaler.fit_transform(self.data[columns_to_robust_scale])
        
        self.preprocessed_data = self.data

    def get_preprocessed_data(self):
        return self.preprocessed_data

class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def apply_feature_engineering(self, feature_engineering_option, target, n_components_pca=None, k_best=None, degree=None):
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        
        if feature_engineering_option == 'PCA' and n_components_pca:
            pca = PCA(n_components=n_components_pca)
            X = pca.fit_transform(X)
            result = {"pca components": [pca.components_], "explained variance": [pca.explained_variance_], "explained variance ratio": [pca.explained_variance_ratio_]}
            st.write(result)
        elif feature_engineering_option == 'SelectKBest' and k_best:
            selector = SelectKBest(score_func=f_classif, k=k_best)
            X = selector.fit_transform(X, y)
        elif feature_engineering_option == 'Polynomial' and degree:
            poly = PolynomialFeatures(degree=degree)
            X = poly.fit_transform(X)
        
        self.data = pd.DataFrame(X)
        self.data[target] = y

    def get_engineered_data(self):
        return self.data

class Automl:
    def __init__(self, data):
        self.data = data
        self.ml_data = data

    def preprocess(self, num_columns_to_impute, cat_columns_to_impute, num_imputation_strategy, cat_imputation_strategy, columns_to_normalize, columns_to_standardize, columns_to_robust_scale, columns_to_encode):
        preprocessor = DataPreprocessor(self.data)
        preprocessor.clean_data(num_columns_to_impute, cat_columns_to_impute, num_imputation_strategy, cat_imputation_strategy, columns_to_encode)
        preprocessor.preprocess_data(columns_to_normalize, columns_to_standardize, columns_to_robust_scale)
        self.ml_data = preprocessor.get_preprocessed_data()

    def feature_engineer(self, feature_engineering_option, target, n_components_pca=None, k_best=None, degree=None):
        feature_engineer = FeatureEngineer(self.ml_data)
        feature_engineer.apply_feature_engineering(feature_engineering_option, target, n_components_pca, k_best, degree)
        self.ml_data = feature_engineer.get_engineered_data()

    def main(self):
        st.title("AutoML Data Preprocessing and Feature Engineering")
        
        if self.data is not None:
            data = self.data
            st.write("Uploaded Data:")
            st.write(data.head())
            
            # Data Preprocessing Inputs
            st.header("Data Preprocessing")
            num_columns_to_impute = st.multiselect("Numerical Columns to Impute", data.select_dtypes(include='number').columns)
            cat_columns_to_impute = st.multiselect("Categorical Columns to Impute", data.select_dtypes(include='object').columns)
            num_imputation_strategy = st.selectbox("Numerical Imputation Strategy", ["mean", "median", "most_frequent"])
            cat_imputation_strategy = st.selectbox("Categorical Imputation Strategy", ["most_frequent", "constant"])
            columns_to_normalize = st.multiselect("Columns to Normalize", data.columns)
            columns_to_standardize = st.multiselect("Columns to Standardize", data.columns)
            columns_to_robust_scale = st.multiselect("Columns to Robust Scale", data.columns)
            columns_to_encode = st.multiselect("Columns to Encode", data.select_dtypes(include='object').columns)
            
            # Feature Engineering Inputs
            st.header("Feature Engineering")
            feature_engineering_option = st.selectbox("Feature Engineering Option", ["None", "PCA", "SelectKBest", "Polynomial"])
            target = st.selectbox("Target Variable", data.columns)
            
            n_components_pca = None
            k_best = None
            degree = None
            
            if feature_engineering_option == "PCA":
                n_components_pca = st.slider("Number of PCA Components", 1, len(data.columns)-1)
            elif feature_engineering_option == "SelectKBest":
                k_best = st.slider("Number of K Best Features", 1, len(data.columns)-1)
            elif feature_engineering_option == "Polynomial":
                degree = st.slider("Degree of Polynomial Features", 2, 5)
            
            if st.button("Run AutoML Pipeline"):
                automl = Automl(data)
                automl.preprocess(num_columns_to_impute, cat_columns_to_impute, num_imputation_strategy, cat_imputation_strategy, columns_to_normalize, columns_to_standardize, columns_to_robust_scale, columns_to_encode)
                automl.feature_engineer(feature_engineering_option, target, n_components_pca, k_best, degree)
                
                st.write("Processed Data:")
                st.data_editor(automl.ml_data)