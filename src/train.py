import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report as cr, confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class TextClassificationPipeline:
    def __init__(
        self,
        df,
        text_column,
        target_columns,
        random_state=42,
        pca_components=2,
        model_to_train="xgb",
    ):
        self.df = df
        self.text_column = text_column
        self.target_columns = target_columns
        self.random_state = random_state
        self.pca_components = pca_components
        self.model_to_train = model_to_train
        self.spacy_model = spacy.load("en_core_web_sm")

    def split_data(self):
        # Split the data into train, validation, and test sets
        train_df, test_df = train_test_split(
            self.df, test_size=0.20, random_state=self.random_state
        )
        return train_df, test_df

    def clean_text(self, text):
        # Use Spacy to clean the text
        doc = self.spacy_model(text)
        cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        return cleaned_text

    def build_pipeline(self):
        # Create a Scikit-Learn pipeline
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("svd", TruncatedSVD(n_components=self.pca_components)),
            ]
        )

        return pipeline

    def train_xgboost_model(self, train_X, train_y):
        # Train an XGBoost model with cross-validation and grid search for best parameters
        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.1, 0.01, 0.001],
            "n_estimators": [100, 200, 300],
        }

        xgb = XGBClassifier()
        grid_search = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(train_X, train_y)
        best_xgb = grid_search.best_estimator_

        return best_xgb

    def train_svc_model(self, train_X, train_y):
        # Train an SVC model with cross-validation and grid search for best parameters
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],  # You can customize the gamma values
        }

        svc = SVC()
        grid_search = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(train_X, train_y)
        best_svc = grid_search.best_estimator_

        return best_svc

    def evaluate_model(self, model, test_X, test_y):
        # Generate classification report and confusion matrix
        y_pred = model.predict(test_X)
        report = cr(test_y, y_pred)
        matrix = confusion_matrix(test_y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"output/cm_{self.model_to_train}.png")
        return report, matrix

    def run_pipeline(self):
        # Split the data
        train_df, test_df = self.split_data()

        # Clean the text column
        train_df["cleaned_text"] = train_df[self.text_column].apply(self.clean_text)
        test_df["cleaned_text"] = test_df[self.text_column].apply(self.clean_text)

        # Build the pipeline
        pipeline = self.build_pipeline()

        # Fit and transform on the training data
        train_X = pipeline.fit_transform(train_df["cleaned_text"])
        train_X = np.concatenate(
            (train_X, train_df["group_id"].values.reshape(-1, 1)), axis=1
        )
        # Train an XGBoost model
        train_y = train_df[self.target_columns]
        if self.model_to_train == "xgb":
            best_xgb = self.train_xgboost_model(train_X, train_y)

            # Evaluate the model
            test_X = pipeline.transform(test_df["cleaned_text"])
            test_X = np.concatenate(
                (test_X, test_df["group_id"].values.reshape(-1, 1)), axis=1
            )
            test_y = test_df[self.target_columns]
            report, matrix = self.evaluate_model(best_xgb, test_X, test_y)

            return report, matrix
        else:
            best_svc = self.train_svc_model(train_X, train_y)
            # Evaluate the model
            test_X = pipeline.transform(test_df["cleaned_text"])
            test_X = np.concatenate(
                (test_X, test_df["group_id"].values.reshape(-1, 1)), axis=1
            )
            test_y = test_df[self.target_columns]
            report, matrix = self.evaluate_model(best_svc, test_X, test_y)

            return report, matrix
