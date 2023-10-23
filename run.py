import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.train import TextClassificationPipeline

# df is your DataFrame, text_column is the column with text data, and target_columns is the categorical column.
df = pd.read_csv("data/nlp_data_scientist_challenge_dataset.csv")
df["is_consultancy"] = df["is_consultancy"].astype("int")
# Create a LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical variable
df["group_id"] = label_encoder.fit_transform(df["group_id"])
df_modified = df.drop(["company_id", "predicted_consultancy"], axis=1)
text_pipeline = TextClassificationPipeline(
    df, "text", "is_consultancy", model_to_train="xgb", pca_components=2
)
classification_report, confusion_matrix = text_pipeline.run_pipeline()
print("-----------------XGB Results-----------------------------")
print(classification_report)
print("-----------------XGB Results Finished-----------------------------")


print("-------------Train SVC model-----------------------")
# df is your DataFrame, text_column is the column with text data, and target_columns is the categorical column.
df2 = pd.read_csv("data/nlp_data_scientist_challenge_dataset.csv")
df2["is_consultancy"] = df2["is_consultancy"].astype("int")
# Create a LabelEncoder
label_encoder2 = LabelEncoder()

# Fit and transform the categorical variable
df2["group_id"] = label_encoder2.fit_transform(df["group_id"])
df_modified2 = df2.drop(["company_id", "predicted_consultancy"], axis=1)
text_pipeline2 = TextClassificationPipeline(
    df, "text", "is_consultancy", model_to_train="svc", pca_components=2
)
classification_report2, confusion_matrix2 = text_pipeline2.run_pipeline()
print("-----------------XGB Results-----------------------------")
print(classification_report2)
print("-----------------XGB Results Finished-----------------------------")
