# -----------------------------
# Import modules
# -----------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

# -----------------------------
# Upload datasets
# -----------------------------

data = pd.read_csv("preprocessed_data.csv")
data1 = pd.read_csv("StudentsPerformance.csv")


# -----------------------------
# Feature importance
# -----------------------------

X = data.drop(columns=['math score', 'math_score_binary', 'math_original'])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)


# FIT and TRANSFORM
X_processed = preprocessor.fit_transform(X)

# Numeric features
numeric_features = numeric_cols
feature_names = numeric_features.copy()

# Categorical features
if categorical_cols:
    ohe = preprocessor.named_transformers_['cat']  # fitted encoder
    ohe_features = ohe.get_feature_names_out()     # just call without argument
    feature_names += list(ohe_features)

y = LabelEncoder().fit_transform(data['math_score_binary'])

# Identify categorical vs numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()  # gender, race, parental, lunch, test prep
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()      # probably empty

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Fit and transform
X_processed = preprocessor.fit_transform(X)

# Feature names
numeric_features = numeric_cols
feature_names = numeric_features.copy()
if categorical_cols:
    ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    feature_names += list(ohe_features)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Logistic Regression
# -----------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# -----------------------------
# Create Plotly figures
# -----------------------------
# 1️⃣ Box plot
fig_box = px.box(
    data,
    x='lunch',
    y='math_original',
    color='lunch',
    labels={'lunch':'Lunch Type', 'math_original':'Math Score'},
    title='Math Scores by Lunch Type'
)
for trace in fig_box.data:
    trace.showlegend = False  # hide box legend

# 2️⃣ Scatter plot
fig_scatter = px.scatter(
    data1,
    x='reading score',
    y='math score',
    color='test preparation course',
    trendline='ols',
    labels={'reading score':'Reading Score', 'math score':'Math Score', 'test preparation course':'Test Prep'},
    title='Math vs Reading Scores by Test Prep'
)

# 3️⃣ Feature importance
feat_df = pd.DataFrame({
    'feature': feature_names,
    'importance': abs(log_reg.coef_[0])
}).sort_values(by='importance', ascending=False)

fig_feat = px.bar(
    feat_df,
    x='feature',
    y='importance',
    labels={'feature':'Feature','importance':'Coefficient Magnitude'},
    title='Feature Importance'
)

# -----------------------------
# Initialize Dash app
# -----------------------------
app = dash.Dash(__name__)

# -----------------------------
# Layout
# -----------------------------
app.layout = html.Div([
    html.H1("Student Performance Dashboard",
    style={
        'textAlign': 'center',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '36px',
        'color': 'black'
    }),

    html.Div([
        dcc.Graph(figure=fig_box)
    ], style={'margin-bottom':'50px'}),

    html.Div([
        dcc.Graph(figure=fig_scatter)
    ], style={'margin-bottom':'50px'}),

    html.Div([
        dcc.Graph(figure=fig_feat)
    ])
])

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)




