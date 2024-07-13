import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle, joblib, io
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ------------------------------------------------------- 
# General setting
# ------------------------------------------------------- 
st.set_page_config(page_title='Iris ML App', layout='wide')

# ------------------------------------------------------- 
# Sidebar
# ------------------------------------------------------- 
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](./data/Iris.csv)
""")

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['sqrt', 'log2', None])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

# ------------------------------------------------------- 
# Utils
# ------------------------------------------------------- 
def formating_pipeline(df):
    st.markdown('**1.2. Dataset Info**')
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    df = df.drop(columns = ['Id'])
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = pd.DataFrame(df[features])
    y = pd.DataFrame(df.drop(columns = features))

    st.markdown('**1.3. Checking Imbalance Data**')
    value_counts = y.value_counts()
    st.write(value_counts)

    y = y.Species.map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
    df = pd.concat([X, y], axis=1)
    return df

def build_model(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100)
    
    st.markdown('**1.4. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.4. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('y variable')
    st.info(y.name)

    # model
    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, y_train)

    st.subheader('2. Model Performance')
    st.markdown('**2.1. Training set**')
    y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(y_train, y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(y_train, y_pred_train) )

    st.markdown('**2.2. Test set**')
    y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(y_test, y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(y_test, y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

    return rf, X_train
    
def save_model(model, X_train):
    st.subheader('4. Save Model')
    model_format = st.selectbox('Select Model Format', ('joblib', 'pickle', 'onnx'))
    
    if model_format == 'joblib':
        model_file = io.BytesIO()
        joblib.dump(model, model_file)
        model_file.seek(0)
        st.download_button(
            label='Download Model as .joblib',
            data=model_file,
            file_name='model.joblib',
            mime='application/octet-stream'
        )
    elif model_format == 'pickle':
        model_file = io.BytesIO()
        pickle.dump(model, model_file)
        model_file.seek(0)
        st.download_button(
            label='Download Model as .pkl',
            data=model_file,
            file_name='model.pkl',
            mime='application/octet-stream'
        )
    elif model_format == 'onnx':
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        model_file = io.BytesIO()
        model_file.write(onnx_model.SerializeToString())
        model_file.seek(0)
        st.download_button(
            label='Download Model as .onnx',
            data=model_file,
            file_name='model.onnx',
            mime='application/octet-stream'
        )

# ------------------------------------------------------- 
# Main
# ------------------------------------------------------- 
st.write("""
# The Machine Learning App
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
Try adjusting the hyperparameters!
""")
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Preview**')
    st.write(df)
    
    # data transformation pipeline
    df = formating_pipeline(df)

    # Build model
    rf, X_train = build_model(df)

    # Save the modem
    save_model(rf, X_train)
else:
    st.info('Awaiting for CSV file to be uploaded.')