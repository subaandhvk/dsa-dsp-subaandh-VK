import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from pathlib import Path
from xgboost import XGBRegressor

ROOT_DIR = Path('.').resolve().parents[1].absolute()
target_column = 'SalePrice'


def read_data(path):
    """
        path (str): the path to the data
        :returns datafrane
    """
    return pd.read_csv(path, index_col='Id')


def cont_preprocessing(df, to_drop):
    """
        df (pd.Dataframe): the continuous dataframe to be processes
        to_drop: the columns to be dropped
        :returns pd.datafrane
    """
    df = df.select_dtypes(include='number').copy()
    df = df.dropna()

    return df


def base_preprocessing(df, to_drop):
    """
        df (pd.Dataframe): the base dataframe to be processes
        to_drop: the columns to be dropped
        :returns pd.datafrane
    """

    df['LotFrontage'] = df['LotFrontage'].fillna(np.mean(df['LotFrontage']))
    df = df.dropna()
    df = drop_cols(df, to_drop)

    return df


def category_cont_preprocessing(df, to_drop):
    """
        df (pd.Dataframe): the final dataframe to be processes
        to_drop: the columns to be dropped
        :returns pd.datafrane
    """

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        df[col] = df[col].fillna('None')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')

    objects = df.select_dtypes(include='object').columns
    df.update(df[objects].fillna('None'))

    df['LotFrontage'] = df['LotFrontage'].fillna(np.median(df['LotFrontage']))

    numerics = df.select_dtypes(include='number').columns
    df.update(df[numerics].fillna(0))

    df['LotArea'] = df['LotArea'].fillna(np.mean(df['LotArea']))

    df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                               df['1stFlrSF'] + df['2ndFlrSF'])

    df['Total_Bathrooms'] = (df['FullBath'] + (df['HalfBath']) +
                             df['BsmtFullBath'] + (df['BsmtHalfBath']))

    df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                            df['EnclosedPorch'] + df['ScreenPorch'] +
                            df['WoodDeckSF'])

    df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    df = drop_cols(df, to_drop)

    category_cols = df.select_dtypes(include='object').columns
    one_hot = one_hot_encoder(df, category_cols)
    df = encode(df, category_cols, one_hot)

    return df


def train_preprocessing(df, to_drop, type):
    """
        df (pd.Dataframe): the dataframe to be processes
        to_drop: the columns to be dropped
        type : the type of processing to perform 1 - continuous 2 - base 3- final
        :returns datafrane
    """
    if type == 1:
        return cont_preprocessing(df, to_drop)
    elif type == 2:
        return base_preprocessing(df, to_drop)
    elif type == 3:
        return category_cont_preprocessing(df, to_drop)


def drop_cols(df, to_drop):
    """
        df (pd.Dataframe): the dataframe to be processes
        to_drop: the columns to be dropped
        :returns datafrane
    """
    df = df.drop(columns=to_drop, axis=1)
    df = df.dropna()

    return df


def one_hot_encoder(df, categorical_cols):
    """
        df (pd.Dataframe): the dataframe to be processes
        categorical_cols : the columns to be fitted for one hot encoder
        :returns encoded values
    """
    one_hot = OneHotEncoder(handle_unknown='ignore')
    one_hot = one_hot.fit(df[categorical_cols])
    dump(one_hot, ROOT_DIR / 'models' / 'onehot.pkl')

    return one_hot


def encode(df, categorical_cols, one_hot):
    """
        df (pd.Dataframe): the dataframe to be processes
        categorical_cols : the columns to be fitted for one hot encoder
        one_hot : fitted one hot encoder
        :returns pd.Dataframe
    """

    oh_df = one_hot.transform(df[categorical_cols]).toarray()
    categ_col_name = one_hot.get_feature_names(categorical_cols)

    # Convert it to df
    oh_df = pd.DataFrame(oh_df, index=df.index, columns=categ_col_name)

    # Extract only the columns that didnt need to be encoded
    df_without_categ = df.drop(columns=categorical_cols)

    # Concatenate the two dataframes :
    df = pd.concat([oh_df, df_without_categ], axis=1)

    return df


def split_data(df):
    """
        df (pd.Dataframe): the dataframe to be processes
        :returns X, y and the splitted X and y train and test values
    """

    # Split dependant and independant variables
    X, y = df.drop(target_column, axis=1), df[target_column]

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X, y, X_train, X_test, y_train, y_test


def linear_model(df, X_train, y_train):
    """
        df (pd.Dataframe): the dataframe to be processes
        X_train, Y_train : the train and test data to be fit
        :returns model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def xgb_model(X_train, y_train):
    """
        df (pd.Dataframe): the dataframe to be processes
        X_train, Y_train : the train and test data to be fit
        :returns model
    """
    model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3)
    model.fit(X_train, y_train)

    return model


def ols(df):
    """
        df (pd.Dataframe): the dataframe to be processes
        :returns None,Just prints the summary
    """
    X, y = df.drop(target_column, axis=1), df[target_column]
    X_sm = sm.add_constant(X)
    ols_reg = sm.OLS(y, X_sm).fit()
    print(ols_reg.summary())


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    """
        y_test: the test data
        y_pred : the predicted data
        :returns float
    """

    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def predict(model, X_test):
    """
        model: the model to be predicted
        X_test : the test data to predict
        :returns predictions as np array
    """

    y_pred = model.predict(X_test)
    # Replace negative predictions with 0
    y_pred = np.where(y_pred < 0, 0, y_pred)

    return y_pred

def cont_inference(inference_df):
    """
        inference_df: the inference data
        :returns pd.Dataframe
    """
    inference_df = inference_df.select_dtypes(include='number')
    inference_df = inference_df.dropna()

    return inference_df


def baseline_inference(inference_df, to_drop):
    """
        inference_df: the inference data
        to_drop : the columns to be dropped
        :returns pd.Dataframe
    """

    inference_df = inference_df.select_dtypes(include='number')

    drop = []

    for col in inference_df.columns:
        if col in to_drop:
            drop.append(col)

    inference_df = inference_df.drop(columns=drop)
    inference_df['LotFrontage'] = inference_df.fillna(np.mean(inference_df['LotFrontage']))
    inference_df = inference_df.dropna()

    return inference_df


def category_cont_if_preprocessing(inference_df):
    """
        inference_df: the inference data
        :returns pd.Dataframe
    """
    for col in ('GarageArea', 'GarageCars'):
        inference_df[col] = inference_df[col].fillna(0)

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        inference_df[col] = inference_df[col].fillna('None')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        inference_df[col] = inference_df[col].fillna('None')

    objects = inference_df.select_dtypes(include='object').columns
    inference_df.update(inference_df[objects].fillna('None'))

    inference_df['LotFrontage'] = inference_df['LotFrontage'].fillna(np.median(inference_df['LotFrontage']))

    numerics = inference_df.select_dtypes(include='number').columns
    inference_df.update(inference_df[numerics].fillna(0))

    inference_df['LotArea'] = inference_df['LotArea'].fillna(np.mean(inference_df['LotArea']))

    inference_df['YrBltAndRemod'] = inference_df['YearBuilt'] + inference_df['YearRemodAdd']
    inference_df['TotalSF'] = inference_df['TotalBsmtSF'] + inference_df['1stFlrSF'] + inference_df['2ndFlrSF']

    inference_df['Total_sqr_footage'] = (inference_df['BsmtFinSF1'] + inference_df['BsmtFinSF2'] +
                                         inference_df['1stFlrSF'] + inference_df['2ndFlrSF'])

    inference_df['Total_Bathrooms'] = (inference_df['FullBath'] + (inference_df['HalfBath']) +
                                       inference_df['BsmtFullBath'] + (inference_df['BsmtHalfBath']))

    inference_df['Total_porch_sf'] = (inference_df['OpenPorchSF'] + inference_df['3SsnPorch'] +
                                      inference_df['EnclosedPorch'] + inference_df['ScreenPorch'] +
                                      inference_df['WoodDeckSF'])

    inference_df['haspool'] = inference_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    inference_df['has2ndfloor'] = inference_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    inference_df['hasgarage'] = inference_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    inference_df['hasbsmt'] = inference_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    inference_df['hasfireplace'] = inference_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    return inference_df


def category_cont_inference(inference_df, to_drop):
    """
        inference_df: the inference data
        to_drop : the columns to be dropped
        :returns pd.Dataframe
    """

    one_hot = load(ROOT_DIR / 'models' / 'onehot.pkl')

    drop = []

    for col in inference_df.columns:
        if col in to_drop:
            drop.append(col)

    inference_df = category_cont_if_preprocessing(inference_df)
    inference_df = inference_df.drop(columns=drop)
    inference_df = inference_df.dropna()
    imp_inference_categ = inference_df.select_dtypes(include='object').columns

    inference_df = encode(inference_df, imp_inference_categ, one_hot)

    return inference_df


def inference(inference_df, to_drop):
    pass


def predict_inference(df, model):
    """
        df: the inference data
        model : the trained model
        :returns pd.Dataframe
    """
    return model.predict(df)


def fill_negatives(predictions):
    """
        predictions: the inference data
        :returns array with negative values filled
    """
    # Check if there is negative predictions, if so get their indexs
    np.where(predictions < 0)

    # Replace negative predictions with 0
    predictions = np.where(predictions < 0, 0, predictions)
    np.where(predictions < 0)

    return predictions


def prepare_submission(inference_df, changed_df, predictions):
    """
        inference_df: the inference data
        changed_df: the processed/changed data to be submitted
        predictions: the predicted data
        :returns pd.Dataframe, id_array
    """

    # Assign predictions to target column
    changed_df[target_column] = predictions

    # Reset the indexes
    changed_df = changed_df[[target_column]].reset_index()
    ids_df = inference_df.reset_index()[['Id']]

    # Check the number of missing predictions
    print('Number of missing predictions', {len(ids_df) - len(changed_df)})

    # Merge the dataset
    return ids_df.merge(changed_df, on='Id', how='left'), inference_df, ids_df


def validate_missing_predictions(submission_df, df_final, final_id):
    """
        submission_df: the submission data with missing values to be filled
        df_final: the final data to be submitted
        final_id: the id's of submission data data
        :returns pd.Dataframe with the missing predictions addressed
    """
    # Validate the number of missing predictions
    submission_df[target_column].isna().sum() == len(final_id) - len(df_final)

    # Fill null values with 0
    submission_df[target_column] = submission_df[target_column].fillna(0)

    return submission_df
