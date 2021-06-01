from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        
        self.X = X
        self.y = y
        

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe
        return self

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.trainedpipe = self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.trainedpipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse
        """evaluates the pipeline on df_test and return the RMSE"""


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    train1 = Trainer(X_train, y_train)
    evaluated = train1.evaluate(X_val, y_val)
    

