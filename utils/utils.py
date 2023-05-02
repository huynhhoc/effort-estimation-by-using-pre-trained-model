"""
Written by Huynh Thai Hoc
April 26, 2023

Feel free to use and modify this program as you see fit.
"""
import pandas as pd
import numpy as np
def savetofile(Xtest, ytest, ypred, features, filename):
    try:
        ypred = ypred.numpy()
        y_test_np = ytest.reshape(-1, 1)
        y_pred_np = ypred.reshape(-1, 1)
        X_test_np = Xtest.numpy().reshape(-1, Xtest.shape[-1])
        data = np.concatenate([y_test_np, y_pred_np, X_test_np], axis=1)
        df = pd.DataFrame(data=data, columns=['y_test', 'y_pred'] + features)
        df.to_csv(filename, index=False)
        print(filename + " has been saved successfully!")
    except Exception as ex:
        print(ex)