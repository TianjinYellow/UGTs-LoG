import pandas as pd
import numpy as np
import pickle
import os

class PDLogger(object):
    def __init__(self, filename=None):
        if filename is not None:
            with open(filename, "rb") as f:
                self.dfs = pickle.load(f)
        else:
            self.dfs = dict()

        self.filename = filename

    def set_filename(self, filename):
        self.filename = filename

    def save(self):
        if self.filename is None:
            raise Exception
        else:
            with open(self.filename, "wb") as f:
                pickle.dump(self.dfs, f)

    def load(self):
        if self.filename is None:
            raise Exception
        else:
            if os.path.getsize(self.filename) <= 0:
                raise Exception
            with open(self.filename, "rb") as f:
                self.dfs = pickle.load(f)

    def add(self, attr, value, index=None, columns=None):
        if attr in self.dfs:
            df = self.dfs[attr]
            if index[0] in df.index:
                print(f'[PDLogger] Warning: The results are already set at index={index[0]}.')
                return

            if columns is None:
                columns = df.columns
            df_new = pd.DataFrame(value, index=index, columns=columns)
            self.dfs[attr] = df.append(df_new)
        else:
            self.dfs[attr] = pd.DataFrame(value, index=index, columns=columns)

    def get_df(self, attr):
        return self.dfs[attr]
