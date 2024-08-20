from pyreadr import read_r
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data, verbose=True):
    #### preprocessing
    ## label encoding for categorical variables
    labelencoder = LabelEncoder()
    ### identify categorical variables
    cat_vars = data.select_dtypes(include=['object','category']).columns
    num_vars = data.select_dtypes(include=['float64']).columns

    if verbose:
        print(f'Found {len(cat_vars)} categorical variables and {len(num_vars)} numerical variables')
    assert len(cat_vars) + len(num_vars) == data.shape[1], f'All variables should be either categorical or numerical \n There are {data.shape[1]-len(cat_vars)-len(num_vars)} missing variables'

    data_preprocessed = data.copy()
    for col in cat_vars:
        data_preprocessed[col] = labelencoder.fit_transform(data[col])

    ## standard scaling for numerical variables
    scaler = StandardScaler()
    data_preprocessed[num_vars] = scaler.fit_transform(data[num_vars])

    return data_preprocessed

def save_pickle(obj, filename, verbose=True):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=-1)
        if verbose:
            print(f'Dumped PICKLE to {filename}')

def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        if verbose:
            print(f'Loaded PICKLE from {filename}')
        return obj

def load_process_save(path, verbose=False):
    data = read_r(path)
    data = data[None]

    data_preprocessed = preprocess_data(data)

    ## save preprocessed data
    path_out = path.replace('.rds', '.pkl').replace('R/', '')
    save_pickle(data_preprocessed, path_out)

    ### import and save DAG
    to_replace = path.split('_')[-1]
    dag_path = path.replace(to_replace, 'DAG.rds')
    dag = read_r(dag_path)
    dag = dag[None]
    save_pickle(dag, dag_path.replace('.rds', '.pkl').replace('R/', ''))


location_dict = {
    'mehra-complete': 'lingauss/medium/',
    'healthcare': 'lingauss/small/',
    'sangiovese': 'lingauss/small/',
    'ecoli70': 'gbn/medium/',
    'magic-niab': 'gbn/medium/',
    'magic-irri': 'gbn/large/',
    'arth150': 'gbn/verylarge/'
}

seed_list = [357,470,2743,4951,5088,5657,5852,6049,6659,9076]

for name in location_dict.keys():
    for seed in seed_list:
        path = f'{location_dict[name]}R/{name}_{seed}.rds'
        load_process_save(path, verbose=True)
        print(f'Processed {name} with seed {seed}')
    