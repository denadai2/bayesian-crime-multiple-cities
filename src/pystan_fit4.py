import pystan
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import time
import argparse
import arviz as az
import csv
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error
import hashlib
from io import BytesIO
from PIL import Image
import pickle
import os
from scipy.sparse.csgraph import minimum_spanning_tree
matplotlib.use('Agg')


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Launch MCMC computation for crime"
    )
    parser.add_argument('--iterations', '-I',
                        default=15000, type=int)
    parser.add_argument('--tuningsteps', '-T',
                        default=10000, type=int)
    parser.add_argument('--njobs', '-J',
                        default=-1, type=int,
                        help='Number of parallel processes')
    parser.add_argument('--city', '-C',
                        help='City', default=None)
    parser.add_argument('--spatial_name', '-S',
                        help='Spatial name for the spatial_groups', default='ego')
    parser.add_argument('--dependent', '-D',
                        help='Depedent variable (crime/people ...)', default='ncrimes', choices=['ncrimes',
                                                                                                 'ambient',
                                                                                                 'Property crime',
                                                                                                 'Violent crime'])
    parser.add_argument('--removeprogressbar', '-R',
                        help='Progressbar remove?', default=None)
    parser.add_argument('--model', '-M',
                        help='Name of the model to be used', default='BSF', choices=['nb', 'ESF', 'REESF', 'REESF-a', 'BSF', 'BSF_ODall'])
    parser.add_argument('--normalization', '-Y',
                        help='Normalization for the mobility matrix', default=None,
                        choices=[None, 'pop', 'eff'])
    # Brokmann normalization by outgoing fluxes from A
    # pop = average number of trips between ABBA

    parser.add_argument('--test', nargs='+', help='Test features')

    parser.add_argument('--sd', dest='sd', action='store_true')
    parser.add_argument('--no-sd', dest='sd', action='store_false')
    parser.add_argument('--uf', dest='uf', action='store_true')
    parser.add_argument('--no-uf', dest='uf', action='store_false')
    parser.add_argument('--m', dest='m', action='store_true')
    parser.add_argument('--no-m', dest='m', action='store_false')
    parser.add_argument('--full', dest='full', action='store_true')
    parser.add_argument('--no-full', dest='full', action='store_false')
    parser.add_argument('--core-only', dest='core', action='store_true')
    parser.add_argument('--no-core-only', dest='core', action='store_false')
    parser.add_argument('--minimal', dest='minimal', action='store_true')
    parser.add_argument('--no-minimal', dest='minimal', action='store_false')
    parser.add_argument('--od-m', dest='ODm', action='store_true')
    parser.add_argument('--no-od-m', dest='ODm', action='store_false')
    parser.add_argument('--od-d', dest='ODd', action='store_true')
    parser.add_argument('--no-od-d', dest='ODd', action='store_false')
    parser.add_argument('--od-md', dest='ODmd', action='store_true')
    parser.add_argument('--no-od-md', dest='ODmd', action='store_false')
    parser.add_argument('--CV', dest='CV', action='store_true')
    parser.add_argument('--no-CV', dest='CV', action='store_false')

    parser.set_defaults(sd=False, uf=False, m=False, full=False, core=False, ODm=False, ODd=False, CV=False, test=False, minimal=False, ODmd=False)
    return parser


def make_model_name(args):
    name = ['full']
    if not args.full and not args.minimal:
        name = []
        if args.core:
            name.append('core')
        else:
            if args.sd:
                name.append('sd')
            if args.uf:
                name.append('uf')
            if args.m:
                name.append('m')
    if args.minimal:
        name = ['minimal']
    if args.ODm:
        name.append('ODm')
        if args.normalization:
            name.append(args.normalization)
    elif args.ODd:
        name.append('ODd')
    elif args.ODmd:
        name.append('ODmd')
    name = '_'.join(name)
    model_name = '{test}{city}_{spatial_name}_{name}_{iterations}_{dependent}'.format(test=('-'.join(args.test))+'_' if args.test else '',
                                                                                       iterations=args.iterations,
                                                                                name=name,
                                                                                city='all' if args.city is None else args.city,
                                                                                dependent=args.dependent,
                                                                                spatial_name=args.spatial_name)
    return model_name


def median_absolute_p_error(y_true, y_pred, axis=None):
    return np.median(np.abs(100 * (y_pred - y_true) / y_true), axis=axis)


def mean_squared_log_error(y_true, y_pred):
    return np.mean(np.power((np.log(y_pred + 1) - np.log(y_true + 1)), 2))


def savefig(plt, filename):
    ram = BytesIO()
    plt.savefig(ram, format='png', bbox_inches='tight')
    ram.seek(0)
    im = Image.open(ram)
    im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
    im2.save(filename, format='PNG')


def load_data(args, CITY=None):
    # MERGED FEATURES
    data_df = pd.read_csv('../data/generated_files/merged_features.csv', dtype={
            'sp_id': str
        })
    data_df = data_df[data_df['spatial_name'] == args.spatial_name]

    if CITY:
        print("chosen city", CITY)
        data_df = data_df[data_df.city == CITY]

    data_df = data_df.sort_values(['city', 'sp_id']).reset_index(drop=True)

    # SPATIAL WEIGHTS
    W2 = None
    if args.ODd:
        fname = '../data/generated_files/spatial_dmatrix.parquet'
        filesize = os.path.getsize(fname)
        hash_md5 = hashlib.md5(str(filesize).encode('utf-8')).hexdigest()
        cache_name = '../cache/{city}_{spatial_name}_{hex}.npz'.format(city=CITY, hex=hash_md5, spatial_name=args.spatial_name)
        if os.path.exists(cache_name):
            W = np.load(cache_name)['W']
        else:
            weights_df = pd.read_parquet(fname)
            weights_df = weights_df[(weights_df['spatial_name'] == args.spatial_name)]
            weights_df = weights_df.sort_values(['city', 'o_sp_id'])
            weights_df['o'] = weights_df.city + '_' + weights_df.o_sp_id.astype(str)
            weights_df['d'] = weights_df.city + '_' + weights_df.d_sp_id.astype(str)

            if CITY:
                weights_df = weights_df[weights_df.city == CITY]

            # TODO: remove
            weights_df = weights_df.sort_values(['city', 'o_sp_id']).reset_index(drop=True)
            print("data_shape", data_df.values.shape)
            print("weights n", len(set(weights_df['o_sp_id'].values).union(set(weights_df['d_sp_id'].values))))
            # Distance threshold
            weights_df = weights_df[(weights_df['w'] > 0)]

            # Transform weights in [node_index1, node_index2]
            # TODO bug when CITY is not specified
            # TODO sp_ids non esistenti
            weights_df = weights_df[weights_df['o_sp_id'].isin(data_df['sp_id'].values)]
            weights_df = weights_df[weights_df['d_sp_id'].isin(data_df['sp_id'].values)]

            W = scipy.spatial.distance.squareform(weights_df.sort_values(['o_sp_id', 'd_sp_id'])['w'].values)

            Tcsr = minimum_spanning_tree(csr_matrix(W))

            t = Tcsr.max()
            print("threshold", t)
            W[W > t] = 0.
            print(W[W > t])
            idxs = np.argwhere((W > 0))
            rows, cols = idxs[:, 0], idxs[:, 1]
            #W[rows, cols] = np.exp(-W[rows, cols]/t)
            W[rows, cols] = 1-(W[rows, cols] / (4*t))**2
            #W[rows, cols] = 1.-(W[rows, cols]/(4*t))**2
            print("Saving cache W")
            np.savez_compressed(cache_name, W=W)

    elif args.ODm:
        filesize = os.path.getsize('../data/generated_files/{city}_ODs.csv'.format(city=args.city))
        hash_md5 = hashlib.md5(str(filesize).encode('utf-8')).hexdigest()
        cache_name = '../cache/{city}_{spatial_name}_m_{hex}.npz'.format(city=CITY, hex=hash_md5, spatial_name=args.spatial_name)
        if os.path.exists(cache_name):
            W = np.load(cache_name)['W']
        else:
            ODs_matrix_df = pd.read_csv('../data/generated_files/{city}_ODs.csv'.format(city=args.city),
                                        dtype={'o_sp_id': str}).set_index('o_sp_id')
            ODs_matrix_df.columns = [str(x) for x in ODs_matrix_df.columns]

            ODs_matrix_df = ODs_matrix_df[ODs_matrix_df.index.isin(data_df['sp_id'].values)]
            ODs_matrix_df = ODs_matrix_df[[x for x in data_df['sp_id'].values]]

            if args.normalization == 'pop':
                ODs_matrix_df = ODs_matrix_df.stack().reset_index()
                ODs_matrix_df.columns = ['o_sp_id', 'd_sp_id', 'flow']
                ODs_matrix_df = pd.merge(ODs_matrix_df,
                                         data_df.set_index('sp_id')[['population']].rename(
                                             columns={'population': 'o_pop'}),
                                         left_on='o_sp_id', right_index=True)
                ODs_matrix_df = pd.merge(ODs_matrix_df,
                                         data_df.set_index('sp_id')[['population']].rename(
                                             columns={'population': 'd_pop'}),
                                         left_on='d_sp_id', right_index=True)
                ODs_matrix_df.loc[:, 'flow'] = ODs_matrix_df['flow'] / (ODs_matrix_df['o_pop'] + ODs_matrix_df['d_pop'] + 1)
                ODs_matrix_df = ODs_matrix_df.pivot(index='o_sp_id', columns='d_sp_id', values='flow')
            elif args.normalization == 'eff':
                ODs_matrix_df.loc[:, ODs_matrix_df.columns] = ODs_matrix_df[ODs_matrix_df.columns] / ODs_matrix_df.sum(1)

            ODs_matrix_df = ODs_matrix_df.sort_index()
            ODs_weights = ODs_matrix_df.fillna(0).values
            np.fill_diagonal(ODs_weights, 0)
            W = np.nan_to_num(ODs_weights).astype(np.float32)
            W = (W + W.T)/2

            print("Saving cache W")
            np.savez_compressed(cache_name, W=W)
    elif args.ODmd:
        fname = '../data/generated_files/spatial_dmatrix.parquet'
        filesize = os.path.getsize(fname)
        filesize += os.path.getsize('../data/generated_files/{city}_ODs.csv'.format(city=args.city))
        hash_md5 = hashlib.md5(str(filesize).encode('utf-8')).hexdigest()
        cache_name = '../cache/{city}_{spatial_name}_ODmd_{hex}.npz'.format(city=CITY, hex=hash_md5,
                                                                       spatial_name=args.spatial_name)
        if 1==0 and os.path.exists(cache_name):
            W = np.load(cache_name)['W']
        else:
            weights_df = pd.read_parquet(fname)
            weights_df = weights_df[(weights_df['spatial_name'] == args.spatial_name)]
            weights_df = weights_df.sort_values(['city', 'o_sp_id'])
            weights_df['o'] = weights_df.city + '_' + weights_df.o_sp_id.astype(str)
            weights_df['d'] = weights_df.city + '_' + weights_df.d_sp_id.astype(str)

            if CITY:
                weights_df = weights_df[weights_df.city == CITY]

            # TODO: remove
            weights_df = weights_df.sort_values(['city', 'o_sp_id']).reset_index(drop=True)
            print("data_shape", data_df.values.shape)
            print("weights n", len(set(weights_df['o_sp_id'].values).union(set(weights_df['d_sp_id'].values))))

            # Distance threshold
            weights_df = weights_df[(weights_df['w'] > 0)]

            # Transform weights in [node_index1, node_index2]
            # TODO bug when CITY is not specified
            # TODO sp_ids non esistenti
            weights_df = weights_df[weights_df['o_sp_id'].isin(data_df['sp_id'].values)]
            weights_df = weights_df[weights_df['d_sp_id'].isin(data_df['sp_id'].values)]

            weights_df.loc[:, 'w'] = 1/weights_df['w']
            W = scipy.spatial.distance.squareform(weights_df.sort_values(['o_sp_id', 'd_sp_id'])['w'].values)

            ODs_matrix_df = pd.read_csv('../data/generated_files/{city}_ODs.csv'.format(city=args.city),
                                        dtype={'o_sp_id': str}).set_index('o_sp_id')
            ODs_matrix_df.columns = [str(x) for x in ODs_matrix_df.columns]

            ODs_matrix_df = ODs_matrix_df[ODs_matrix_df.index.isin(data_df['sp_id'].values)]
            ODs_matrix_df = ODs_matrix_df[[x for x in data_df['sp_id'].values]]
            ODs_matrix_df = ODs_matrix_df.sort_index()

            ODs_weights = ODs_matrix_df.fillna(0).values
            ODs_weights = (ODs_weights + ODs_weights.T)/2
            np.fill_diagonal(ODs_weights, 0)

            W_OD = np.nan_to_num(ODs_weights).astype(np.float32)

            # normalize
            W_OD = W_OD / W_OD.sum()
            W = W / W.sum()

            # Merge the matrices
            W = 0.5*W + 0.5*W_OD
            assert 0.9 < W.sum() < 1.1
            print("Saving cache W MD")
            np.savez_compressed(cache_name, W=W)
    elif args.model == 'BSF_ODall':
        fname = '../data/generated_files/spatial_dmatrix.parquet'
        weights_df = pd.read_parquet(fname)
        weights_df = weights_df[(weights_df['spatial_name'] == args.spatial_name)]
        weights_df = weights_df.sort_values(['city', 'o_sp_id'])
        weights_df['o'] = weights_df.city + '_' + weights_df.o_sp_id.astype(str)
        weights_df['d'] = weights_df.city + '_' + weights_df.d_sp_id.astype(str)

        if CITY:
            weights_df = weights_df[weights_df.city == CITY]

        # TODO: remove
        weights_df = weights_df.sort_values(['city', 'o_sp_id']).reset_index(drop=True)
        print("data_shape", data_df.values.shape)
        print("weights n", len(set(weights_df['o_sp_id'].values).union(set(weights_df['d_sp_id'].values))))
        # Distance threshold
        weights_df = weights_df[(weights_df['w'] > 0)]

        # Transform weights in [node_index1, node_index2]
        # TODO bug when CITY is not specified
        # TODO sp_ids non esistenti
        weights_df = weights_df[weights_df['o_sp_id'].isin(data_df['sp_id'].values)]
        weights_df = weights_df[weights_df['d_sp_id'].isin(data_df['sp_id'].values)]

        W = scipy.spatial.distance.squareform(weights_df.sort_values(['o_sp_id', 'd_sp_id'])['w'].values)

        Tcsr = minimum_spanning_tree(csr_matrix(W))

        t = Tcsr.max()
        print("threshold", t)
        W[W > t] = 0.
        print(W[W > t])
        idxs = np.argwhere((W > 0))
        rows, cols = idxs[:, 0], idxs[:, 1]
        # W[rows, cols] = np.exp(-W[rows, cols]/t)
        W[rows, cols] = 1 - (W[rows, cols] / (4 * t)) ** 2
        # W[rows, cols] = 1.-(W[rows, cols]/(4*t))**2

        ODs_matrix_df = pd.read_csv('../data/generated_files/{city}_ODs.csv'.format(city=args.city),
                                    dtype={'o_sp_id': str}).set_index('o_sp_id')
        ODs_matrix_df.columns = [str(x) for x in ODs_matrix_df.columns]

        ODs_matrix_df = ODs_matrix_df[ODs_matrix_df.index.isin(data_df['sp_id'].values)]
        ODs_matrix_df = ODs_matrix_df[[x for x in data_df['sp_id'].values]]
        ODs_matrix_df = ODs_matrix_df.sort_index()

        ODs_weights = ODs_matrix_df.fillna(0).values
        ODs_weights = (ODs_weights + ODs_weights.T)/2
        np.fill_diagonal(ODs_weights, 0)

        W2 = np.nan_to_num(ODs_weights).astype(np.float32)
        W2 = W2 / W2.sum()

    else:
        fname = '../data/generated_files/egohoods_intersects.parquet'
        cache_name = '../cache/{city}_{spatial_name}_W.npz'.format(city=CITY, spatial_name='ego')
        if os.path.exists(cache_name):
            W = np.load(cache_name)['W']
        else:
            weights_df = pd.read_parquet(fname)
            weights_df = weights_df[(weights_df['spatial_name'] == args.spatial_name)]
            weights_df = weights_df.sort_values(['city', 'o_sp_id'])

            if CITY:
                weights_df = weights_df[weights_df.city == CITY]

            #weights_df = weights_df[weights_df['o_sp_id'] < weights_df['d_sp_id']]
            # TODO: remove
            weights_df = weights_df.sort_values(['city', 'o_sp_id']).reset_index(drop=True)
            print("data_shape", data_df.values.shape)
            print("weights n", len(set(weights_df['o_sp_id'].values)))

            union_sp_ids = set(weights_df['o_sp_id'].values).union(set(weights_df['d_sp_id'].values))
            islands = set(data_df['sp_id']).difference(union_sp_ids)
            if len(islands) > 0:
                print("ISLANDS", islands)
                raise AssertionError

            # Transform weights in [node_index1, node_index2]
            # TODO bug when CITY is not specified
            # TODO sp_ids non esistenti
            weights_df = weights_df[weights_df['o_sp_id'].isin(data_df['sp_id'].values)]
            weights_df = weights_df[weights_df['d_sp_id'].isin(data_df['sp_id'].values)]
            edges1 = [data_df.index[data_df['sp_id'] == x][0] for x in weights_df['o_sp_id'].values]
            edges2 = [data_df.index[data_df['sp_id'] == x][0] for x in weights_df['d_sp_id'].values]

            n = len(data_df)
            W = np.zeros((n, n), dtype='float32')
            for a, b in zip(edges1, edges2):
                W[a, b] = 1

            W = (W + W.T)/2
            W[W > 0] = 1

            print("Saving cache W")
            np.savez_compressed(cache_name, W=W)

    assert np.sum(np.diagonal(W) == 0)

    print("Considered cities", set(list(data_df.city)))

    jacobs_features = ['land_use_mix3',
                       'small_blocks',

                       'building_diversity2',

                       'density_dwellings',
                       'core_walkscore',
                       ]
    demo_features = [
        'disadvantage',
        'ethnic_diversity',
        'residential_stability'
    ]

    features = []
    if not args.core:
        if args.full or args.uf:
            features.extend(jacobs_features)
        if args.full or args.sd:
            features.extend(demo_features)


    # Core Features
    features.append('core_population')
    features.extend(['core_nightlife', 'core_shops', 'core_food'])

    if (args.full or args.m) and CITY != 'chicago':
        if args.dependent != 'ambient':
            features.append('core_ambient')
            features.append('attractiveness')
            #features.append('ambient')
            #features.append('attractiveness')
            '''
            if not args.core:
                features.append('n_trips')
                features.append('attractiveness')
            '''
        if not args.core:
            pass

    if args.minimal:
        features = [
            'small_blocks',
            'core_walkscore',
            'disadvantage',
            'ethnic_diversity',
            'core_ambient',
            'core_population',
            'core_nightlife', 'core_shops', 'core_food'
        ]
        if args.city == 'chicago':
            features.remove('core_ambient')

    if args.test:
        features.extend(args.test)

        if 'residual' in args.test:
            ODs_matrix_df = pd.read_csv('../data/generated_files/{city}_norm_ODs.csv'.format(city=args.city),
                                        dtype={'o_sp_id': str}).set_index('o_sp_id')
            ODs_matrix_df.columns = [str(x) for x in ODs_matrix_df.columns]

            ODs_matrix_df = ODs_matrix_df[ODs_matrix_df.index.isin(data_df['sp_id'].values)]
            ODs_matrix_df = ODs_matrix_df[[x for x in data_df['sp_id'].values]]
            ODs_matrix_df = ODs_matrix_df.sort_index()

            ODs_matrix_df = ODs_matrix_df.dot(1/np.exp(data_df['population'].values))
            ODs_matrix_df = ODs_matrix_df/np.exp(data_df['population'].values)
            data_df['residual'] = ODs_matrix_df.values

    print("FEATURES_SET", features)

    pp_risk = data_df['population'].values

    y = data_df[args.dependent].values
    X = (data_df[features] - data_df[features].mean()) / data_df[features].std()
    sp_ids = data_df['sp_id'].values
    print(W.shape, X.shape)
    return X, y, W, features, pp_risk, sp_ids, W2


def StanModel_cache(file=None, model_code=None, model_name=None, **kwargs):
    """Use just as you would `stan`"""

    def md5(fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    if file:
        cache_fn = '../cache/cached-{}.gz'.format(md5(file))
    else:
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        cache_fn = '../cache/cached-model-{}.gz'.format(code_hash)

    if os.path.isfile(cache_fn):
        sm = pickle.load(open(cache_fn, "rb"))
    else:
        extra_compile_args = ['-O3', '-march=native', '-pipe']

        if model_code:
            code_hash = md5(model_code.encode('ascii')).hexdigest()
            file = '../cache/{}.stan'.format(code_hash)
            with open(file, 'w') as f:
                f.write(model_code)

        sm = pystan.StanModel(file=file, extra_compile_args=extra_compile_args)
        pickle.dump(sm, open(cache_fn, "wb"))
    return sm


def main():
    sns.set_style("whitegrid")
    parser = make_argument_parser()
    args = parser.parse_args()

    # Checks
    if args.full and (args.sd or args.uf or args.m or args.core):
        parser.error("Either full or non-full parameters have to be specified.")
    if args.city == 'chicago' and (args.m or args.full or args.ODm):
        parser.error("Chicago has no mobility data.")
    if args.model == 'nb' and (args.ODd or args.ODm or args.ODmd):
        parser.error("Negative binomial does not use an OD matrix.")

    print("PARAMETERS", args)

    X, y, W, features, pp_risk, sp_ids, W2 = load_data(args, args.city)

    # Float32
    X = X.astype('float32')
    y = y.astype('int32')
    pp_risk = pp_risk.astype('float32')
    N = len(X)

    null_columns = X.columns[X.isnull().any()]
    print(X[null_columns].isnull().sum())

    assert (not (X.isnull().values.any()))
    assert (not (np.isnan(y).any()))
    assert (not (np.isnan(pp_risk).any()))

    if scipy.sparse.issparse(W):
        W = W.todense()

    X = X.values
    #ones = np.ones((N, 1), dtype='float32')
    #P_t = np.identity(N, dtype='float32') - ones.dot(ones.T)/N
    P_t = np.identity(N, dtype='float32') - X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    M = P_t.dot(W).dot(P_t)

    v, E = np.linalg.eigh(M)
    # Best eigenvectors
    #sortid = np.abs(v).argsort()[::-1]
    # Best positive eigenvectors
    sortid = v.argsort()[::-1]

    v = v[sortid]
    E = E[:, sortid]
    threshold = 0.25#.25
    #if args.ODm:
    #    threshold = 0.05
    print("eigenvectors", v[0], v[0]*0.5, len(v[v > v[0]*0.5]), v[v > 0].min())
    E = E[:, v / v.max() >= (threshold + 1e-07)]
    v = v[v / v.max() >= (threshold + 1e-07)]
    del M

    print("n. eigens", E.shape[1])

    # Dimensions
    print("N: ", X.shape[0])
    print("p: ", X.shape[1])
    print("Q: ", E.shape)

    model_variables = {
        'N': X.shape[0],
        'K': X.shape[1],
        'x': X,
        'y': y.ravel(),
        'y_real': y.ravel()
    }
    model_params = [
        'y_pred_full', 'y_pred', 'log_lik', 'betas', 'beta0', 'r2', 'r2_c', 'varI', 'varR', 'varF', 'randomE', 'intercept'
    ]
    model_params_traces_toplot = [
        'betas', 'beta0',
    ]
    if args.model == 'ESF':
        # TODO: add a list of variables
        new_variables = {
            'Q': E,
            'p': E.shape[1]
        }
        model_variables = {**model_variables, **new_variables}
        model_params.extend([
            'phi2', 'tau', 'tau2', 'phi'
        ])
        model_params_traces_toplot.extend([
            'phi2', 'tau', 'tau2'
        ])
    elif args.model == 'REESF':
        # TODO: add a list of variables
        new_variables = {
                   'Q': E,
                   'p': E.shape[1],
                   'lambdas': v
        }
        model_variables = {**model_variables, **new_variables}
        model_params.extend([
            'phi2', 'phi'
        ])
        model_params_traces_toplot.extend([
            'phi2'
        ])
    elif args.model == 'REESF-a':
        # TODO: add a list of variables
        new_variables = {
                   'Q': E,
                   'p': E.shape[1],
                   'lambdas': v
        }
        model_variables = {**model_variables, **new_variables}
        model_params.extend([
            'phi2', 'tau', 'tau2', 'phi', 'alpha'
        ])
        model_params_traces_toplot.extend([
            'phi2', 'tau', 'tau2', 'alpha'
        ])
    elif args.model == 'BSF':
        D = np.diag(np.squeeze(np.asarray(W.sum(1))))
        # L is the Laplacian
        L = D - W
        # TODO: add a list of variables
        new_variables = {
            'Q': E,
            'W': L,
            'p': E.shape[1]
        }
        model_variables = {**model_variables, **new_variables}
        model_params.extend([
            'phi2', 'tau',  'phi'
        ])
        model_params_traces_toplot.extend([
            'phi2', 'tau',
        ])

        # Deallocate
        del L
    elif args.model == 'BSF_ODall':
        M2 = P_t.dot(W2).dot(P_t)

        v2, E2 = np.linalg.eigh(M2)
        # Best eigenvectors
        # sortid = np.abs(v).argsort()[::-1]
        # Best positive eigenvectors
        sortid = v2.argsort()[::-1]

        v2 = v2[sortid]
        E2 = E2[:, sortid]
        threshold = 0.25  # .25
        # if args.ODm:
        #    threshold = 0.05
        E2 = E2[:, v2 / v2.max() >= (threshold + 1e-07)]
        v2 = v2[v2 / v2.max() >= (threshold + 1e-07)]


        D1 = np.diag(np.squeeze(np.asarray(W.sum(1))))
        D2 = np.diag(np.squeeze(np.asarray(W2.sum(1))))
        # L is the Laplacian
        L1 = D1 - W
        L2 = D2 - W2
        # TODO: add a list of variables
        new_variables = {
            'Q1': E,
            'W1': L1,
            'p1': E.shape[1],
            'Q2': E2,
            'W2': L2,
            'p2': E2.shape[1],
        }
        model_variables = {**model_variables, **new_variables}
        model_params.extend([
            'phi2', 'tau',  'phi'
        ])
        model_params_traces_toplot.extend([
            'phi2', 'tau',
        ])

        # Deallocate
        del L1
        del L2

    elif args.model == 'nb':
        model_params.extend([
            'phi2', 'tau'
        ])
        model_params_traces_toplot.extend([
            'tau', 'phi2'
        ])

    sm = StanModel_cache(file='stan_models/{}.stan'.format(args.model))

    start = time.time()
    control = None
    if args.ODd or args.ODm or args.ODmd:
        control = {'max_treedepth': 12, 'adapt_delta': 0.9}
    fit = sm.sampling(data=model_variables, iter=args.iterations, warmup=args.tuningsteps, chains=2,
                      n_jobs=args.njobs, pars=model_params, control=control)
    print(pystan.check_hmc_diagnostics(fit))

    model_name = make_model_name(args)
    data = az.from_pystan(posterior=fit,
                          observed_data=['y'],
                          log_likelihood='log_lik',
                          posterior_predictive='y_pred_full',
                          coords={'features': features},
                          dims={'betas': ['features']})

    end = time.time()
    print("Elapsed time", end - start)
    la = fit.extract(permuted=False, inc_warmup=False, pars=model_params)  # return a dictionary of arrays
    for p in model_params:
        if len(la[p].shape) > 2:
            la[p] = la[p][:, 0, :]
        else:
            la[p] = la[p][:, 0]
        la[p] = np.squeeze(la[p])

    # RESULTS
    print("PLOTTING")
    az.plot_trace(data, var_names=model_params_traces_toplot)
    savefig(plt, '../figures/traces/{spatial_model}_{model_name}.png'.format(spatial_model=args.model, model_name=model_name))
    _ = az.plot_forest(data, var_names=['betas'], credible_interval=0.9)
    plt.axvline(x=0)
    savefig(plt, '../figures/forests/{spatial_model}_{model_name}.png'.format(spatial_model=args.model, model_name=model_name))

    print("SAVING MODEL")
    with open('../data/generated_files/pkl/{spatial_model}_{model_name}.pkl'.format(spatial_model=args.model, model_name=model_name), "wb") as f:
        pickle.dump({'model': sm, 'fit': fit}, f, protocol=-1)

    print("RESULTS:")
    with open('../data/generated_files/model_results/pystan_{spatial_model}_{model_name}.csv'.format(spatial_model=args.model, model_name=model_name), "w") as ofile:
        writer = csv.writer(ofile)
        writer.writerow(['Metric', 'R1', 'R2', 'R3', 'warnings'])
        writer.writerow(['N. Eigens', str(E.shape[1]), '', '', ''])

        loo_result = az.loo(data, pointwise=True, scale='log').values
        if loo_result[5]:
            print("warnings Pareto on", round(len(loo_result[7][loo_result[7] > 0.7])/len(loo_result[7]), 2))
            print("sp_ids warnings", sp_ids[loo_result[7] > 0.7])
        writer.writerow(['LOO', loo_result[0], loo_result[1], loo_result[2], 'YES' if loo_result[3] else 'NO'])
        print("LOO", loo_result[0], 'se:', loo_result[1], 'p-value', loo_result[2])

        ll = -2 *np.sum(la['log_lik'], 1)
        dic = np.mean(ll) + 0.5 * np.var(ll)
        writer.writerow(['DIC', dic, 0.5 * np.var(ll), '', ''])
        print("DIC", dic, 0.5 * np.var(ll))

        for score_name, suffix in [('y_pred_full', ' (no RE)')]:
            result_mdape = median_absolute_p_error(y + np.finfo(float).eps,
                                                   np.mean(la[score_name] + np.finfo(float).eps, 0))
            print('MdAPE{}'.format(score_name), result_mdape)
            writer.writerow(['MdAPE{}'.format(score_name), result_mdape, '', '', ''])
            writer.writerow(['MAE{}'.format(score_name), mean_absolute_error(y, np.mean(la[score_name], 0)), '', '', ''])
            writer.writerow(['LRMSE{}'.format(score_name), np.sqrt(mean_squared_log_error(y, np.mean(la[score_name], 0))), '', '', ''])

        r2 = np.median(la['r2'])
        r2_c = np.median(la['r2_c'])
        print("R2 marginal", r2)
        print("R2 conditional", r2_c)
        writer.writerow(['R2 marginal', r2, '', '', ''])
        writer.writerow(['R2 conditional', r2_c, '', '', ''])

        e = np.mean(la['y_pred_full'], 0) - y
        MC_i = len(e)/W.sum()*(e.T.dot(W).dot(e)) / e.T.dot(e)
        writer.writerow(['MC', MC_i, '', '', ''])

        yhat = np.mean(la['y_pred_full'], 0)
        e = (y-yhat)/np.sqrt(yhat)
        MC_p = len(e)/W.sum()*(e.T.dot(W).dot(e)) / e.T.dot(e)
        writer.writerow(['MC_p', MC_p, '', '', ''])
        print("MC_p", MC_p)

    with open('../data/generated_files/model_predictions/pystan_{spatial_model}_{model_name}.csv'.format(spatial_model=args.model, model_name=model_name), "w") as ofile:
        writer = csv.writer(ofile)
        writer.writerow(['sp_id', 'y', 'y_pred_full', 'y_pred', 'randomE'])
        ys_pred_full = np.median(la['y_pred_full'], 0)
        ys_pred = np.median(la['y_pred'], 0)

        for sp_id, a, y_pred_full, y_pred, randome in zip(sp_ids, y, ys_pred_full, ys_pred, ys_pred_full-ys_pred):
            writer.writerow([sp_id, a, y_pred_full, y_pred, randome])


if __name__ == '__main__':
    main()

