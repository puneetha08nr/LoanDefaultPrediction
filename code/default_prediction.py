import pickle

import numpy as np
import pandas as pd

DATASET_PATH = '../data/dataset(2).csv'
TARGET = "default"
REQUIRED_COLUMNS = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
                    'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24',
                    'v25', 'v26', 'v27', 'v28', 'v29'
    , 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v46',
                    'v47', 'v48'
    , 'v49', 'v50', 'v51', 'v52', 'v53', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v63',
                    'v64', 'v65', 'v66', 'v67', 'v68', 'v69', 'v70', 'v71', 'v72', 'v73', 'v74', 'v75', 'v76', 'v77',
                    'v78', 'v79', 'v80'
    , 'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88', 'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96',
                    'v97'
    , 'v98', 'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v105', 'v106', 'v107', 'v108', 'v109', 'v110', 'v111',
                    'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v123',
                    'v124', 'v125',
                    'v126', 'v127', 'v128', 'v129', 'v130', 'v131', 'v132', 'v133', 'v134', 'v135', 'v136', 'v137',
                    'v138', 'v139', 'v140'
    , 'v141', 'v142', 'v143', 'v144', 'v145', 'v146', 'v147', 'v148', 'v149', 'v150', 'v151', 'v152', 'v153', 'v154',
                    'v155',
                    'v156', 'v157', 'v158', 'v159', 'v160', 'v161', 'v162', 'v163', 'v164', 'v165', 'v166', 'v167',
                    'v168', 'v169', 'v170',
                    'v171', 'v172', 'v173', 'v174', 'v175', 'v176', 'v177', 'v178', 'v179', 'v180', 'v181', 'v182',
                    'v183', 'v184', 'v185', 'v186'
    , 'v187', 'v188', 'v189', 'v190', 'v191', 'v192', 'v193', 'v194', 'v195', 'v196', 'v197', 'v198', 'v199', 'v200',
                    'v201', 'v202', 'v203', 'v204', 'v205', 'v206', 'v207', 'v208', 'v209', 'v210', 'v211', 'v212',
                    'v213', 'v214', 'v215', 'v216',
                    'v217', 'v218', 'v219', 'v220', 'v221', 'v222', 'v223', 'v224', 'v225', 'v226', 'v227', 'v228',
                    'v229', 'v230', 'v231', 'v232',
                    'v233', 'v234', 'v235', 'v236', 'v237', 'v238', 'v239', 'v240', 'v241', 'v242', 'v243', 'v244',
                    'v245', 'v246', 'v247', 'v248',
                    'v249', 'v250', 'v251', 'v252', 'v253', 'v254', 'v255', 'v256', 'v257', 'v258', 'v259', 'v260',
                    'v261', 'v262', 'v263', 'v264',
                    'v265', 'v266', 'v267', 'v268', 'v269', 'v270', 'v271', 'v272', 'v273', 'v274', 'v275', 'v276',
                    'v277', 'v278', 'v279', 'v280',
                    'v281', 'v282', 'v283', 'v284', 'v285', 'v286', 'v287', 'v288', 'v289', 'v290', 'v291', 'v292',
                    'v293', 'v294', 'v295', 'v296',
                    'v297', 'v298', 'v299', 'v300', 'v301', 'v302', 'v303', 'v304', 'v305', 'v306', 'v307', 'v308',
                    'v309', 'v310', 'v311', 'v312',
                    'v313', 'v314', 'v315', 'v316', 'v317', 'v318', 'v319', 'v320', 'v321', 'v322', 'v323', 'v324',
                    'v325', 'v326', 'v327', 'v328', 'v329'
    , 'v330', 'v331', 'v332', 'v333', 'v335', 'v336', 'v337', 'v338',
                    'v339', 'v340', 'v341', 'v342', 'v343']


def load_data(path):
    """Load data """
    print('Loading the data')
    df = pd.read_csv(path, delimiter=',')
    df1 = df[df["label"] == "oot"]
    print('Successfully uploaded the data')

    return df1


def prepare_data(df):
    """function to prepare data in order to pass as an input to our model"""
    print('Preparating the data')
    # 1. Check for default column in test data

    if TARGET in df:
        df = df[df["default"].isna()].drop(["default"], axis=1).reset_index(drop=True)

    # 2. Check for all required columns
    for col_name in REQUIRED_COLUMNS:

        if col_name not in df.columns:
            raise Exception('Required column  is missing:{}', format(col_name))

    print('Writing the data to csv file where required column values are missing')

    df[df[REQUIRED_COLUMNS].isnull().any(axis=1)].to_csv(
        '../output/required_columns_values_missing.csv')

    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    dataset = load_data(DATASET_PATH)

    dataset.drop(['v38'], axis=1, inplace=True)
    dataset.drop(['v334'], axis=1, inplace=True)
    dataset.drop(dataset.columns[344:358], axis=1, inplace=True)
    dataset.drop(dataset[dataset.isnull().sum(axis=1) > 100].index, axis=0, inplace=True)
    train_test_data = dataset[dataset['label'].notna()].reset_index(drop=True)
    dataset.drop(['label'], axis=1, inplace=True)

    for col in train_test_data:
        col_vals = train_test_data[col]
        if sum(col_vals.isnull()) != 0:
            train_test_data[col] = col_vals.fillna(col_vals.median())

    print('Data preparation finished successfully')
    return train_test_data


def predict_default(df_final):
    """This function is used to predict the default and write the result to prediction.csv file"""

    # load model from disk
    default_predictor_rf = pickle.load(open('../output/default_predictor_rf.pkl', 'rb'))

    df_final.drop(['default'], axis=1, inplace=True)
    df_final.drop(['label'], axis=1, inplace=True)
    if 'decision_id' in df_final.columns:

        df_final['defualt_prediction'] = default_predictor_rf.predict(
            df_final.drop(["decision_id"], axis=1))

        prediction = df_final[['decision_id', 'defualt_prediction']]


    else:
        df_final['defualt_prediction'] = default_predictor_rf.predict(df_final)
        prediction = df_final['defualt_prediction']

    prediction.to_csv('../output/prediction.csv')

    print('Successfully predicted the data, please check: ../output/prediction.csv')