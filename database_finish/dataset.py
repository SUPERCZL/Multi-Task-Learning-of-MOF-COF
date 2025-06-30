import pandas as pd
import numpy as np

class data():
    def __init__(self,
                 database_name_,
                 feature_,
                 target_,
                 multitask_=False,
                 feature_list=False,
                 path_rdf='E://Machine learning/Database/H2/feature_finish/',
                 path_fe='E://Machine learning/Database/H2/feature_finish/free-energy/',
                 path_database='E://Machine learning/Database/H2/database_finish/',
                 path_database_nom='E://Machine learning/Database/H2/database_finish/'):
        self.path_database = path_database
        self.database_name = database_name_
        self.feature = feature_
        self.target = target_
        self.feature_list = feature_list
        self.path_rdf = path_rdf
        self.path_fe = path_fe
        self.path_database = path_database
        self.path_database_nom = path_database_nom
        self.multitask = multitask_

    def rdf_in(self, rdf_char=False):
        if self.database_name == 'MOF' or self.database_name == 'MOF-nom':
            result = pd.read_csv(self.path_rdf + 'aprdf_epsilon_mof_nom(s).csv')
            return result, result.columns
        if self.database_name == 'COF7' or self.database_name == 'COF7-nom':
            result = pd.read_csv(self.path_rdf + 'aprdf_epsilon_cof7_nom(s).csv')
            return result, result.columns
        
    def fe_in(self, rdf_char=False):
        usecols = np.arange(0, 10, 1)
        if self.database_name == 'MOF':
            result = pd.read_csv(self.path_fe + 'mof_fe(s).csv', usecols=usecols)
            return result, result.columns
        if self.database_name == 'COF7':
            result = pd.read_csv(self.path_fe + 'cof7_fe(s).csv', usecols=usecols)
            return result, result.columns
        if self.database_name == 'MOF-nom':
            result = pd.read_csv(self.path_fe + 'mof_fe_nom(s).csv', usecols=usecols)
            return result, result.columns
        if self.database_name == 'COF7-nom':
            result = pd.read_csv(self.path_fe + 'cof7_fe_nom(s).csv', usecols=usecols)
            return result, result.columns

    def data_read(self):
        if self.database_name == 'MOF':
            return pd.read_csv(self.path_database + 'mof_data(s).csv')
        elif self.database_name == 'COF':
            return pd.read_csv(self.path_database + 'cof_data(s).csv')
        elif self.database_name == 'COF7':
            return pd.read_csv(self.path_database + 'cof7_data(s).csv')
        elif self.database_name == 'MOF-nom':
            return pd.read_csv(self.path_database_nom + 'mof_data_nom(s).csv')
        elif self.database_name == 'COF-nom':
            return pd.read_csv(self.path_database_nom + 'cof_data_nom(s).csv')
        elif self.database_name == 'COF7-nom':
            return pd.read_csv(self.path_database_nom + 'cof7_data_nom(s).csv')

    def data_in(self):
        geo = ['VF', 'Density', 'ASA', 'LCD', 'PLD']
        energy_ave = ['Henry_ave', 'Min(Ef)_ave', 'V<12', 'V=[12,16]', 'V>16']
                      #'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        #energy_ave = ['Henry_ave']
        voro = ['Voro_H']

        data = self.data_read()
        result = self.feature.split("+")
        feature = []
        Chemical = False
        fe = False
        if 'Geometric' in result:
            feature.extend(geo)
        if 'Energy' in result:
            feature.extend(energy_ave)
            #fe = True
        if 'Chemical' in result:
            feature.extend(voro)
            Chemical = True  
        print(feature)
        data_x = data[feature]
        if self.multitask:
            data_y = data[[val for val in self.target]]
        else:
            data_y = data[self.target]
        if Chemical:
            rdf_data, rdf_char = self.rdf_in()
            data_x = np.hstack((data_x, np.array(rdf_data)))
            feature = np.hstack((feature, rdf_char))
        if fe:
            fe_data, fe_char = self.fe_in()
            data_x = np.hstack((data_x, np.array(fe_data)))
            feature = np.hstack((feature, fe_char))
            
        if self.feature_list:
            return np.array(data_x), np.array(data_y), feature
        return np.array(data_x), np.array(data_y)



