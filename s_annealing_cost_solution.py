import numpy as np
from scipy import optimize
import os
import pandas as pd
import ipdb
import warnings
from scipy.optimize import dual_annealing
from numpy.random import rand
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
warnings.filterwarnings("ignore")


def compute_distance_matrix(X, center_roads):
    lat2 = np.radians(X[:,0]).reshape((-1,1))
    lat1 = np.radians(center_roads[:,0]).reshape((1,-1))
    lon2 = np.radians(X[:,1]).reshape((-1,1))
    lon1 = np.radians(center_roads[:,1]).reshape((1,-1))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    r = 6371
    distance_matrix_km = c*r
    return distance_matrix_km


def compute_output_costs(distance_matrix_km, flows, maximal_distance):
    output_costs = (distance_matrix_km.min(axis=1)>maximal_distance)*flows
    return output_costs
    


def crash_test_function(X, center_roads):
    # le cout serait donc la difference --> black-box solution avec de l'annealing rapide
    # On aurait comme cout --> 0 si tous les centres de route sont inférieurs à 80km par rapport à notre objectif
    # Et sinon le coût serait la distance * le nombre de camions ou qqchose dans le style.
    # Si les calucls sont bien faits pour la distance, on devrait être bon
    pass

def compute_proper_ratio_pl(element):
    if not element:
        return None
    element = float(str(element).replace(',',"."))
    if element > 40:
        element /=10
    return element

class AnnealingSolver:

    def __init__(self, number_clusters:int) -> None:
        self.number_clusters = number_clusters
    
    def compute_distance_matrix(self, X, center_roads):
        lat2 = np.radians(X[:,0]).reshape((-1,1))
        try:
            lat1 = np.radians(center_roads[:,0]).reshape((1,-1))
        except:
            ipdb.set_trace()
        lon2 = np.radians(X[:,1]).reshape((-1,1))
        lon1 = np.radians(center_roads[:,1]).reshape((1,-1))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2*np.arcsin(np.sqrt(a))
        r = 6371
        distance_matrix_km = c*r
        return distance_matrix_km
    
    def compute_output_costs(self, distance_matrix_km, flows, maximal_distance):
        output_costs = (distance_matrix_km.max(axis=0)>maximal_distance)*flows
        return output_costs.mean()
    
    def create_final_cost_function(self, center_roads, flows, maximal_distance):
        def final_cost_function(X):
            new_length = int(X.shape[0]//2)
            X = X.reshape((new_length, 2))
            function_distance = self.compute_distance_matrix
            function_cost = self.compute_output_costs
            distance_matrix_km = self.compute_distance_matrix(X, center_roads)
            output_costs = self.compute_output_costs(distance_matrix_km, flows, maximal_distance)
            return output_costs          
        return final_cost_function

    def create_linear_bounds_unconstrained_version(self, number_variables):
        lower_bounds_vector = np.zeros((number_variables,))
        upper_bounds_vector = 73*np.ones((number_variables, ))
        lower_bounds = optimize.Bounds(lb=lower_bounds_vector,ub=upper_bounds_vector)
        return lower_bounds
    
    def fit_annealing(self, data_roads:pd.DataFrame, maximal_distance:float, max_iters):
        self.center_roads = data_roads[['center_x','center_y']].values
        self.flows = data_roads['daily_flow_trucks'].values
        self.maximal_distance = maximal_distance

        number_variables = self.number_clusters*2
        cost_function = self.create_final_cost_function(self.center_roads, self.flows, self.maximal_distance)
        lower_bounds = self.create_linear_bounds_unconstrained_version(number_variables)
        X = np.random.rand(1000,2)
        Y = np.random.randn(number_variables//2,2)
        Y[:,0]*=5
        Y[:,0]+=1
        Y[:,1]*=64
        Y[:,1]+=9
        res = optimize.dual_annealing(
            cost_function, 
            bounds=lower_bounds,
            maxiter=max_iters,
            x0 = Y.flatten()
            )
        self.final_position = res.x.reshape((self.number_clusters, 2))
    
    def fit_genetic(self, data_roads:pd.DataFrame, maximal_distance:float, max_iters):
        self.center_roads = data_roads[['center_x','center_y']].values
        self.flows = data_roads['daily_flow_trucks'].values
        self.maximal_distance = maximal_distance

        number_variables = self.number_clusters*2
        cost_function = self.create_final_cost_function(self.center_roads, self.flows, self.maximal_distance)
        lower_bounds = self.create_linear_bounds_unconstrained_version(number_variables)
        X = np.random.rand(1000,2)
        Y = np.random.rand(number_variables//2,2)
        Y[:,0]*=5
        Y[:,0]+=1
        Y[:,1]*=64
        Y[:,1]+=9
        res = optimize.differential_evolution(
            cost_function, 
            bounds=lower_bounds,
            maxiter=max_iters,
            x0 = Y.flatten()
            )
        self.final_position = res.x.reshape((self.number_clusters, 2))

if __name__ == "__main__":
    raw_data = pd.read_csv(os.path.join("data","tmja-2019.csv"), sep=";")
    for colonne in raw_data.columns:
        try:
            raw_data[colonne] = raw_data[colonne].apply(lambda element:element.replace(',',"."))
        except:
            # print(raw_data[colonne].dtype)
            pass
    for colonne in raw_data.columns:
        try:
            raw_data[colonne] = raw_data[colonne].astype(float)
        except:
            pass

    df = raw_data[["route",'xD',"yD","xF", "yF","TMJA","ratio_PL"]]
    df.loc[:, ['xD',"yD","xF", "yF"]]/=1e5
    values =["x","y"]
    for value in values:
        df.loc[:,f'center_{value}'] = (df.loc[:, f"{value}D"]+df.loc[:,f"{value}F"])/2
    df['proper_ratio_PL'] = df["ratio_PL"].apply(compute_proper_ratio_pl)
    df.dropna(inplace=True)
    df["daily_flow_trucks"] = (df["proper_ratio_PL"]/100) *df['TMJA']
    problem_solver =  AnnealingSolver(number_clusters=120)
    def objective(v):
        x, y = v
        return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
    problem_solver.fit(df[['center_x','center_y','daily_flow_trucks']],maximal_distance=300, max_iters=10)
    # define range for input
    # r_min, r_max = -5.0, 5.0
    # # define the bounds on the search
    # bounds = [[r_min, r_max], [r_min, r_max]]
    # # perform the dual annealing search
    # result = dual_annealing(objective, bounds)
    # # summarize the result
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    # # evaluate solution
    # solution = result['x']
    # evaluation = objective(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))
    
    ipdb.set_trace()