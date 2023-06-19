from utils import EikonalSolver2D
import numpy as np
import tensorflow as tf
import argparse
import os







exps = [
{"path":"Exp_1","prov_opt":"Adam","prov_act_tt":"tanh","prov_act_vel":"tanh","layers_tt":[4]+8*[20]+[1],"layers_vel":[2]+4*[32]+[1]},
{"path":"Exp_2","prov_opt":"Adam","prov_act_tt":"relu","prov_act_vel":"relu","layers_tt":[4]+8*[20]+[1],"layers_vel":[2]+4*[32]+[1]},
{"path":"Exp_3","prov_opt":"Adam","prov_act_tt":"sigmoid","prov_act_vel":"sigmoid","layers_tt":[4]+8*[20]+[1],"layers_vel":[2]+4*[32]+[1]},
{"path":"Exp_4","prov_opt":"Adam","prov_act_tt":"relu","prov_act_vel":"tanh","layers_tt":[4]+8*[20]+[1],"layers_vel":[2]+4*[32]+[1]},
{"path":"Exp_5","prov_opt":"RMSProp","prov_act_tt":"tanh","prov_act_vel":"tanh","layers_tt":[4]+8*[20]+[1],"layers_vel":[2]+4*[32]+[1]},
{"path":"Exp_6","prov_opt":"Adam","prov_act_tt":"tanh","prov_act_vel":"tanh","layers_tt":[4]+16*[40]+[1],"layers_vel":[2]+8*[64]+[1]},
{"path":"Exp_7","prov_opt":"RMSProp","prov_act_tt":"tanh","prov_act_vel":"tanh","layers_tt":[4]+16*[40]+[1],"layers_vel":[2]+8*[64]+[1]},
{"path":"Exp_8","prov_opt":"Adam","prov_act_tt":"tanh","prov_act_vel":"tanh","layers_tt":[4]+16*[20]+[1],"layers_vel":[2]+8*[32]+[1]},
{"path":"Exp_9","prov_opt":"RMSProp","prov_act_tt":"tanh","prov_act_vel":"tanh","layers_tt":[4]+16*[20]+[1],"layers_vel":[2]+8*[32]+[1]},
]

local = os.getcwd() + '/mnt/Potato/Experimentos'

for element in exps:
    path = element["path"]
    local_save = local + '/' + path
    
    
    """ Parser """

    parser = argparse.ArgumentParser(description="Eikonal Problem")

    # Adam its
    #parser.add_argument('--adam_its', type=int, default=400000)
    parser.add_argument('--adam_its', type=int, default=400000)

    # Number of grid points (in each dimension)
    parser.add_argument('--nx', type=int, default=10*20)

    parser.add_argument('--ny', type=int, default=10*25)

    if not os.path.exists(local_save):
        os.mkdir(local_save)
    
    parser.add_argument('--folder', type=str, default=local_save)

    # Batch size
    parser.add_argument('--batchsize', type=int, default=3000)

    # Arguments
    args = parser.parse_args()
    
    """ Preprocess the data """
    data_np = np.load('data/crosshole_data.npz')

    pp_data = {'xs': data_np['datax'][:, 0:1],
               'ys': data_np['datax'][:, 1:2],
               'xr': data_np['datax'][:, 2:3],
               'yr': data_np['datax'][:, 3:4],
               'tt': data_np['datay'][:, 0:1],
               'sp_x': data_np['sources'][:, 0:1],
               'sp_y': data_np['sources'][:, 1:2]}

    """ Set the solver """
    

    # layers_vel = [2] + 4 * [32] + [1]
    # layers_tt = [4] + 8 * [20] + [1]



    
    eikonal_solver = EikonalSolver2D(layers_vel=element["layers_vel"],
                                     layers_tt=element["layers_tt"],
                                     batch_size=args.batchsize,
                                     vmin=1.5, vmax=4.0,
                                     xmin=0, xmax=20,
                                     ymin=-27.5, ymax=0,
                                     data=pp_data,
                                     base_folder=args.folder,
                                     adam_its=args.adam_its,
                                     prov_opt = element["prov_opt"],
                                     prov_act_tt = element["prov_act_tt"],
                                     prov_act_vel = element["prov_act_vel"],
                                     path = path)

    """ Train the model """


    eikonal_solver.train(Nx=args.nx, Ny=args.ny)
    
    
