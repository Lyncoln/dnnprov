from external_optimizer import ScipyOptimizerInterface
from pyevtk.hl import gridToVTK
import sys
import time
import numpy as np
import tensorflow.experimental.numpy as tnp
import tensorflow.compat.v1 as tf
import os
import pandas as pd
from datetime import datetime



from sklearn.metrics import r2_score



tf.disable_v2_behavior()

tf.set_random_seed(10)

""" TensorFlow Session """


def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)

    # init
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


""" MSE Loss """


def mean_squared_error(pred, exact):
    return tf.reduce_mean(tf.square(pred - exact))


""" Gradient Computation """

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    # Y_x = tf.gradients(Y, x)[0]
    return Y_x


""" Velnet """


class neural_net_vel(object):
    def __init__(self, *inputs, layers, n_acti=3.0, activation_fn=tf.nn.tanh, vmax=5.5, vmin=1.5):

        self.layers = layers
        self.num_layers = len(self.layers)
        self.activation_fn = activation_fn
        self.n_acti = n_acti

        X = np.array([[0.0, -27.5],
                      [0.0, 0.0],
                      [20, -27.5],
                      [20, 0.0]])
        self.X_mean = X.mean(0, keepdims=True)
        self.X_std = X.std(0, keepdims=True)

        self.layers[0] = self.layers[0] + 4
        # self.layers[-1] = 2

        self.vmax = vmax
        self.vmin = vmin

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = np.zeros([1, out_dim])
            g = 1.0 / self.n_acti
            # tensorflow variables
            self.weights.append(tf.Variable(
                W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(
                b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
                           dtype=tf.float32)

    def __call__(self, *inputs):
        Hi = (tf.concat(inputs, 1) - self.X_mean) / self.X_std
        # Hi = tf.concat(inputs, 1)
        # H = Hi
        H = tf.concat([Hi,
                       tf.sin(2.0 * np.pi * Hi),
                       tf.cos(2.0 * np.pi * Hi)], 1)

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            # V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, W)
            # add bias
            H = self.n_acti * g * (H + b)
            # activation
            if l < self.num_layers - 2:
                H = self.activation_fn(H)

        H = tf.nn.sigmoid(H) * (self.vmax - self.vmin) + self.vmin
        # H = H1 * H2 + H1 + b
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)

        return Y


""" QRes """


# QRes for the traveltimes field
class QRes_TT(object):
    def __init__(self, *inputs, layers, activation_fn=tf.nn.tanh, dtype=tf.float32):
        # Layers and activation
        self.layers = layers
        self.num_layers = len(self.layers)
        self.activation_fn = activation_fn

        # Floating point precidion
        self.dtype = dtype

        X = np.array([[0.0, -30, 0.0, -30],
                      [0.0, 0.0, 0.0, 0.0],
                      [20, -30, 20, -30],
                      [20, 0.0, 20, 0.0]])
        self.X_mean = X.mean(0, keepdims=True)
        self.X_std = X.std(0, keepdims=True)

        # self.layers[0] = int(2 * self.layers[0])

        self.weights = []
        self.biases = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W1 = self.xavier_init(size=[layers[l], layers[l + 1]])
            W2 = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = np.zeros([1, out_dim])

            # tensorflow variables
            self.weights.append((W1, W2))
            self.biases.append(tf.Variable(
                b, dtype=self.dtype, trainable=True))

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.dtype),
                           dtype=self.dtype)

    def __call__(self, *inputs):
        num_layers = len(self.weights) + 1
        # H = tf.concat(inputs, 1)
        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std
        # H = tf.concat([H, tf.math.cos(2.0 * np.pi * H[:, 2:] / np.array([20, 30])),
        #                tf.math.sin(2.0 * np.pi * H[:, 2:] / np.array([20, 30]))], 1)
        # H = tf.concat([H, tf.math.cos(2.0 * np.pi * H[:, 2:]), tf.math.sin(2.0 * np.pi * H[:, 2:])], 1)

        """ My Code """
        for l in range(0, num_layers - 2):
            W1, W2 = self.weights[l]
            V1 = W1 / tf.norm(W1, axis=0, keepdims=True)
            V2 = W2 / tf.norm(W2, axis=0, keepdims=True)
            b = self.biases[l]
            H1 = tf.matmul(H, V1)
            H2 = tf.matmul(H, V2)
            H = self.activation_fn(H1 * H2 + H1 + b)

        W1, W2 = self.weights[-1]
        V1 = W1 / tf.norm(W1, axis=0, keepdims=True)
        V2 = W2 / tf.norm(W2, axis=0, keepdims=True)
        b = self.biases[-1]
        H1 = tf.matmul(H, V1)
        H2 = tf.matmul(H, V2)
        H = tf.math.abs(H1 * H2 + H1 + b)
        # H = H1 * H2 + H1 + b
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)

        return Y

    def tt(self, *inputs):
        H = tf.concat(inputs, 1)
        xs = H[:, 0:1]
        ys = H[:, 1:2]
        xr = H[:, 2:3]
        yr = H[:, 3:4]
        [tau] = self.__call__(H)
        T0 = tf.sqrt(tf.square(xr - xs) + tf.square(yr - ys))

        return [tf.multiply(T0, tau)]


# QRes for the velocity field
class QRes_Vel(object):
    def __init__(self, *inputs, layers, vmin, vmax, activation_fn=tf.nn.tanh, dtype=tf.float32):
        # Layers and activation
        self.layers = layers
        self.num_layers = len(self.layers)
        self.activation_fn = activation_fn

        # Inf and Sup limits for the velocity field
        self.vmin = vmin
        self.vmax = vmax

        # self.layers[0] = 4+self.layers[0]

        # Floating point precision
        self.dtype = dtype

        # if len(inputs) == 0:
        #     in_dim = self.layers[0]
        #     self.X_mean = np.zeros([1, in_dim])
        #     self.X_std = np.ones([1, in_dim])
        # else:
        X = np.array([[0.0, -30],
                      [0.0, 0.0],
                      [20, -30],
                      [20, 0.0]])
        self.X_mean = X.mean(0, keepdims=True)
        self.X_std = X.std(0, keepdims=True)
        # self.X_mean = np.mean(np.array([]))
        self.layers[0] = int(3 * self.layers[0])

        self.weights = []
        self.biases = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W1 = self.xavier_init(size=[layers[l], layers[l + 1]])
            W2 = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = np.zeros([1, out_dim])

            # tensorflow variables
            self.weights.append((W1, W2))
            self.biases.append(tf.Variable(
                b, dtype=self.dtype, trainable=True))

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.dtype),
                           dtype=self.dtype)

    def __call__(self, *inputs):
        num_layers = len(self.weights) + 1
        # H = tf.concat(inputs, 1)
        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std
        # H = tf.concat([H, tf.math.cos(2.0 * np.pi * H / np.array([5, 1.5])),
        #                tf.math.sin(2.0 * np.pi * H / np.array([5, 1.5]))], 1)
        # H = tf.concat([H, tf.math.cos(2.0 * np.pi * H),
        #                tf.math.sin(2.0 * np.pi * H)], 1)
        H = tf.concat([H, tf.math.cos(2.0 * np.pi * H), tf.math.sin(2.0 * np.pi * H)], 1)
        """ My Code """
        for l in range(0, num_layers - 2):
            W1, W2 = self.weights[l]
            V1 = W1 / tf.norm(W1, axis=0, keepdims=True)
            V2 = W2 / tf.norm(W2, axis=0, keepdims=True)
            b = self.biases[l]
            H1 = tf.matmul(H, V1)
            H2 = tf.matmul(H, V2)
            H = self.activation_fn(H1 * H2 + H1 + b)

        W1, W2 = self.weights[-1]
        V1 = W1 / tf.norm(W1, axis=0, keepdims=True)
        V2 = W2 / tf.norm(W2, axis=0, keepdims=True)
        b = self.biases[-1]
        H1 = tf.matmul(H, V1)
        H2 = tf.matmul(H, V2)

        # H = (self.vmax - self.vmin) * \
        #     tf.nn.sigmoid(H1 * H2 + H1 + b) + self.vmin
        H = H1 * H2 + H1 + b
        # H = (self.vmax - self.vmin) * tf.clip_by_value(H, clip_value_max=1.0, clip_value_min=0.0) + self.vmin
        H = tf.nn.sigmoid(H) * (self.vmax - self.vmin) + self.vmin
        # H = tf.abs(tf.cos(H))*self.vmax
        # H = H1 * H2 + H1 + b
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)

        return Y


""" Eikonal Residual """


def FEE2D(tau, vel, xs, ys, xr, yr):
    # \tau gradient
    tau_xr = fwd_gradients(tau, xr)
    tau_yr = fwd_gradients(tau, yr)

    # Distance from source to receiver
    T0 = tf.sqrt(tf.square(xr - xs) + tf.square(yr - ys))

    # Components of the residual
    T1 = tf.square(T0) * (tf.square(tau_xr) + tf.square(tau_yr))

    T2 = 2.0 * tau * (tau_xr * (xr - xs) + tau_yr * (yr - ys))

    T3 = tf.square(tau)

    # Vel derivatives for TV Regularization
    vel_xr = fwd_gradients(vel, xr)
    vel_yr = fwd_gradients(vel, yr)

    res = T1 + T2 + T3 - tf.square(1.0 / vel)
    TV = tf.square(vel_xr) + tf.square(vel_yr)

    return [res, TV]


""" Eikonal 2D Solver """


class EikonalSolver2D(object):
    def __init__(self, layers_vel, layers_tt,
                 batch_size, data,
                 vmin, vmax,
                 xmin, xmax,
                 ymin, ymax,
                 base_folder,
                 adam_its,
                 prov_opt,
                 prov_act_tt,
                 prov_act_vel,
                 path,
                 weight_lb,
                 weight_lr,
                 weight_ld,
                 dtype=tf.float32
                 ):

        self.weight_lb = weight_lb
        self.weight_lr = weight_lr
        self.weight_ld = weight_ld

                 
        # Prov capture
        self.prov_act_tt = prov_act_tt
        self.prov_act_vel = prov_act_vel
        self.prov_opt = prov_opt
        self.path = path
        
        # Layers and batch_size
        self.layers_vel = layers_vel
        self.layers_tt = layers_tt
        self.batch_size = batch_size

        # Floating point precision
        self.dtype = dtype

        # Inf and Sup limits for the velocity field
        self.vmin = vmin
        self.vmax = vmax

        # Inf and Sup limits for the domain
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

        # Number of iterations
        self.adam_its = adam_its

        """ Extracting data """
        # Measurements
        self.xs_data = data['xs']
        self.ys_data = data['ys']
        self.xr_data = data['xr']
        self.yr_data = data['yr']
        self.tt_data = data['tt']
        # Sources
        self.sp_x = data['sp_x']
        self.sp_y = data['sp_y']

        """ Folders and DataFrame """
        # Convergence dataframe
        conv_cols = ['Epoch', 'Ld', 'Lr', 'Lb', 'Loss', 'Rt']

        # Create DataFrame
        self.conv_dataframe = pd.DataFrame(columns=conv_cols)

        # Root folder for saving stuff
        self.root_folder = base_folder

        self.folder2save = os.path.join(self.root_folder, 'marmousi_pinn', 'vel')

        self.folder4model = os.path.join(self.root_folder, 'marmousi_pinn', 'model_save')

        os.makedirs(self.folder2save, exist_ok=True)

        os.makedirs(self.folder4model, exist_ok=True)

        """ Placeholders """
        # Placeholders for the data points
        self.xs_data_tf = tf.placeholder(dtype=self.dtype, shape=[
            None, 1], name="xs_data")
        self.ys_data_tf = tf.placeholder(dtype=self.dtype, shape=[
            None, 1], name="ys_data")
        self.xr_data_tf = tf.placeholder(dtype=self.dtype, shape=[
            None, 1], name="xr_data")
        self.yr_data_tf = tf.placeholder(dtype=self.dtype, shape=[
            None, 1], name="yr_data")
        self.tt_data_tf = tf.placeholder(dtype=self.dtype, shape=[
            None, 1], name="tt_data")

        # Placeholders for the residual points
        self.xs_eqn_tf = tf.placeholder(
            dtype=self.dtype, shape=[None, 1], name="xs_eqn")
        self.ys_eqn_tf = tf.placeholder(
            dtype=self.dtype, shape=[None, 1], name="ys_eqn")
        self.xr_eqn_tf = tf.placeholder(
            dtype=self.dtype, shape=[None, 1], name="xr_eqn")
        self.yr_eqn_tf = tf.placeholder(
            dtype=self.dtype, shape=[None, 1], name="yr_eqn")

        # Placeholder for boundary conditions
        self.xs_bc_tf = tf.placeholder(
            dtype=self.dtype, shape=[None, 1], name="xs_bc")
        self.ys_bc_tf = tf.placeholder(
            dtype=self.dtype, shape=[None, 1], name="ys_bc")

        # Placeholder for prediction
        self.xr_star_tf = tf.placeholder(dtype=self.dtype, shape=[
            None, 1], name="xr_star")
        self.yr_star_tf = tf.placeholder(dtype=self.dtype, shape=[
            None, 1], name="yr_star")
            
        self.df = pd.DataFrame(columns=[self.path])
        
        

        """ Neural Networks """
        if(self.prov_act_tt == "tanh"):
            acttt = lambda x: tf.nn.tanh(x)
        elif(self.prov_act_tt == "relu"):
            acttt = lambda x: tf.nn.relu(x)
        elif(self.prov_act_tt == "sigmoid"):
            acttt = lambda x: tf.nn.sigmoid(x)
        

        self.net_tt = QRes_TT(layers=self.layers_tt, activation_fn=acttt)

        actvel = lambda x: tf.nn.tanh(x)
        if(self.prov_act_vel == "tanh"):
            actvel = lambda x: tf.nn.tanh(x)
        elif(self.prov_act_vel == "relu"):
            actvel = lambda x: tf.nn.relu(x)
        elif(self.prov_act_vel == "sigmoid"):
            actvel = lambda x: tf.nn.sigmoid(x)

        self.net_vel = QRes_Vel(layers=self.layers_vel, vmin=self.vmin, vmax=self.vmax,
                                activation_fn=actvel)

        """ Prediction """
        [self.vel_star] = self.net_vel(self.xr_star_tf, self.yr_star_tf)

        """ Boundary Condition """

        # Evaluate tt (xs -> xs) and vel at the source positions
        [self.tau_bc_pred] = self.net_tt(
            self.xs_bc_tf, self.ys_bc_tf, self.xs_bc_tf, self.ys_bc_tf)
        [self.vel_bc_pred] = self.net_vel(self.xs_bc_tf, self.ys_bc_tf)

        # Compute the Boundary Loss
        self.Lb = mean_squared_error(
            pred=self.tau_bc_pred, exact=1.0 / self.vel_bc_pred)

        """ Residual Points """
        [self.tau_eqn_pred] = self.net_tt(
            self.xs_eqn_tf, self.ys_eqn_tf, self.xr_eqn_tf, self.yr_eqn_tf)
        [self.vel_eqn_pred] = self.net_vel(self.xr_eqn_tf, self.yr_eqn_tf)

        # Compute the residual vector
        [self.e1_pred,
         self.tv_reg_pred] = FEE2D(self.tau_eqn_pred,
                                   self.vel_eqn_pred,
                                   self.xs_eqn_tf,
                                   self.ys_eqn_tf,
                                   self.xr_eqn_tf,
                                   self.yr_eqn_tf)

        # Compute the residual loss
        self.Lr = mean_squared_error(pred=self.e1_pred, exact=0.0)  # + 1e-6 * mean_squared_error(self.tv_reg_pred, 0.0)

        """ Data Points """
        [self.tt_data_pred] = self.net_tt.tt(
            self.xs_data_tf, self.ys_data_tf, self.xr_data_tf, self.yr_data_tf)

        # Compute the data loss
        self.Ld = mean_squared_error(
            pred=self.tt_data_pred, exact=self.tt_data_tf)

        """ Final Loss """
        self.loss = self.weight_ld*self.Ld + self.weight_lr*self.Lr + self.weight_lb*self.Lb

        """ Optimizers """
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(
            starter_learning_rate, self.global_step, 1000, 0.99, staircase=False)
            
        self.starter_learning_rate = 1e-3    
	
        
        self.train_op_Adam = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.train_op_RMS = tf.train.RMSPropOptimizer(
            self.learning_rate).minimize(self.loss, global_step=self.global_step)

        """
	
        self.train_op_Adam = tf.train.AdamOptimizer(self.learning_rate)

        self.train_opt_Adam = self.train_op_Adam.minimize(self.loss, global_step=self.global_step)

        self.train_op_RMS = tf.train.RMSPropOptimizer(self.learning_rate)

        self.train_opt_RMS = self.train_op_RMS.minimize(self.loss, global_step=self.global_step)
        
        """
        
        
        #self.train_op = self.train_op_Adam ###################################
        #self.train_op = self.train_op_RMS
        
        
        self.block_size_BFGS = int(100)
        

        self.n_blocks_BFGS = int(self.adam_its / self.block_size_BFGS)

        self.opt_LBFGS = ScipyOptimizerInterface(self.loss,
                                                 method='L-BFGS-B',
                                                 options={'maxiter': self.block_size_BFGS,
                                                          'maxfun': self.block_size_BFGS,
                                                          'maxcor': self.block_size_BFGS,
                                                          'maxls': self.block_size_BFGS,
                                                          'ftol': 1.0 * np.finfo(float).eps})

        """ TensorFlow Session """
        self.sess = tf_session()

    def vel_true_func(self,x,y):
        vel = 2.0 * np.ones_like(x)
        c1_flag = np.sqrt((x-7)**2 + (y+10)**2) < 3
        vel[c1_flag] = 4.0
        c2_flag = np.sqrt((x-12)**2 + (y+18)**2) < 4
        vel[c2_flag] = 1.5
        return vel
        
        
    def train(self, Nx, Ny):
        # Create a list with the residual points
        x_res = np.linspace(self.xmin, self.xmax, Nx, dtype=np.float32)
        y_res = np.linspace(self.ymin, self.ymax, Ny, dtype=np.float32)

        x_res_grid, y_res_grid = np.meshgrid(x_res, y_res)

        x_res = x_res_grid.reshape(-1)[..., None]
        y_res = y_res_grid.reshape(-1)[..., None]

        # Get the number of points (data, boundary and residual)
        N_data = self.xr_data.shape[0]
        N_eqn = x_res.shape[0]
        N_bd = self.sp_x.shape[0]

        # Set time
        start_time = time.time()
        running_time = 0
        it = 0
        
        
 

        while it < self.adam_its:

            # Samples IDs
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            idx_eqn_res = np.random.choice(N_eqn, min(self.batch_size, N_eqn))
            idx_eqn_sources = np.random.choice(N_bd, min(self.batch_size, N_eqn))
            idx_bd = np.random.choice(N_bd, min(self.batch_size, N_bd))

            # Feed dictionary
            tf_dict = {self.xs_data_tf: self.xs_data[idx_data, :],
                       self.ys_data_tf: self.ys_data[idx_data, :],
                       self.xr_data_tf: self.xr_data[idx_data, :],
                       self.yr_data_tf: self.yr_data[idx_data, :],
                       self.tt_data_tf: self.tt_data[idx_data, :],
                       self.xs_bc_tf: self.sp_x[idx_bd, :],
                       self.ys_bc_tf: self.sp_y[idx_bd, :],
                       self.xs_eqn_tf: self.sp_x[idx_eqn_sources, :],
                       self.ys_eqn_tf: self.sp_y[idx_eqn_sources, :],
                       self.xr_eqn_tf: x_res[idx_eqn_res, :],
                       self.yr_eqn_tf: y_res[idx_eqn_res, :]}  # ,

            # Perform Adam or RMSProp update
            if it < 500000:
                if(self.prov_opt == "Adam"):
                    self.sess.run([self.train_op_Adam], tf_dict)
                elif(self.prov_opt == "RMSProp"):
                    self.sess.run([self.train_op_RMS], tf_dict)
                # self.sess.run([self.train_op.minimize(self.loss, global_step=self.global_step)], tf_dict)


            # if it % self.block_size_BFGS == 0 and it > 50000:
            #     self.opt_LBFGS.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss])

            # Print
            if it % 100 == 0:

                elapsed = time.time() - start_time
                running_time += elapsed / 3600.0
                [loss_value,
                 Lb,
                 Ld,
                 Lr,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.Lb,
                                                       self.Ld,
                                                       self.Lr,
                                                       self.learning_rate], tf_dict)

                print(self.path,
                    ' It: %d, Loss: %.3e, Lb: %.3e, Ld: %.3e, Lr: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                    % (it, loss_value, Lb, Ld, Lr, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
                
                '''
                
                tf1_output = DataSet("oTrainingModel", [Element([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), elapsed, loss_value, Lb, Lr, Ld,0, it])])
                t1.add_dataset(tf1_output)
		    
                
                
                t2 = Task(2, dataflow_tag, exec_tag, "Adaptation", dependency=t1)
                t2.begin()                 
                tf2_output = DataSet("oAdaptation", [Element([learning_rate_value, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), it, it])])
                t2.add_dataset(tf2_output)
                
                t1.end()
                t2.end()
                
                
                if(it == self.adam_its - 100):
                  t1.end()
                  t2.end()
                else:
                  t1.save()
                  t2.save()
                '''  

                vel_pred = self.predict(x_star=x_res, y_star=y_res)
                ####
                vel_true = self.vel_true_func(x_res, y_res)
                
                r2 = r2_score(vel_true,vel_pred)
                print(r2)
                
                
                self.df = self.df.append({self.path:elapsed},ignore_index=True)
		    
                
     
                
                #t1.end()
                #t2.end()
                ####
                if(it == self.adam_its - 100):
                	pass

                else:
                
                	pass
                if it % 1000 == 0:
                    gridToVTK(self.folder2save + "/pred" + str(it), x=x_res_grid[:, :, np.newaxis],
                              y=y_res_grid[:, :, np.newaxis],
                              z=np.zeros_like(x_res_grid[:, :, np.newaxis]),
                              pointData={"vel": vel_pred.reshape(x_res_grid[:, :, np.newaxis].shape)})
                    # if it % 1000 == 0:
                    # self.saver.save(self.sess, self.folder4model + '/feepinn', global_step=it)
                    self.conv_dataframe = self.conv_dataframe.append(pd.Series(
                        data={'Epoch': it, 'Ld': Ld, 'Lr': Lr, 'Lb': Lb, 'Loss': loss_value, 'Rt': elapsed}),
                        ignore_index=True)

                    self.conv_dataframe.to_csv(self.folder2save + "convergence.csv")
            it += 1
            
        #t1.save()
        #t2.save()
        self.df.to_csv(self.path+".csv")

    def predict(self, x_star, y_star):
        tf_dict = {self.xr_star_tf: x_star,
                   self.yr_star_tf: y_star}

        vel_star = self.sess.run(self.vel_star, tf_dict)

        return vel_star
