import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# python data types
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.python.client import timeline
from tensorflow.python.ops import clip_ops
from optimisers.adamax import AdamaxOptimizer

DTYPE = tf.float32
NP_DTYPE = np.float32

tfd = tf.contrib.distributions
tfb = tfd.bijectors
np.random.seed(1)
tf.set_random_seed(1)


class init_dist(tfd.Normal):

    def __init__(self, loc, scale, batch_dims, target_dims):
        self.batch_dims = batch_dims
        self.target_dims = target_dims
        tfd.Normal.__init__(self, loc=loc, scale=scale)

    def slp(self, p):
        sample = self.sample(p)
        log_prob = tf.reduce_sum(self.log_prob(
            sample)[:, -self.batch_dims:], axis=1)
        return sample, log_prob


class IAF():

    """
    single-stack local IAF with feature injection
    """

    def __init__(self, network_dims, theta, ts_feats, permuted=False):
        self.network_dims = network_dims
        self.num_layers = len(network_dims)
        self.theta = theta
        self.ts_feats = ts_feats
        self.permuted = permuted

    def _create_flow(self, base_dist, p, kernel_len, batch_dims, target_dims):
        base_sample, self.base_logprob = base_dist.slp(p)

        ts_aug_net = [tf.concat([self.ts_feats[:, 1:, :], self.ts_feats[:, 1:, :-2] - self.ts_feats[:, :-1, :-2]], 2)]

        for i in range(4):
            ts_aug_net.append(tf.layers.dense(inputs=ts_aug_net[-1], units=self.network_dims[0], activation=tf.nn.elu))

        convnet_inp = tf.concat(
            [tf.expand_dims(base_sample[:, :-1], 2), ts_aug_net[-1]], axis=2)

        layer1A = tf.layers.conv1d(inputs=convnet_inp, filters=network_dims[0],
                                   kernel_size=kernel_len, strides=1, padding='valid', activation=None)
        layer1B1 = tf.layers.dense(
            inputs=self.theta, units=self.network_dims[0], activation=None)
        layer1B2 = tf.layers.dense(
            inputs=layer1B1, units=self.network_dims[0], activation=None)
        layer1B = tf.layers.dense(
            inputs=layer1B2, units=self.network_dims[0], activation=None)

        layer1C = layer1A + tf.expand_dims(layer1B, 1)
        layers = [tf.nn.elu(layer1C)]

        for i in range(1, self.num_layers - 1):
            layers.append(tf.layers.conv1d(
                inputs=layers[-1], filters=self.network_dims[i], kernel_size=1, strides=1, activation=tf.nn.elu))
            layers.append(tf.layers.batch_normalization(layers[-1]))
        layers.append(tf.layers.conv1d(
            inputs=layers[-1], filters=2, kernel_size=1, strides=1, activation=None))

        mu_temp, sigma_temp = tf.split(layers[-1], 2, axis=2)
        mu = tf.squeeze(mu_temp)

        sigma = tf.nn.softplus(tf.squeeze(sigma_temp)) + 1e-10
        self.sigma_log = tf.log(sigma[:, -batch_dims:])
        self.output = base_sample[:, kernel_len:] * sigma + mu

    def slp(self, *args):
        logprob = self.base_logprob - tf.reduce_sum(self.sigma_log, axis=1)
        return self.output, logprob


class Flow_Stack():

    """
    Create locally variant IAF stack
    """

    def __init__(self, flows, kernel_len, batch_dims, target_dims):
        base_dims = kernel_len * no_flows + batch_dims + 1
        base_dist = init_dist(loc=[0.0] * base_dims, scale=[1e0] *
                              base_dims, batch_dims=batch_dims, target_dims=target_dims)
        flows.insert(0, base_dist)

        for i in range(1, len(flows)):
            flows[i]._create_flow(
                flows[i - 1], p, kernel_len, batch_dims, target_dims)

        self.output = flows[-1]

    def slp(self):
        return self.output.slp()


class Permute():
    '''
    class to permute IAF
    '''

    def __init__(self, permute_tensor):
        '''
        :params permute_index: permutations as list
        '''
        self.permute_tensor = permute_tensor

    def _create_flow(self, base_dist, *args):
        '''
        function to permute base dist order
        :params base_dist: base dist to permute
        '''

        sample, self.log_prob = base_dist.slp()
        shape = tf.shape(sample)
        self.sample = tf.scatter_nd(self.permute_tensor, sample, shape)

    def slp(self, *args):
        return self.sample, self.log_prob


class VI_SSM():

    def __init__(self, obs, x0, theta_dist, priors, dt, T, p, kernel_len, batch_dims, network_dims, target_dims, no_flows, feat_window, learn_rate = 1e-3, pre_train=False):
        # raw inputs -> class variables
        self.obs = obs
        self.theta = theta_dist.sample(p)
        self.theta_log_prob = theta_dist.log_prob(self.theta)
        self.priors = priors
        self.dt = dt
        self.p = p
        self.kernel_len = kernel_len
        self.batch_dims = batch_dims
        self.network_dims = network_dims
        self.target_dims = target_dims
        self.no_flows = no_flows
        self.theta_eval = self._theta_strech()
        self.pre_train = pre_train
        self.kernel_ext = self.kernel_len * self.no_flows + self.batch_dims + 1
        self.learn_rate = learn_rate

        var_store = []
        for i in range(0, self.obs.shape[0] - kernel_len):
            var_store.append(np.var(self.obs[i:i+kernel_len]))
        self.var_pad = np.concatenate(
            (np.zeros((no_flows + 1) * kernel_len), var_store), axis=0)

        var_diff_store = []
        obs_diff = self.obs[1:] - self.obs[:-1]
        for i in range(0, obs_diff.shape[0] - kernel_len):
            var_diff_store.append(np.var(obs_diff[i:i+kernel_len]))
        self.var_diff_pad = np.concatenate(
            (np.zeros((no_flows + 1) * kernel_len), np.log(var_diff_store), np.zeros(1)), axis=0)

        # augementing raw inputs
        self.obs_pad_store = []
        for i in range(0, feat_window*5, 5):
            self.obs_pad_store.append(np.concatenate(
                (np.zeros(no_flows * kernel_len - i), self.obs, np.zeros(i)), axis=0))
        self.time_pad = np.concatenate(
            (np.zeros(no_flows * kernel_len + 1), np.arange(0.1, T + dt, dt)), axis=0)
        self.bin_feats = np.float32(np.concatenate(
            (np.ones(no_flows * kernel_len + 1), np.zeros(self.target_dims)), axis=0))
        self.mask_vals = np.concatenate(
            (np.zeros((1, 1)), np.ones((1, self.target_dims))), axis=1)
        self.shift_vals = np.concatenate(
            (np.array([[x0]]), np.zeros((1, self.target_dims))), axis=1)

        # model placeholders
        self.time_feats = tf.placeholder(
            shape=[self.p, self.kernel_len * self.no_flows + self.batch_dims + 1, feat_window + 3], dtype=DTYPE)
        self.mask = tf.placeholder(
            shape=[self.p, self.batch_dims + 1], dtype=DTYPE)
        self.shift = tf.placeholder(
            shape=[self.p, self.batch_dims + 1], dtype=DTYPE)
        self.dim_one = tf.placeholder(
            shape=[self.p, self.batch_dims + 1], dtype=DTYPE)

    def _theta_strech(self):
        slice_stash = []
        for i in range(len(self.priors)):
            slice_stash.append(tf.reshape(tf.tile(tf.expand_dims(
                self.theta[:, i], 1), [1, self.batch_dims]), [-1]))
        return slice_stash

    def _ELBO(self):
        flow_head = self.lf_sample[:, :, :-1]
        flow_tail = self.lf_sample[:, :, 1:]

        latent_flow_diff = flow_tail - flow_head
        latent_flow_diff_flat = tf.concat([tf.reshape(latent_flow_diff[:, 0, :], [-1, 1]),
                                           tf.reshape(latent_flow_diff[:, 1, :], [-1, 1])], 1)

        def alpha(x1, x2, theta):
            drift_vec = tf.concat([tf.reshape(theta[0] * x1, [-1, 1]),
                                   tf.reshape(theta[1] - tf.exp(theta[2]) * x2, [-1, 1])], axis=1)
            return drift_vec

        dim1 = tf.reshape(flow_head[:, 0, :], [-1])
        dim2 = tf.reshape(flow_head[:, 1, :], [-1])

        beta_sqrt = tf.sqrt(self.dt) * tf.concat([tf.expand_dims(item, 1) for item in [
            dim1 * tf.exp(0.5 * dim2), tf.exp(self.theta_eval[3])]], 1)

        SDE_log_prob = tf.reduce_sum(tf.reshape(tfd.MultivariateNormalDiag(
            loc=self.dt * alpha(dim1, dim2, self.theta_eval), scale_diag=beta_sqrt).log_prob(latent_flow_diff_flat), [self.p, -1]), 1)

        prior_mean = [item[0] for item in self.priors]
        prior_scale = [item[1] for item in self.priors]

        prior_log_prob = tfd.MultivariateNormalDiag(
            loc=prior_mean, scale_diag=prior_scale).log_prob(self.theta)

        ELBO = (self.target_dims / self.batch_dims) * (SDE_log_prob -
                                                       self.lf_log_prob) + prior_log_prob - self.theta_log_prob

        return ELBO, SDE_log_prob

    def build_flow(self):
        flows = []
        for i in range(self.no_flows):
            flows.append(IAF(network_dims=self.network_dims, theta=self.theta,
                             ts_feats=self.time_feats[:, i * self.kernel_len:, :]))
        SSM = Flow_Stack(flows, self.kernel_len,
                         self.batch_dims, self.target_dims)
        lf_sample_temp, self.lf_log_prob = SSM.slp()

        self.lf_sample = tf.concat([tf.expand_dims(item, 1) for item in [self.dim_one,
                                                                         lf_sample_temp * self.mask + self.shift]], axis=1)

        loss, self.sde_loss = self._ELBO()
        self.mean_loss = tf.reduce_mean(loss)

        self.pre_train_step = AdamaxOptimizer(
            learning_rate=1e-3, beta1=0.9).minimize((self.lf_sample + 7.) ** 2)
        self.param_init = AdamaxOptimizer(learning_rate=1e-3, beta1=0.9).minimize(
            (self.theta - tf.tile([[0.001, -.6, np.log(0.08), np.log(0.5)]], [self.p, 1])) ** 2)

        # do something nicer with this!
        theta_pos_index = [False, False, True, True]
        with tf.name_scope('loss'):
            tf.summary.scalar('ELBO', self.mean_loss)
            tf.summary.scalar(
                'SDE_log_prob', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.sde_loss))
            tf.summary.scalar('theta_log_prob',
                              tf.reduce_mean(self.theta_log_prob))
            tf.summary.scalar(
                'path_log_prob', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.lf_log_prob))

            # theta summaries
            for i in range(len(theta_pos_index)):
                if theta_pos_index[i]:
                    tf.summary.histogram(str(i), tf.exp(
                        self.theta[:, i]), family='parameters')
                else:
                    tf.summary.histogram(
                        str(i), self.theta[:, i], family='parameters')

        with tf.name_scope('optimize'):
            opt = AdamaxOptimizer(learning_rate=self.learn_rate, beta1=0.95)
            gradients, variables = zip(
                *opt.compute_gradients(-loss))
            global_norm = tf.global_norm(gradients)
            self.gradients, _ = tf.clip_by_global_norm(gradients, 1e7)
            self.train_step = opt.apply_gradients(
                zip(self.gradients, variables))
            tf.summary.scalar(
                'global_norm', global_norm)

        self.merged = tf.summary.merge_all()
        self.loss = loss

    def train(self, tensorboard_path, save_path):
        writer = tf.summary.FileWriter(
            '%s/%s' % (tensorboard_path, datetime.now().strftime("%d:%m:%y-%H:%M:%S")), sess.graph)

        min_glob_loss = (1e99, -1)
        run = 0
        print("Training model...")
        converged = False

        if self.batch_dims * self.p >= self.target_dims:
            replace_bool = True
        else:
            replace_bool = False

        while not converged:
            batch_select = np.random.choice(
                np.arange(0, self.target_dims, self.batch_dims), size=self.p, replace=replace_bool)
            obs_pad_feats = []

            for item in self.obs_pad_store:
                obs_pad_feats.append(np.concatenate([np.reshape(
                    item[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0))
            feat1 = np.concatenate(obs_pad_feats, axis=2)
            feat2 = np.concatenate([np.reshape(
                self.time_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat3 = np.concatenate([np.reshape(
                self.var_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat4 = np.concatenate([np.reshape(
                self.var_diff_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            time_feats_feed = np.concatenate(
                [feat1, feat2, feat3, feat4], axis=2)

            mask_feed = np.concatenate([np.reshape(self.mask_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)
            shift_feed = np.concatenate([np.reshape(self.shift_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)

            dim_one_feed = np.concatenate([np.reshape(
                self.obs[index: index + self.batch_dims + 1], [1, -1]) for index in batch_select], axis=0)

            if self.pre_train:
                if run == 0:
                    print("Pre-training...")
                _, _, test = sess.run([self.pre_train_step, self.param_init, self.lf_sample], feed_dict={
                                      self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.dim_one: dim_one_feed})
                print(test)
                if run == 1000:
                    self.pre_train = False
                    print("Finished pre-training...")
                    run = 0
            else:
                _, summary, batch_loss = sess.run([self.train_step, self.merged, self.mean_loss], feed_dict={self.time_feats: time_feats_feed,
                                                                                                             self.mask: mask_feed, self.shift: shift_feed, self.dim_one: dim_one_feed})
                writer.add_summary(summary, run)

            if run % 1000 == 0:
                self.save(save_path)

            run += 1

    def save(self, PATH):
        saver = tf.train.Saver()
        saver.save(sess, PATH)
        print("Model saved")

    def load(self, PATH):
        self.pre_train = False
        saver = tf.train.Saver()
        saver.restore(sess, PATH)
        print("Model restored")

    def save_paths(self, PATH_obs):

        path_stack = []
        for temp_index in np.arange(0, self.target_dims, self.batch_dims):
            print(temp_index, "/", self.target_dims)

            batch_select = np.tile(temp_index, self.p)

            obs_pad_feats = []

            for item in self.obs_pad_store:
                obs_pad_feats.append(np.concatenate([np.reshape(
                    item[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0))
            feat1 = np.concatenate(obs_pad_feats, axis=2)
            feat2 = np.concatenate([np.reshape(
                self.time_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat3 = np.concatenate([np.reshape(
                self.var_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat4 = np.concatenate([np.reshape(
                self.var_diff_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            time_feats_feed = np.concatenate(
                [feat1, feat2, feat3, feat4], axis=2)

            mask_feed = np.concatenate([np.reshape(self.mask_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)
            shift_feed = np.concatenate([np.reshape(self.shift_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)

            dim_one_feed = np.concatenate([np.reshape(
                self.obs[index: index + self.batch_dims + 1], [1, -1]) for index in batch_select], axis=0)

            path_out = sess.run(self.lf_sample, feed_dict={
                                self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.dim_one: dim_one_feed})

            path_stack.append(path_out[:, :, 1:])

        paths=np.concatenate(path_stack, axis=2)

        f=open(PATH_obs, 'w')
        np.savetxt(f, np.reshape(paths, (self.p, -1)))
        f.close()


########### setting up the model ###########
# hyperparams
obs = np.loadtxt('dat/SV.dat', NP_DTYPE)[300:]
T = obs.shape[0] - 1
# T = 20*52
p = 200
kernel_len = 50
dt = 1.0
# T = 1500.
target_dims = np.int32(T / dt)
batch_dims = 52
network_dims = [50] * 5
no_flows = 5
priors = [(0., 10.0), (0., 10.0), (0.0, 10.0), (0.0, 10.0)]
feat_window = 5

# obs and theta
x0 = -8.5
# f1 = open('/home/b2028663/scripts/arf/locally_variant/SV_obs_partial_v2.txt', 'r')
# # f1 = open("/Users/localadmin/Documents/PhD/autoregressive_flows/locally_variant/SV_obs_partial_v2.txt", 'r')
# obs = np.loadtxt(f1, NP_DTYPE)
# f1.close()

# theta dist
bijectors = []
num_bijectors = 5
for i in range(num_bijectors):
    bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
            hidden_layers=[5, 5, 5], activation=tf.nn.relu))))
    if i < (num_bijectors - 1):
        bijectors.append(tfb.Permute(
            permutation=np.random.permutation(np.arange(0, len(priors)))))
flow_bijector = tfb.Chain(list(reversed(bijectors)))

theta_dist = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=flow_bijector,
    event_shape=[len(priors)])

# theta_dist = tfd.MultivariateNormalDiag(loc = [tf.Variable(0.05), tf.Variable(.05)], scale_diag= [tf.Variable(1.), tf.Variable(1.)])

# buiding the model
var_model = VI_SSM(obs, x0, theta_dist, priors, dt, T, p,
                   kernel_len, batch_dims, network_dims, target_dims, no_flows, feat_window, learn_rate = 1e-4, pre_train=True)
var_model.build_flow()

# new session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# training!
#var_model.load('model_saves/SV_model_%i_v211.ckpt' % batch_dims)

np.savetxt('locally_variant/SV_local_post.txt',
           sess.run(theta_dist.sample([100000])))
var_model.save_paths('locally_variant/SV_obs_paths.txt')

var_model.train(tensorboard_path='locally_variant/train/',
                save_path='model_saves/SV_model_%i_v211.ckpt' % batch_dims)
