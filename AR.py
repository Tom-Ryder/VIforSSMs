import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# python data types
import numpy as np
from datetime import datetime
from tensorflow.python.ops import clip_ops
from optimisers.adamax import AdamaxOptimizer

DTYPE = tf.float32
NP_DTYPE = np.float32

tfd = tf.contrib.distributions
tfb = tfd.bijectors

dat_dir = os.getcwd()

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.InteractiveSession()


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

    def __init__(self, network_dims, theta, ts_feats):
        self.network_dims = network_dims
        self.num_layers = len(network_dims)
        self.theta = theta
        self.ts_feats = ts_feats

    def _create_flow(self, base_dist, p, kernel_len, batch_dims, target_dims):
        base_sample, self.base_logprob = base_dist.slp(p)

        feat_layers = [self.ts_feats[:, :-1, :]]
        for i in range(4):
            feat_layers.append(tf.layers.dense(
                inputs=feat_layers[-1], units=self.network_dims[0], activation=tf.nn.elu))

        convnet_inp = tf.concat(
            [tf.expand_dims(base_sample[:, :-1], 2), feat_layers[-1]], axis=2)

        layer1A = tf.layers.conv1d(inputs=convnet_inp, filters=self.network_dims[0],
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

    def __init__(self, flows, kernel_len, batch_dims, target_dims, kernel_ext, p):
        base_dist = init_dist(loc=[0.0] * kernel_ext, scale=[1e0] *
                              kernel_ext, batch_dims=batch_dims, target_dims=target_dims)
        flows.insert(0, base_dist)

        for i in range(1, len(flows)):
            flows[i]._create_flow(
                flows[i - 1], p, kernel_len, batch_dims, target_dims)

        self.output = flows[-1]

    def slp(self):
        return self.output.slp()


class VI_SSM():

    def __init__(self, obs, obs_std, x0, theta_dist, priors, T, p, kernel_len, batch_dims, network_dims, no_flows, feat_window, obs_bin, time_till, pre_train=False, early_stopping=1e99, learn_rate=1e-3, grad_clip=2.5e8):
        # raw inputs -> class variables
        self.theta = theta_dist.sample(p)
        self.theta_log_prob = theta_dist.log_prob(self.theta)
        self.priors = priors
        self.obs_std = obs_std
        self.T = np.int32(T)
        self.p = p
        self.kernel_len = kernel_len
        self.batch_dims = batch_dims
        self.network_dims = network_dims
        self.no_flows = no_flows
        self.theta_eval = self._theta_strech()
        self.pre_train = pre_train
        self.early_stopping = early_stopping
        self.learn_rate = learn_rate
        self.grad_clip = grad_clip
        self.kernel_ext = self.kernel_len * self.no_flows + self.batch_dims + 1

        # augementing raw inputs
        self.obs_pad_store = []
        for i in range(feat_window):
            self.obs_pad_store.append(np.concatenate(
                (np.zeros(no_flows * kernel_len + 1 - i), obs, np.zeros(i)), axis=0))
        self.time_pad = np.concatenate(
            (np.zeros(no_flows * kernel_len + 1), np.arange(self.T + 1)), axis=0)
        self.bin_feats = np.float32(np.concatenate(
            (np.ones(no_flows * kernel_len + 1), np.zeros(self.T)), axis=0))
        self.obs_bin = np.concatenate(
            (np.zeros(no_flows * kernel_len + 1), obs_bin), axis=0)
        self.mask_vals = np.concatenate(
            (np.zeros((1, 1)), np.ones((1, self.T))), axis=1)
        self.shift_vals = np.concatenate(
            (np.array([[x0]]), np.zeros((1, self.T))), axis=1)
        self.time_till = np.concatenate(
            (np.arange((no_flows * kernel_len + 1) + time_till[0], time_till[0], -1), time_till), axis=0)

        # model placeholders
        self.time_feats = tf.placeholder(
            shape=[self.p, self.kernel_len * self.no_flows + self.batch_dims + 1, feat_window + 4], dtype=DTYPE)
        self.obs_eval = self.time_feats[:, -self.batch_dims:, 0]
        self.mask = tf.placeholder(
            shape=[self.p, self.batch_dims + 1], dtype=DTYPE)
        self.shift = tf.placeholder(
            shape=[self.p, self.batch_dims + 1], dtype=DTYPE)

    def _theta_strech(self):
        slice_stash = []
        for i in range(len(self.priors)):
            slice_stash.append(tf.tile(tf.expand_dims(
                self.theta[:, i], 1), [1, self.batch_dims]))
        return slice_stash

    def _ELBO(self):
        obs_log_prob = tf.reduce_sum(tfd.Normal(loc=self.obs_eval, scale=self.obs_std).log_prob(
            self.lf_sample[:, 1:]) * self.time_feats[:, -self.batch_dims:, -1], axis=1)

        flow_head = self.lf_sample[:, :-1]
        flow_tail = self.lf_sample[:, 1:]

        SDE_log_prob = tfd.MultivariateNormalDiag(loc=self.theta_eval[1] * flow_head +
                                                  self.theta_eval[0], scale_diag=tf.exp(self.theta_eval[2])).log_prob(flow_tail)

        prior_mean = [item[0] for item in self.priors]
        prior_scale = [item[1] for item in self.priors]

        prior_log_prob = tfd.MultivariateNormalDiag(
            loc=prior_mean, scale_diag=prior_scale).log_prob(self.theta)

        ELBO = (self.T / self.batch_dims) * (SDE_log_prob -
                                             self.lf_log_prob + obs_log_prob) + (prior_log_prob - self.theta_log_prob)

        return ELBO, SDE_log_prob, obs_log_prob

    def build_flow(self):
        flows = []
        for i in range(self.no_flows):
            flows.append(IAF(network_dims=self.network_dims, theta=self.theta,
                             ts_feats=self.time_feats[:, i * self.kernel_len:, :]))
        SSM = Flow_Stack(flows, self.kernel_len,
                         self.batch_dims, self.T, self.kernel_ext, self.p)
        self.lf_sample, self.lf_log_prob = SSM.slp()

        loss, self.sde_loss, self.obs_loss = self._ELBO()
        self.mean_loss = tf.reduce_mean(loss)

        self.pre_train_step = AdamaxOptimizer(
            learning_rate=1e-3, beta1=0.9).minimize(-self.obs_loss)

        # do something nicer with this!
        theta_pos_index = [False, False, True]
        with tf.name_scope('loss'):
            tf.summary.scalar('ELBO', self.mean_loss)
            tf.summary.scalar(
                'SDE_log_prob', (self.T / self.batch_dims) * tf.reduce_mean(self.sde_loss))
            tf.summary.scalar('theta_log_prob',
                              tf.reduce_mean(self.theta_log_prob))
            tf.summary.scalar(
                'obs_log_prob', (self.T / self.batch_dims) * tf.reduce_mean(self.obs_loss))
            tf.summary.scalar(
                'path_log_prob', (self.T / self.batch_dims) * tf.reduce_mean(self.lf_log_prob))

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
            self.gradients, _ = tf.clip_by_global_norm(
                gradients, self.grad_clip)
            self.train_step = opt.apply_gradients(
                zip(self.gradients, variables))
            tf.summary.scalar(
                'global_norm', global_norm)

        self.merged = tf.summary.merge_all()

    def train(self, tensorboard_path, save_path):

        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)

        save_parent_dir = '/'.join(save_path.split('/')[1:-1]) + '/'
        if not os.path.exists(save_parent_dir):
            os.makedirs(save_parent_dir)

        writer = tf.summary.FileWriter(
            '%s/%s' % (tensorboard_path, datetime.now().strftime("%d:%m:%y-%H:%M:%S")), sess.graph)

        min_glob_loss = (1e99, -1)
        run = 0
        print("Training model...")
        converged = False

        if self.batch_dims * self.p >= self.T:
            replace_bool = True
        else:
            replace_bool = False

        while not converged:
            sample_index = np.arange(0, self.T, self.batch_dims)
            batch_select = np.random.choice(
                sample_index, size=self.p, replace=replace_bool)

            obs_pad_feats = []

            for item in self.obs_pad_store:
                obs_pad_feats.append(np.concatenate([np.reshape(
                    item[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0))
            feat1 = np.concatenate(obs_pad_feats, axis=2)
            feat2 = np.concatenate([np.reshape(
                self.bin_feats[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat3 = np.concatenate([np.reshape(
                self.time_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat4 = np.concatenate([np.reshape(
                self.time_till[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat5 = np.concatenate([np.reshape(
                self.obs_bin[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)

            time_feats_feed = np.concatenate(
                [feat1, feat2, feat3, feat4, feat5], axis=2)

            mask_feed = np.concatenate([np.reshape(self.mask_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)
            shift_feed = np.concatenate([np.reshape(self.shift_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)

            if self.pre_train:
                if run == 0:
                    print("Pre-training...")
                _ = sess.run([self.pre_train_step], feed_dict={
                    self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed})
                if run == 500:
                    self.pre_train = False
                    print("Finished pre-training")
                    run = 0
            else:
                _, summary = sess.run([self.train_step, self.merged], feed_dict={
                                      self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed})
                writer.add_summary(summary, run)

            if run == self.early_stopping:
                converged = True

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
        for temp_index in np.arange(0, self.T, self.batch_dims):
            print(temp_index, "/", self.T)

            batch_select = np.tile(temp_index, self.p)

            obs_pad_feats = []

            for item in self.obs_pad_store:
                obs_pad_feats.append(np.concatenate([np.reshape(
                    item[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0))
            feat1 = np.concatenate(obs_pad_feats, axis=2)
            feat2 = np.concatenate([np.reshape(
                self.bin_feats[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat3 = np.concatenate([np.reshape(
                self.time_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat4 = np.concatenate([np.reshape(
                self.time_till[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)
            feat5 = np.concatenate([np.reshape(
                self.obs_bin[index: index + self.kernel_ext], [1, -1, 1]) for index in batch_select], axis=0)

            time_feats_feed = np.concatenate(
                [feat1, feat2, feat3, feat4, feat5], axis=2)

            mask_feed = np.concatenate([np.reshape(self.mask_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)
            shift_feed = np.concatenate([np.reshape(self.shift_vals[:, index:(
                index + self.batch_dims + 1)], [1, -1]) for index in batch_select], axis=0)

            path_out = sess.run(self.lf_sample, feed_dict={
                                self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.diffusivity: [1.0]})
            path_stack.append(path_out[:, 1:])

        paths = np.concatenate(path_stack, 1)

        f = open(PATH_obs, 'w')
        np.savetxt(f, paths)
        f.close()

def main(p, kernel_len, T, batch_dims, network_dims, no_flows, priors, feat_window, x0, obs_std, learn_rate=1e-3, grad_clip=2.5e8):
    # obs and theta
    f1 = open(dat_dir + "/dat/AR_obs_partial.txt", 'r')
    f2 = open(dat_dir + "/dat/AR_obs_binary.txt", 'r')
    f3 = open(dat_dir + "/dat/AR_time_till.txt", 'r')
    obs = np.loadtxt(f1, NP_DTYPE)
    obs_bin = np.loadtxt(f2, NP_DTYPE)
    time_till = np.loadtxt(f3, NP_DTYPE)
    f1.close()
    f2.close()
    f3.close()

    # theta dist
    bijectors = []
    num_bijectors = 5
    for i in range(num_bijectors):
        bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                hidden_layers=[5, 5, 5], activation=tf.nn.elu))))
        if i < (num_bijectors - 1):
            bijectors.append(tfb.Permute(
                permutation=np.random.permutation(np.arange(0, len(priors)))))
    flow_bijector = tfb.Chain(list(reversed(bijectors)))

    theta_dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=1.5, scale=.5),
        bijector=flow_bijector,
        event_shape=[len(priors)])

    # buiding the model
    var_model = VI_SSM(obs, obs_std, x0, theta_dist, priors, T, p, kernel_len, batch_dims,
                       network_dims, no_flows, feat_window, obs_bin, time_till, pre_train=True, learn_rate=learn_rate, grad_clip=grad_clip)
    var_model.build_flow()

    # new session
    sess.run(tf.global_variables_initializer())

    # training!
    var_model.train(tensorboard_path=dat_dir + "/train/",
                    save_path=dat_dir + "/model_saves/AR_save.ckpt")


########### setting up the model ###########
if __name__ == "__main__":
    # hyperparams
    p = 50
    kernel_len = 50
    T = 5000.
    batch_dims = 50
    network_dims = [50] * 3
    no_flows = 3
    priors = [(0., 10.0)] * 3
    feat_window = 10
    x0 = 10.0
    obs_std = 1.0
    main(p, kernel_len, T, batch_dims, network_dims, no_flows, priors, feat_window, x0, obs_std)
