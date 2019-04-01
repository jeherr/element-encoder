from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from ..force_modifiers.neighbors import *
# from ..tm_math.tf_math import * # Why is this imported here?
# from ..tm_math.linear_operations import *
from element_data import AtomData
import numpy as np
import tensorflow as tf
import time
import os

class ElementCoder(object):
	def __init__(self, latent_size=4, batches_per_epoch=100, hidden_layers=[128, 128],
				max_steps=2000, batch_size=64, learning_rate=0.0001, test_freq=5):
		self.tf_precision = eval("tf.float32")
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		self.weight_decay = None
		self.max_steps = max_steps
		self.batch_size = batch_size
		self.max_checkpoints = 3
		self.activation_function = tf.tanh
		self.step = 0
		self.test_freq = test_freq
		self.name = "ECoder_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.network_directory = "./"+self.name
		self.latent_size = latent_size
		self.batches_per_epoch = batches_per_epoch

		self.atom_data = AtomData
		self.atom_features = np.array([data[2:] for data in AtomData], dtype=np.float64)
		self.feature_length = np.shape(self.atom_features)[1]
		self.data_mean = np.mean(self.atom_features, axis=0)
		self.data_std = np.std(self.atom_features, axis=0)
		return

	def train(self):
		self.build_graph()
		for i in range(self.max_steps):
			self.step += 1
			self.train_step()
			if self.step % self.test_freq == 0:
				test_loss = self.test_step()
				if self.step == self.test_freq:
					self.best_loss = test_loss
					self.save_checkpoint()
				elif test_loss < self.best_loss:
					self.best_loss = test_loss
					self.save_checkpoint()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		test_loss = self.test_step()
		batch_data = self.atom_features[1:,0]
		feed_dict = self.fill_feed_dict(batch_data)
		latent_features = self.sess.run(self.latent_features,  feed_dict=feed_dict)
		latent_shape = latent_features.shape
		with open("emodes.dat", "w") as f:
			for i in range(latent_shape[0]):
				f.write(AtomData[i+1][0]+", "+str(latent_features[i,0])+", "+str(latent_features[i,1])+", "
					+str(latent_features[i,2])+", "+str(latent_features[i,3])+"\n")
		self.sess.close()
		return

	def build_graph(self, restart=False):
		self.Zs_pl = tf.placeholder(tf.int32, shape=[None])
		self.tf_atom_features = tf.Variable(self.atom_features, trainable=False, dtype = self.tf_precision)

		self.gather_idx = tf.where(tf.equal(tf.expand_dims(tf.cast(self.Zs_pl, self.tf_precision), axis=-1),
				self.tf_atom_features[:,0]))[:,1]
		self.batch_features = tf.gather(self.tf_atom_features, self.gather_idx)
		self.norm_batch_features = (self.batch_features - self.data_mean) / self.data_std
		self.latent_features = self.encoder(self.batch_features)
		self.norm_decoded_features = self.decoder(self.latent_features)
		self.decoded_features = (self.norm_decoded_features * self.data_std) + self.data_mean
		self.reconstruction_loss = self.loss_op(self.norm_batch_features - self.norm_decoded_features)
		self.train_op = self.optimizer(self.reconstruction_loss, self.learning_rate)
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
		if restart:
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)
		return

	def encoder(self, features):
		for i in range(len(self.hidden_layers)):
			if i == 0:
				layer = tf.layers.dense(inputs=features, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)
			else:
				layer = tf.layers.dense(inputs=layer, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)

		latent_features = tf.layers.dense(inputs=layer, units=self.latent_size,
				activation=None, use_bias=True)
		return latent_features

	def decoder(self, latent_features):
		for i in range(len(self.hidden_layers)):
			if i == 0:
				layer = tf.layers.dense(inputs=latent_features, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)
			else:
				layer = tf.layers.dense(inputs=layer, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)

		decoded_features = tf.layers.dense(inputs=layer, units=self.feature_length,
				activation=None, use_bias=True)
		return decoded_features

	def optimizer(self, loss, learning_rate):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def loss_op(self, error):
		loss = tf.nn.l2_loss(error)
		return loss

	def train_step(self):
		start_time = time.time()
		train_loss =  0.0
		for ministep in range(self.batches_per_epoch):
			batch_data = np.random.choice(self.atom_features[1:,0], size=self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			_, loss = self.sess.run([self.train_op, self.reconstruction_loss], feed_dict=feed_dict)
			train_loss += loss
		train_loss /= self.batches_per_epoch
		train_loss /= self.batch_size
		duration = time.time() - start_time
		print("step:", self.step, " duration:", duration, " reconstruction loss:", train_loss)
		return

	def test_step(self):
		print("testing...")
		start_time = time.time()
		test_loss =  0.0
		batch_data = self.atom_features[1:,0]
		feed_dict = self.fill_feed_dict(batch_data)
		test_loss, decoded_features, batch_features = self.sess.run([self.reconstruction_loss, self.decoded_features, self.batch_features],  feed_dict=feed_dict)
		test_loss /= np.shape(batch_data)
		duration = time.time() - start_time
		for i in np.random.choice(np.arange(1,self.atom_features.shape[0]-1), size=10):
			atom_feats = self.atom_features[i+1]
			decoded_feats = decoded_features[i]
			print("Element Features: AN {} Mass {} ns {} np {} nd {}".format(
				atom_feats[0], atom_feats[1], atom_feats[2], atom_feats[3], atom_feats[4]))
			print("Element Features: Electroneg. {} Radius {} Ionization {} Elec. Aff. {} Polariz. {}".format(
				atom_feats[5], atom_feats[6], atom_feats[7], atom_feats[8], atom_feats[9]))
			print("Decoded Features: AN {} Mass {} ns {} np {} nd {}".format(
				decoded_feats[0], decoded_feats[1], decoded_feats[2], decoded_feats[3], decoded_feats[4]))
			print("Decoded Features: Electroneg. {} Radius {} Ionization {} Elec. Aff. {} Polariz. {}".format(
				decoded_feats[5], decoded_feats[6], decoded_feats[7], decoded_feats[8], decoded_feats[9]))
		print("step:", self.step, " duration:", duration, " reconstruction loss:", test_loss)
		return test_loss

	def fill_feed_dict(self, batch_data):
		feed_dict={self.Zs_pl:batch_data}
		return feed_dict

	def save_checkpoint(self):
		checkpoint_file = os.path.join(self.network_directory,self.name+'-checkpoint')
		self.saver.save(self.sess, checkpoint_file, global_step=self.step)
		return
