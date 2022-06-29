import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm


class Buffer:
    def __init__(self, var_size: list, buffer_capacity: int, batch_size: int) -> None:
        """
        Creates buffer object to store states, controls, etc information.

        Buffer class to store data. The buffer can be used to train neural network for reinforcement learning based algorithm.

        Parameters
        ----------
        var_size : list
            List of integers. Each element of list is the size of variable whose value needs to be stored.
        buffer_capacity : int
            Maximum number of observations that can be stored in the buffer. When buffer overflow occurs, new data overwrites old data starting from index 0.
        batch_size : int
            Number of observations that needs to be randomly drawn.
        """
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.var_size = var_size

        self.buffer_counter = 0

        self.buffer_data = [np.zeros((self.buffer_capacity, var_size[k]))
                            for k in range(len(self.var_size))]

    def record(self, data_tuple):
        """
        Function to store new data into buffer. The data is stored into new location if buffer has not overflown. Otherwise, old data is overwritten.

        Parameters
        ----------
        data_tuple : tuple
            Tuple of data to be stored. The order of data in the tuple should be same as var_size provided while creating the buffer object.
        """
        index = self.buffer_counter % self.buffer_capacity

        for k in range(len(self.var_size)):
            self.buffer_data[k][index, :] = data_tuple[k]

        self.buffer_counter += 1

    def get_batch(self):
        """
        Function to randomly draw specified number of observation.

        Returns
        -------
        list
            List of batch data. Each element of list corresponds to the specified variable.
        """
        record_range = min(self.buffer_counter, self.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        data_out = []
        for k in range(len(self.var_size)):
            data_out.append(tf.cast(tf.convert_to_tensor(
                self.buffer_data[k][batch_indices, :]), dtype=tf.float32))

        return data_out


class OUControlNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """
        Class to create Ornstein-Uhlenbeck process. Link: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process 

        Parameters
        ----------
        mean : float or numpy.array
            Mean of the random process.
        std_deviation : float or numpy.array
            Standard deviation of the random process
        theta : float, optional
            Theta parameter of the process, by default 0.15
        dt : float, optional
            Step time, by default 1e-2
        x_initial : float or numpy.array, optional
            Initial value of the random process, by default None
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def step(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * \
            np.random.normal(size=self.mean.shape)

        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DDPG:
    def __init__(self, num_states, num_controls, actor_model, critic_model, actor_optimizer, critic_optimizer, gamma=0.99, tau=0.005, buffer_capacity=100000, batch_size=64, epochs=10) -> None:
        """
        Class for deep deterministic policy gradient (DDPG) approach for reinforcement learning.

        Parameters
        ----------
        num_states : int
            Number of states/observations
        num_controls : int
            Number of control inputs
        actor_model : keras model 
            Keras neural network model for actor network.
        critic_model : _keras model
            Keras neural network model for critic network.
        actor_optimizer : keras optimizer
            keras optimizer for actor network
        critic_optimizer : keras model
            keras optimizer for critic network
        gamma : float, optional
            Discount factor, by default 0.99
        tau : float, optional
            Target network update factor, by default 0.005
        buffer_capacity : int, optional
            Capacity of buffer, by default 100000
        batch_size : int, optional
            Batch size to randomly sample data to train network, by default 64
        """
        self.actor_model = keras.models.clone_model(actor_model)
        self.critic_model = keras.models.clone_model(critic_model)
        self.actor_target = keras.models.clone_model(actor_model)
        self.critic_target = keras.models.clone_model(critic_model)

        self.num_states = num_states
        self.num_controls = num_controls

        self.gamma = gamma
        self.tau = tau

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.buffer = Buffer(var_size=[num_states, num_controls, 1, num_states, 1],
                             buffer_capacity=buffer_capacity, batch_size=batch_size)

        self.epochs = epochs

    def train(self, state, control, stage_cost, next_state, done, epochs):
        """
        Function to train actor and critic network by providing data. 
        Note: This function does not call the environment. Simulation of environment has to be done by user and data ave to be passed to call this function.

        Parameters
        ----------
        state : float or numpy.array
            States/observation from the system
        control : float or numpy.array
            Control input given to the system
        stage_cost : float
            Stage cost at that discrete time
        next_state : float or numpy.array
            Next state of the system as a result of applying control input
        done : bool
            Whether end of episode has been reached
        """
        self.buffer.record((state, control, stage_cost, next_state, done))

        state_batch, control_batch, cost_batch, next_state_batch, done_batch = self.buffer.get_batch()

        for _ in range(epochs):
            self.__learn_actor_critic(
                state_batch, control_batch, cost_batch, next_state_batch, done_batch)

        self.__update_target()

    @tf.function
    def __learn_actor_critic(self, state_batch, control_batch, cost_batch, next_state_batch, done_batch):
        # Update critic
        with tf.GradientTape() as tape:
            tape.watch(self.critic_model.trainable_variables)
            target_controls = self.actor_target(
                next_state_batch, training=True)

            y = cost_batch + self.gamma * (1-done_batch) * self.critic_target(
                [next_state_batch, target_controls], training=True
            )

            critic_value = self.critic_model(
                [state_batch, control_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        # Update actor
        with tf.GradientTape() as tape:
            tape.watch(self.actor_model.trainable_variables)
            controls = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model(
                [state_batch, controls], training=True)

            actor_loss = tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(
            actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables))

        return critic_loss, actor_loss

    @tf.function
    def __update_target(self):
        for (a, b) in zip(self.actor_target.variables, self.actor_model.variables):
            a.assign(b * self.tau + a * (1 - self.tau))

        for (a, b) in zip(self.critic_target.variables, self.critic_model.variables):
            a.assign(b * self.tau + a * (1 - self.tau))

    def train_sim(self, env_info, Ts, random_process, lower_bounds=None, upper_bounds=None, update_after=0, update_every=1, total_episodes=100):
        """
        Function to train actor and critic network. This function automatically calls environment. 

        Parameters
        ----------
        env_info : dict
            Dictionary that contains environment info. Following keywords are used:

        Ts : float
            Step time for environment
        random_process : any random process
            Random process with number of variables as same as number of controls
        lower_bounds : float or numpy.array, optional
            Lower bound on control control, by default None
        upper_bounds : float or numpy.array, optional
            Upper bound on control control, by default None
        total_episodes : int, optional
            Total number of episodes to be used to train, by default 100

        Returns
        -------
        numpy.array
            Array of stage cost. Different row corresponds to different episodes and different column corresponds to different time-step.
        """
        env = env_info['env']
        env_Ts = env_info['Ts']
        slow_factor = int(Ts/env_Ts)

        ep_cost_list = []
        critic_loss_list = []
        actor_loss_list = []
        for _1 in tqdm(range(total_episodes)):
            prev_state = env.reset()
            ep_cost = []
            critic_loss = []
            actor_loss = []
            count = 0
            while True:
                tf_prev_state = tf.expand_dims(
                    tf.convert_to_tensor(prev_state, dtype=tf.float32), 0)

                control = self.actor_model(
                    tf_prev_state) + random_process.step()
                if lower_bounds is not None and upper_bounds is not None:
                    control = np.clip(control, lower_bounds, upper_bounds)
                elif lower_bounds is not None and upper_bounds is None:
                    control = np.clip(control, lower_bounds, np.inf)
                elif lower_bounds is None and upper_bounds is not None:
                    control = np.clip(control, -np.inf, upper_bounds)
                else:
                    control = np.array(control)
                control = control.flatten()

                for _2 in range(slow_factor):
                    state, cost, done, info = env.step(control)

                self.buffer.record(
                    (np.array([prev_state]), control, cost, state, done))

                if count >= update_after:
                    if count % update_every == 0:
                        state_batch, control_batch, cost_batch, next_state_batch, done_batch = self.buffer.get_batch()
                        for _ in range(self.epochs):
                            loss1, loss2 = self.__learn_actor_critic(
                                state_batch, control_batch, cost_batch, next_state_batch, done_batch)
                            self.__update_target()

                        ep_cost.append(cost)
                        critic_loss.append(loss1)
                        actor_loss.append(loss2)

                if done:
                    break

                prev_state = state
                count += 1

            ep_cost_list.append(ep_cost.copy())
            critic_loss_list.append(critic_loss.copy())
            actor_loss_list.append(actor_loss.copy())

        return np.array(ep_cost_list), np.array(critic_loss_list), np.array(actor_loss_list)
