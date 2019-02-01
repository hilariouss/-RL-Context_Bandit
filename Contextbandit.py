import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Contextual_bandit:
    # 1) 밴딧과 각 액션들의 정의 및 표준정규분포 활용해서 리워드 발생 부분 구현.
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[-.2, -.1, .1, 1], [3, 1, -1, -2], [4, 2, -3, 1]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pullArm(self, action):
        result = np.random.randn(1)
        arm = self.bandits[self.state, action]

        if result > arm: # <-- arm 이 작을 수록, 양의 보상을 얻을 확률을 증가시킨다. 즉, 현재 상태(bandit)에서 더 작은 값을 갖는 arm을 선택하도록.
            return 1
        else:
            return -1


class Agent: # Neural network configuration
    def __init__(self, lr, s_size, a_size):
        # lr : learning rate
        # s_size : state size
        # a_size : action size
        # The agent input the state, and then return action

        # 2-1) Input, output 요소 구현 (Neural network)
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size) # params: (label, size of OH).
        # 가령 '1', '2', '3' 밴딧이 있고, '2'를 골랐다면, self.state_in : '2', s_size : 3.
        # Return example: '2' --> 0 1 0 이런식으로 input을 one hot encoding 됨.
        # Input이 단순히 3 이런 식이 아니라, 신경망의 output과 연결 weights들의 수를 충분히 하기 위해 OH 수행.

        output = slim.fully_connected(state_in_OH, a_size, # input output shape
                                      biases_initializer=None, activation_fn=tf.nn.sigmoid,\
                                      weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1]) # 가로로 결과 쫙 피기
        # !!!output, self.output type, shape 확인해보기 (텐서?)!!!

        self.chosen_action = tf.argmax(self.output, 0) # <-- 선택한 액션.


        # 2-2) 학습 과정 신경망 구현(Neural network)
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1]) # self.output (액션에 대한 확률) 가운데,
                                                                                 # self.action_holder에 담긴 값 추출
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)


tf.reset_default_graph()

CtxBandit = Contextual_bandit()
agent = Agent(lr = 1e-3, s_size = CtxBandit.num_bandits, a_size = CtxBandit.num_actions)
weights = tf.trainable_variables()[0] # <-- weights

episodes = 10000
total_reward = np.zeros([CtxBandit.num_bandits, CtxBandit.num_actions])
e = 0.1

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < episodes:
        state = CtxBandit.getBandit()
        if np.random.rand(1) < e: # random action
            action = np.random.randint(CtxBandit.num_actions)
        else:                     # follows the NN.
            action = sess.run(agent.chosen_action, feed_dict={agent.state_in:[state]})

        reward = CtxBandit.pullArm(action)

        # Update the NN.
        feed_dict = {agent.reward_holder:[reward], agent.state_in:[state],\
                     agent.action_holder:[action]}
        _, ww = sess.run([agent.update, weights], feed_dict=feed_dict)

        # 보상의 총 합 업데이트
        total_reward[state, action] += reward
        if i % 500 == 0:
            print("Mean reward for each of the " + str(CtxBandit.num_bandits) + " bandits: "\
                  + str(np.mean(total_reward, axis=1)))
        i += 1

for a in range(CtxBandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising")
    if np.argmax(ww[a]) == np.argmin(CtxBandit.bandits[a]):
        print("and it was right!")
    else:
        print("and it was wrong!")