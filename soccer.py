import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class soccer:
    def __init__(self):
        self.cord = [np.array([0, 2]), np.array([0, 1])]
        self.possession = 1
        self.goal = [0, 3]


    def move(self, actions):
        action_list = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]
        cord1 = self.cord.copy()
        scores = np.array([0, 0])
        done = 0
        player1 = random.randint(0, 1)
        player2 = 1 - player1

        cord1[player1] = self.cord[player1] + action_list[actions[player1]]

        if (cord1[player1] == self.cord[player2]).all():
            if self.possession == player1:
                self.possession = player2

        elif cord1[player1][0] in range(0, 2) and cord1[player1][1] in range(0, 4):
            self.cord[player1] = cord1[player1]
            s1 = self.cord[0][0] * 4 + self.cord[0][1]
            s2 = self.cord[1][0] * 4 + self.cord[1][1]

            if self.cord[player1][1] == self.goal[player1] and self.possession == player1:
                scores = ([1, -1][player1]) * np.array([100, -100])
                done = 1
                return [s1, s2, self.possession], scores, done

            elif self.cord[player1][1] == self.goal[player2] and self.possession == player1:
                scores = ([1, -1][player1]) * np.array([-100, 100])
                done = 1
                return [s1, s2, self.possession], scores, done

        cord1[player2] = self.cord[player2] + action_list[actions[player2]]

        if (cord1[player2] == self.cord[player1]).all():
            if self.possession == player2:
                self.possession = player1

        elif cord1[player2][0] in range(0, 2) and cord1[player2][1] in range(0, 4):
            self.cord[player2] = cord1[player2]
            s1 = self.cord[0][0] * 4 + self.cord[0][1]
            s2 = self.cord[1][0] * 4 + self.cord[1][1]

            if self.cord[player2][1] == self.goal[player2] and self.possession == player2:
                scores = ([1, -1][player2]) * np.array([100, -100])
                done = 1
                return [s1, s2, self.possession], scores, done

            elif self.cord[player2][1] == self.goal[player1] and self.possession == player2:
                scores = np.array([-100, 100]) * [1, -1][player2]
                done = 1
                return [s1, s2, self.possession], scores, done

        return [self.cord[0][0] * 4 + self.cord[0][1], self.cord[1][0] * 4 + self.cord[1][1], self.possession], scores, done

def Qdiff_plot(errors, title):
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.plot(errors, linestyle='-', linewidth=0.6)
    plt.title(title)
    plt.ylim(0, 0.5)
    plt.xlabel('Simulation Iteartion')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x',
                         scilimits=(0,0), useMathText=True)
    plt.savefig(title+'.png')

def Q_learning():

    np.random.seed(299)
    num = int(1e6)
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.999995
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = 0.999995

    error_list = []

    Q_1 = np.zeros((8, 8, 2, 5))
    Q_2 = np.zeros((8, 8, 2, 5))

    i = 0

    while i < num:
        env = soccer()

        state = [env.cord[0][0] * 4 + env.cord[0][1], env.cord[1][0] * 4 + env.cord[1][1], env.possession]

        while True:
            if i % 1000 == 0:
                print('\rstep: {}\t Percentage: {:.2f}%'.format(i, i*100/num), end="")
            i += 1
            before = Q_1[2][1][1][2]
            if np.random.random() < epsilon:
                action1 = np.random.choice([0, 1, 2, 3, 4], 1)[0]
            else:
                action1 = np.random.choice(np.where(Q_1[state[0]][state[1]][state[2]] == max(Q_1[state[0]][state[1]][state[2]]))[0], 1)[0]
            if np.random.random() < epsilon:
                action2 = np.random.choice([0, 1, 2, 3, 4], 1)[0]
            else:
                action2 = np.random.choice(np.where(Q_2[state[0]][state[1]][state[2]] == max(Q_2[state[0]][state[1]][state[2]]))[0], 1)[0]
            actions = [action1, action2]
            state_out, rewards, done = env.move(actions)

            if done:
                Q_1[state[0]][state[1]][state[2]][actions[0]] = Q_1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] - Q_1[state[0]][state[1]][state[2]][actions[0]])
                Q_2[state[0]][state[1]][state[2]][actions[1]] = Q_2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] - Q_2[state[0]][state[1]][state[2]][actions[1]])
                after = Q_1[2][1][1][2]
                error_list.append(abs(after-before))
                break
            else:
                Q_1[state[0]][state[1]][state[2]][actions[0]] = Q_1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] + gamma * max(Q_1[state_out[0]][state_out[1]][state_out[2]]) - Q_1[state[0]][state[1]][state[2]][actions[0]])
                Q_2[state[0]][state[1]][state[2]][actions[1]] = Q_2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] + gamma * max(Q_2[state_out[0]][state_out[1]][state_out[2]]) - Q_2[state[0]][state[1]][state[2]][actions[1]])
                state = state_out
                after = Q_1[2][1][1][2]
                error_list.append(abs(after-before))

            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)
            alpha *= alpha_decay
            alpha = max(alpha_min, alpha)

    return error_list, Q_1, Q_2

# Q-learning
q_learning_errors, Q_1_qlearning, Q_2_qlearning = Q_learning()
Qdiff_plot(np.array(q_learning_errors), 'Q-learner')


def Friend_Q():

    np.random.seed(299)
    num = int(1e6)
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.999995
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = 0.999995

    error_list = []

    Q_1 = np.zeros((8, 8, 2, 5, 5))
    Q_2 = np.zeros((8, 8, 2, 5, 5))

    i = 0

    while i < num:
        env = soccer()

        state = [env.cord[0][0] * 4 + env.cord[0][1], env.cord[1][0] * 4 + env.cord[1][1], env.possession]

        while True:
            if i % 1000 == 0:
                print('\rstep {}\t Percentage: {:.2f}%'.format(i, i*100/num), end="")
            i += 1
            before = Q_1[2][1][1][4][2]
            if np.random.random() < epsilon:
                action1 = np.random.choice([0, 1, 2, 3, 4], 1)[0]
            else:
                max_idx = np.where(Q_1[state[0]][state[1]][state[2]] == np.max(Q_1[state[0]][state[1]][state[2]]))
                action1 = max_idx[1][np.random.choice(range(len(max_idx[0])), 1)[0]]
            if np.random.random() < epsilon:
                action2 = np.random.choice([0, 1, 2, 3, 4], 1)[0]
            else:
                max_idx = np.where(Q_2[state[0]][state[1]][state[2]] == np.max(Q_2[state[0]][state[1]][state[2]]))
                action2 = max_idx[1][np.random.choice(range(len(max_idx[0])), 1)[0]]
            actions = [action1, action2]
            state_out, rewards, done = env.move(actions)
            alpha = 1 / (i / alpha_min / num + 1)

            if done:
                Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] - Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]])
                Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] - Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]])
                after = Q_1[2][1][1][4][2]
                error_list.append(abs(after-before))
                break
            else:
                Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * np.max(Q_1[state_out[0]][state_out[1]][state_out[2]]) - Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]])
                Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * np.max(Q_2[state_out[0]][state_out[1]][state_out[2]]) - Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]])
                state = state_out
                after = Q_1[2][1][1][4][2]
                error_list.append(abs(after-before))

            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

    return error_list, Q_1, Q_2

# Friend-Q
friend_q_errors, Q_1_friend, Q_2_friend = Friend_Q()
Qdiff_plot(np.array(friend_q_errors), 'Friend-Q')


# Foe-Q Learning
def Foe_Q():

    def max_min(Q, state):
        c = matrix([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        G = matrix(np.append(np.append(np.ones((5,1)), -Q[state[0]][state[1]][state[2]], axis=1), np.append(np.zeros((5,1)), -np.eye(5), axis=1), axis=0))
        h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        A = matrix([[0.0],[1.0], [1.0], [1.0], [1.0], [1.0]])
        b = matrix(1.0)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
        return np.abs(sol['x'][1:]).reshape((5,)) / sum(np.abs(sol['x'][1:])), np.array(sol['x'][0])

    np.random.seed(299)
    num = int(1e6)
    gamma = 0.9
    epsilon_min = 0.001
    epsilon_decay = 0.999993
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = 0.999993


    Q_1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q_2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    Pi_1 = np.ones((8, 8, 2, 5)) * 1/5
    Pi_2 = np.ones((8, 8, 2, 5)) * 1/5

    V_1 = np.ones((8, 8, 2)) * 1.0
    V_2 = np.ones((8, 8, 2)) * 1.0

    errors_list = []

    i = 0
    epsilon = epsilon_decay ** i
    while i < num:
        env = soccer()
        state = [env.cord[0][0] * 4 + env.cord[0][1], env.cord[1][0] * 4 + env.cord[1][1], env.possession]
        done = 0
        while not done:
            if i % 1000 == 0:
                print('\rstep {} \t Percentage: {:.2f}%'.format(i, i*100/num), end="")
            i += 1
            before = Q_1[2][1][1][4][2]
            if np.random.random() < epsilon:
                action1 = np.random.choice([0, 1, 2, 3, 4], 1)[0]
            else:
                action1 = np.random.choice([0, 1, 2, 3, 4], 1, Pi_1[state[0]][state[1]][state[2]])[0]
            if np.random.random() < epsilon:
                action2 = np.random.choice([0, 1, 2, 3, 4], 1)[0]
            else:
                action2 = np.random.choice([0, 1, 2, 3, 4], 1, Pi_2[state[0]][state[1]][state[2]])[0]
            actions = [action1, action2]
            state_out, rewards, done = env.move(actions)

            Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * V_1[state_out[0]][state_out[1]][state_out[2]])

            pi, val = max_min(Q_1, state)
            Pi_1[state[0]][state[1]][state[2]] = pi
            V_1[state[0]][state[1]][state[2]] = val

            Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * V_2[state_out[0]][state_out[1]][state_out[2]])

            pi, val = max_min(Q_2, state)
            Pi_2[state[0]][state[1]][state[2]] = pi
            V_2[state[0]][state[1]][state[2]] = val
            state = state_out

            after = Q_1[2][1][1][4][2]
            errors_list.append(np.abs(after - before))

            alpha = alpha_decay ** i

    return errors_list, Q_1, Q_2, V_1, V_2, Pi_1, Pi_2

# Foe-Q
foe_q_errors, Q_1_foe, Q_2_foe, V_1_foe, V_2_foe, Pi_1_foe, Pi_2_foe = Foe_Q()
Qdiff_plot(np.array(foe_q_errors), 'Foe-Q')


def CE_Q():

    def solve_ce(Q_1, Q_2, state):
        Q_states = Q_1[state[0]][state[1]][state[2]]
        s = block_diag(Q_states - Q_states[0, :], Q_states - Q_states[1, :], Q_states - Q_states[2, :], Q_states - Q_states[3, :], Q_states - Q_states[4, :])
        row_index = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
        parameters_1 = s[row_index, :]

        Q_states = Q_2[state[0]][state[1]][state[2]]
        s = block_diag(Q_states - Q_states[0, :], Q_states - Q_states[1, :], Q_states - Q_states[2, :], Q_states - Q_states[3, :], Q_states - Q_states[4, :])
        col_index = [0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24]
        parameters_2 = s[row_index, :][:, col_index]

        c = matrix((Q_1[state[0]][state[1]][state[2]] + Q_2[state[0]][state[1]][state[2]].T).reshape(25))
        G = matrix(np.append(np.append(parameters_1, parameters_2, axis=0), -np.eye(25), axis=0))
        h = matrix(np.zeros(65) * 0.0)
        A = matrix(np.ones((1, 25)))
        b = matrix(1.0)

        try:
            sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
            if sol['x'] is not None:
                prob = np.abs(np.array(sol['x']).reshape((5, 5))) / sum(np.abs(sol['x']))
                val_1 = np.sum(prob * Q_1[state[0]][state[1]][state[2]])
                val_2 = np.sum(prob * Q_2[state[0]][state[1]][state[2]].T)
            else:
                prob = None
                val_1 = None
                val_2 = None
        except:
            print("error!!")
            prob = None
            val_1 = None
            val_2 = None

        return prob, val_1, val_2

    num = int(1e6)
    np.random.seed(299)
    gamma = 0.9
    epsilon_min = 0.001
    epsilon_decay = 0.999993
    alpha = 1
    alpha_min = 0.001
    alpha_decay = 0.999993

    Q_1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q_2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    V_1 = np.ones((8, 8, 2)) * 1.0
    V_2 = np.ones((8, 8, 2)) * 1.0

    Pi = np.ones((8, 8, 2, 5, 5)) * 1/25

    error_list = []

    i = 0
    epsilon = epsilon_decay ** i
    while i < num:
        env = soccer()
        state = [env.cord[0][0] * 4 + env.cord[0][1], env.cord[1][0] * 4 + env.cord[1][1], env.possession]
        done = 0
        j = 0
        while not done and j <= 100:
            if i % 1000 == 0:
                print('\rstep {}\t Percentage: {:.2f}%'.format(i, i*100/num), end="")
            i, j = i+1, j+1
            before = Q_1[2][1][1][2][4]
            if np.random.random() < epsilon:
                index = np.random.choice(np.arange(25), 1)
                actions = np.array([index // 5, index % 5]).reshape(2)
            else:
                index = np.random.choice(np.arange(25), 1, Pi[state[0]][state[1]][state[2]].reshape(25))
                actions = np.array([index // 5, index % 5]).reshape(2)

            state_out, rewards, done = env.move(actions)
            alpha = alpha_decay ** i

            Q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * Q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[0] + gamma * V_1[state_out[0]][state_out[1]][state_out[2]])

            Q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * Q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[1] + gamma * V_2[state_out[0]][state_out[1]][state_out[2]].T)
            prob, val_1, val_2 = solve_ce(Q_1, Q_2, state)

            if prob is not None:
                Pi[state[0]][state[1]][state[2]] = prob
                V_1[state[0]][state[1]][state[2]] = val_1
                V_2[state[0]][state[1]][state[2]] = val_2
            state = state_out

            after = Q_1[2][1][1][2][4]

            error_list.append(np.abs(after - before))

    return error_list, Q_1, Q_2, V_1, V_2, Pi

# CE_Q
ce_q_errors, Q_1_ce, Q_2_ce, V_1_ce, V_2_ce, Pi_ce = CE_Q()
Qdiff_plot(np.array(ce_q_errors), 'CE-Q')