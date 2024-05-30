import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tictacfuncs as game
import random
import pickle


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        max_mem_size=100000,
        eps_end=0.05,
        eps_dec=5e-5,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(
            lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256
        )
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward

        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation) -> int:
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

        # action_batch = self.action_memory[batch]
        action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.bool).to(
            self.Q_eval.device
        )
        # terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        # Convert reward_batch and terminal_batch to float
        # reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float32).to(
        #     self.Q_eval.device
        # )
        # terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.float32).to(
        #     self.Q_eval.device
        # )

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )


def generate_agent():

    agent1 = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=9,
        eps_end=0.0001,
        input_dims=[9],
        lr=0.001,
        eps_dec=5e-4,
    )

    agent2 = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=9,
        eps_end=0.5,
        input_dims=[9],
        lr=0.001,
    )

    episodes = 10000
    scores1, scores2, eps_history = [], [], []

    for i in range(episodes):
        score1, score2 = 0, 0
        done = False
        board = game.init_board()

        # Randomly select which AI starts
        if random.choice([True, False]):
            current_agent, other_agent = agent1, agent2
        else:
            current_agent, other_agent = agent2, agent1

        player = game.Players.ONE.value

        while not done:
            action = current_agent.choose_action(board)
            board_, reward, done, info = game.place_move_ai_action(
                board, action, player
            )

            if current_agent == agent1:
                score1 += reward
            else:
                score2 += reward

            current_agent.store_transition(board, action, reward, board_, done)
            # if reward == 10:
            #     other_agent.store_transition(board, action, -10, board_, done)

            current_agent.learn()

            board = board_
            # Switch players
            current_agent, other_agent = other_agent, current_agent
            player = (
                game.Players.TWO.value
                if player == game.Players.ONE.value
                else game.Players.ONE.value
            )

        scores1.append(score1)
        scores2.append(score2)

        avg_score1 = np.mean(scores1[-100:])
        avg_score2 = np.mean(scores2[-100:])
        score_diff = avg_score1 - avg_score2
        tot_avg = (np.mean(scores2[-100:]) + avg_score1) / 2
        print(
            "episode ",
            i,
            "score %.2f" % score1,
            "average score %.2f" % avg_score1,
            "epsilon %.2f" % agent1.epsilon,
            "TOTAL AVG %.2f" % tot_avg,
            "BETTER SCORE %.2f" % score_diff,
        )

        agent1.save("agent.pkl")


if __name__ == "__main__":
    generate_agent()
    agent: Agent = Agent.load("agent.pkl")

    pass
