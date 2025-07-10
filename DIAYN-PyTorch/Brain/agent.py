import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator, StateMI
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class SACAgent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.n_object_states = self.config["n_object_states"]
        self.MI_states = int(self.config["n_object_states"] / 2)
        self.batch_size = self.config["batch_size"]
        self.episode_length = self.config["episode_length"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        
        self.MI_discriminator = StateMI(n_object_states=self.MI_states, n_robot_states= self.MI_states, hidden=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])
        self.MI_discriminator_opt = Adam(self.MI_discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def clear_memory(self):
        self.memory.clear()
       
    def store(self, state, z, done, action, obj_state, next_state):
        state = np.array(state, dtype=np.float32)
        obj_state = np.array(obj_state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        state = from_numpy(state).float().to("cpu")
        obj_state = from_numpy(obj_state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        # print("object_state", obj_state)
        # print("state", state.shape)
        # print("z", z.shape)
        # print("done", done.shape)
        # print("action", action.shape)
        # print("object_state", obj_state.shape)
        # print("next_state", next_state.shape)
        self.memory.add(state, z, done, action, obj_state, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.episode_length, self.n_states + self.n_skills).to(self.device)
        object_states = torch.cat(batch.object_state).view(self.batch_size, self.episode_length, self.n_object_states).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, self.episode_length, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, self.episode_length, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.episode_length, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.episode_length, self.n_states + self.n_skills).to(self.device)
        t_samples = np.random.randint(self.episode_length - 1, size=self.batch_size)
        batch_indices = torch.arange(self.batch_size).to(self.device)
        # print("batch_state", states.shape)
        assert not torch.isnan(states).any()
        # print("t_samples", t_samples)
        # print("batch_indices", batch_indices)
        states_sampled = states[batch_indices, t_samples]
        zs_sampled = zs[batch_indices, t_samples]
        dones_sampled = dones[batch_indices, t_samples]
        actions_sampled = actions[batch_indices, t_samples]
        object_states_sampled = object_states[batch_indices, t_samples]
        object_next_states_sampled = object_states[batch_indices, t_samples+1]
        object_states_sampled = torch.reshape(object_states_sampled, (object_states_sampled.shape[0], 1, object_states_sampled.shape[-1]))
        object_next_states_sampled = torch.reshape(object_next_states_sampled, (object_next_states_sampled.shape[0], 1, object_next_states_sampled.shape[-1]))
        object_states_sampled = torch.cat((object_states_sampled, object_next_states_sampled), dim=-2)
        # print("object_states_sampled", object_states_sampled.shape)
        next_states_sampled = next_states[batch_indices, t_samples]
        # print("actions_sampled", actions_sampled)
        # print("next_states_sampled", next_states_sampled)

        # print("states_sampled", states_sampled.shape)
        # print("zs_sampled", zs_sampled.shape)
        # print("dones_sampled", dones_sampled)
        # print("actions_sampled", actions_sampled.shape)
        # print("object_states_sampled", object_states_sampled.shape)
        # print("next_states_sampled", next_states_sampled.shape)

        return states_sampled, zs_sampled, dones_sampled, actions_sampled, object_states_sampled, next_states_sampled, object_states

    def run_MI_discriminator(self, states, object_states):
        # states = from_numpy(states).float().to(self.device)
        # object_states = from_numpy(object_states).float().to(self.device)
        states = states.to(self.device)
        object_states = object_states.to(self.device)
        MI_value = self.MI_discriminator(object_states, states)
        MI_value = MI_value.reshape(-1, 1)
        MI_value = torch.clip(self.config["MI_scale"] * MI_value, min=0, max=1)
        return MI_value
    
    def train(self):
        if len(self.memory) < 1:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, object_states, next_states, pack_object_states = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)
            logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            # print("logq_z_ns", logq_z_ns)
            # print("next_states", next_states.shape)
            # print("states", states.shape)
            # print("object_states", object_states)
            MI_value = self.MI_discriminator(object_states[:, :, 3:6], object_states[:, :, :3])
            # print("object_states", object_states[:, :, 3:6], object_states[:, :, :3])
            MI_reward = -MI_value.reshape(-1, 1)
            # print("MI_reward", MI_reward.shape)
            # print("MI_value", MI_value[:5])
            # print("MI_value", MI_reward)
            # print("MI_reward", MI_reward * self.config["MI_scale"])
            # MI_reward = torch.clip(self.config["MI_scale"] * MI_reward, min=0, max=1) - 1.0
            # print("MI_reward", MI_reward)
            # print(MI_value[:5])
            sk_rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)
            sk_rewards = sk_rewards.reshape(-1, 1)
            sk_rewards = sk_rewards.float()
            # print("sk_rewards", sk_rewards.shape)
            # print("sk_rewards", sk_rewards[:5])
            sk_rewards = torch.clip(self.config["reward_scale"] * sk_rewards, min=-1, max=0) - 1
            # print("sk_rewards", sk_rewards[:5])
            # print("rewards", sk_rewards[:5] * self.config["reward_scale"])
            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = MI_reward + self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            # print("target_q", target_q[:5])
            # print("MI_reward", MI_reward[:5])
            # print("target_q", target_q[:5])
            # print(self.config["gamma"] * self.value_target_network(next_states) * (~dones))
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)
            # print("q1", q1[:5])
            # print("q2", q2[:5])
            # print("target_q", target_q[:5])
            # print("q1_loss", q1_loss.item())
            # print("q2_loss", q2_loss.item())
            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))
            MI_loss = self.MI_discriminator(pack_object_states[:, :, 3:6], pack_object_states[:, :, :3])
            # print("discriminator_loss", discriminator_loss)
            # print("MI_value", MI_value)
            MI_loss = MI_loss.mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.MI_discriminator_opt.zero_grad()
            MI_loss.backward()
            self.MI_discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return -discriminator_loss.item(), MI_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
