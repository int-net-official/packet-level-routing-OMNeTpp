import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append("./")
from src.joint_ppo.agent.actor_critic import ActorCritic
from src.joint_ppo.agent.rollout_buffer import RolloutBuffer

class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        joint_lr,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        summary_path,
        device,
        action_std_init=0.6,
        n_step_num=3,
        n_step_gamma=0.9,
    ) -> None:
        self.has_continuous_action_space = has_continuous_action_space

        self._build_summary_writer(summary_path)
        self.learn_step = 0
        self.n_step_num = n_step_num
        self.n_step_gamma = n_step_gamma
        self.device = device

        if has_continuous_action_space:
            self.action_std = action_std_init
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            self.state_dim, action_dim, device, has_continuous_action_space, self.action_std
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), joint_lr)

        self.policy_old = ActorCritic(
            self.state_dim, action_dim, device, has_continuous_action_space, self.action_std
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def _build_summary_writer(self, summary_path=None) -> None:
        if summary_path:
            self.summary_writer = SummaryWriter(log_dir=summary_path)
        else:
            self.summary_writer = SummaryWriter()

    def get_summary_writer(self):
        if self.summary_writer:
            return self.summary_writer

    def set_action_std(self, new_action_std) -> None:
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print(
                "WARNING : Calling PPO::set_action_std() on discrete action space policy"
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std) -> None:
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

        else:
            print(
                "WARNING : Calling PPO::decay_action_std() on discrete action space policy"
            )

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                action, action_logprob = self.policy_old.act(state)

            return action, action_logprob

        else:
            with torch.no_grad():
                action, action_logprob = self.policy_old.act(state)

            return action.item(), action_logprob

    def addBuffer(self, action, state, logprob, reward, done) -> None:
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.rewards.append((reward))
        self.buffer.is_terminals.append(done)

    def update_mc(self, Loss) -> None:
        # Monte Carlo estimate of returns

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        )

        rewards_ = torch.tensor([rewards.tolist()], dtype=torch.float32).to(self.device)

        all_loss = []
        # Optimize policy for K epochs
        for i in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO 0.01
            with torch.no_grad():
                state_values = state_values.unsqueeze(dim=0).detach().to(self.device)
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards_)
                - 0.01 * dist_entropy
            )

            Loss.append(loss.mean().detach().cpu().numpy())
            self.summary_writer.add_scalar(
                "loss",
                loss.mean().detach().cpu().numpy(),
                self.learn_step + i * 50 / self.K_epochs,
            )  # 这个 100 是 Update_freq
            all_loss.append(loss)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        mean_loss = torch.stack(all_loss).mean().detach().cpu().numpy()
        if mean_loss is not None:
            self.summary_writer.add_scalar(
                "mean_loss_each_step", mean_loss, self.learn_step
            )

        self.learn_step += 100  # 这个 100 是 Update_freq
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path) -> None:
        print("saving...")
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path) -> None:
        print("loading...")
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )

    def clean_buffer(self) -> None:
        self.buffer.clear()
