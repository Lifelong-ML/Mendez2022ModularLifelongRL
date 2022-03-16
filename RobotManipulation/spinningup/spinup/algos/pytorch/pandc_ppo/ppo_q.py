import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.pandc_ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs_buf = {obs_key: np.zeros(core.combined_shape(size, obs_val), dtype=np.float32) for obs_key, obs_val in obs_dim.items()}
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        for ob_key, ob_val in obs.items():
            self.obs_buf[ob_key][self.ptr] = ob_val
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.val_buf[path_slice] + self.adv_buf[path_slice]      # the standard implementation

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        tensor_dict = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items() if k != 'obs'}
        tensor_dict['obs'] = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data['obs'].items()}
        return tensor_dict

def ppo_q(env_fn, ac, macro_buffer=None, *, task_id, seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.obs_dim
    obs_dim['obstacle-id'] = ac.num_modules[0]
    obs_dim['object-id'] = ac.num_modules[1]
    obs_dim['robot-id'] = ac.num_modules[2]
    act_dim = env.action_space.shape

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.qf1])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, *, task_id):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-5) 

        # Policy loss
        pi, logp = ac.pi(obs, act=act, task_id=task_id)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data, *, task_id):
        obs, act, ret, val = data['obs'], data['act'], data['ret'], data['val']
        with torch.no_grad():
            pi,_ = ac.pi(obs, task_id=task_id)
            act_samples = pi.sample((10,))
        obs = {o_k: o_v.unsqueeze(0).expand(10, -1, -1) for o_k, o_v in obs.items()}
        loss_v1 = (ac.qf1(obs, act_samples, task_id=task_id).mean(axis=0) - ret).pow(2).mean()
        loss_v2 = (ac.qf2(obs, act_samples, task_id=task_id).mean(axis=0) - ret).pow(2).mean()
        return loss_v1, loss_v2

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr, eps=1e-5)
    qf1_optimizer = Adam(ac.qf1.parameters(), lr=vf_lr, eps=1e-5)
    qf2_optimizer = Adam(ac.qf2.parameters(), lr=vf_lr, eps=1e-5)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(*, task_id):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, task_id=task_id)
        pi_l_old = pi_l_old.item()
        v_l_old = sum(compute_loss_v(data, task_id=task_id)).item() / 2

        data_batch = data
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data_batch, task_id=task_id)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()
        logger.store(StopIter=i)
            
        for i in range(train_v_iters):
            qf1_optimizer.zero_grad()
            qf2_optimizer.zero_grad()
            loss_v1, loss_v2 = compute_loss_v(data_batch, task_id=task_id)
            loss_v1.backward()
            loss_v2.backward()
            mpi_avg_grads(ac.qf1)    # average grads across MPI processes
            mpi_avg_grads(ac.qf2)    # average grads across MPI processes
            qf1_optimizer.step()
            qf2_optimizer.step()
        

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=((loss_v1 + loss_v2).item() / 2 - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    o['obstacle-id'] = np.zeros(ac.num_modules[0])
    o['obstacle-id'][ac.module_assignments[0][task_id]] = 1
    o['object-id'] = np.zeros(ac.num_modules[1])
    o['object-id'][ac.module_assignments[1][task_id]] = 1
    o['robot-id'] = np.zeros(ac.num_modules[2])
    o['robot-id'][ac.module_assignments[2][task_id]] = 1

    success = 0
    steps_at_goal = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step({o_key: torch.as_tensor(o_val, dtype=torch.float32) for o_key,o_val in o.items()}, task_id=task_id)

            next_o, r, d, _ = env.step(a)
            next_o['obstacle-id'] = np.zeros(ac.num_modules[0])
            next_o['obstacle-id'][ac.module_assignments[0][task_id]] = 1
            next_o['object-id'] = np.zeros(ac.num_modules[1])
            next_o['object-id'][ac.module_assignments[1][task_id]] = 1
            next_o['robot-id'] = np.zeros(ac.num_modules[2])
            next_o['robot-id'][ac.module_assignments[2][task_id]] = 1
            ep_ret += r
            ep_len += 1

            if r == 1:
                success = 1
                steps_at_goal += 1

            # save and log
            buf.store(o, a, r, v, logp)
            if macro_buffer:
                macro_buffer.store(o, a, r, next_o, (d if ep_len < max_ep_len else False))
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1


            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step({o_key: torch.as_tensor(o_val, dtype=torch.float32) for o_key,o_val in o.items()}, task_id=task_id)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    logger.store(Success=success)
                    logger.store(StepsAtGoal=steps_at_goal)

                if terminal or timeout: # Is this equivalent to the vectorized env in Baselines?
                    o, ep_ret, ep_len = env.reset(), 0, 0
                    o['obstacle-id'] = np.zeros(ac.num_modules[0])
                    o['obstacle-id'][ac.module_assignments[0][task_id]] = 1
                    o['object-id'] = np.zeros(ac.num_modules[1])
                    o['object-id'][ac.module_assignments[1][task_id]] = 1
                    o['robot-id'] = np.zeros(ac.num_modules[2])
                    o['robot-id'][ac.module_assignments[2][task_id]] = 1
                    success = 0
                    steps_at_goal = 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update(task_id=task_id)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('Success', average_only=True)
        logger.log_tabular('StepsAtGoal', average_only=True)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)