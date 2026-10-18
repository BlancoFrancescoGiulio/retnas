import os
import signal
from datetime import datetime, timedelta
import copy
import time
from tqdm.auto import trange
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import multiprocessing
from torch.multiprocessing import Process, Value, Queue
from retnas.networks import *
from retnas.environment import *
from retnas.buffer import *
from retnas import utils

def exploration_worker(g_stop_flag, config, batches_queue, update_queue, episode_queue):
    utils.set_seed(config.seed)
    torch.set_num_threads(getattr(config, "torch_threads", 1))
    num_layers = config.num_layers
    policy_type = getattr(config, 'policy_type', 'gru')
    push_to_buffer_after = getattr(config, 'push_to_buffer_after', 0)
    actor_input_size = 4 + config.num_sas
    if config.hrt and config.hrt_flag_provided_to_the_network:
        actor_input_size += 1
    if getattr(config, 'num_tenants', 0) > 0 and getattr(config, 'enable_balancing_reward', False):
        actor_input_size += 2
    actor_output_size = 1 + config.num_sas
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor = NetworkFactory.create_actor(actor_input_size, config.actor_hidden_size, actor_output_size, num_layers=num_layers, use_packing=config.use_packing, type=policy_type).to(device)
    replay_buffer = ReplayBuffer(max_size=config.buffer_size, seed=config.seed)
    env = Environment(config, seed=config.seed)
    state = env.reset()
    print(f'Explore Worker Setup Finish')
    start_time = time.time()
    for i in trange(0, config.num_episodes, desc='Episodes'):
        actor_state_dict = copy.deepcopy(update_queue.get())
        exp_start_time = time.time()
        actor.load_state_dict({k: torch.from_numpy(v) for k, v in actor_state_dict.items()})
        total_reward = 0
        state = env.reset(clear=config.episode_reset_env_state)
        for t in range(config.num_timesteps):
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                with torch.no_grad():
                    action = actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0), torch.tensor(state.shape[0], device=device)).squeeze(0).cpu().data.numpy()
            if i % 2 == 1:
                noise_reduction_factor = (config.num_episodes - i) / config.num_episodes
                action = (action + np.random.normal(0, config.exploration_noise * noise_reduction_factor, size=action.shape)).clip(-1, 1)
            (actor_next_state, rb_next_state, reward, done, hrt_feasible) = env.step(action)
            if t >= push_to_buffer_after:
                total_reward += reward
                replay_buffer.push((state.astype(np.float32), rb_next_state.astype(np.float32), action.astype(np.float32), reward, done))
            state = actor_next_state
        total_deadline_hits = env.deadline_hits_be + env.deadline_hits_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        slast = total_deadline_hits / total_arrived_jobs if total_arrived_jobs > 0 else 1
        slast_be = env.deadline_hits_be / env.total_arrived_jobs_be if env.total_arrived_jobs_be > 0 else 1
        slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 1
        total_deadline_miss = env.deadline_misses_be + env.deadline_misses_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        time_per_iteration = (time.time() - start_time) / (i + 1)
        time_per_exploration = time.time() - exp_start_time
        eta = timedelta(seconds=int(time_per_iteration * (config.num_episodes - (i + 1))))
        if env.deadline_misses_hrt:
            print(':( HRT Misses: ', env.deadline_misses_hrt)
        tqdm.write('Episode: {:4d} | Total Reward: {:8.2f} | Slast: {:5.2f}% | Slast_BE: {:5.2f}% | Slast_HRT: {:5.2f}% HRT/BE MISS: {}/{} | TPI: {:5.3f}s (TPE: {:5.3f}s) | ETA: {:}'.format(i, total_reward, slast * 100, slast_be * 100, slast_hrt * 100, env.deadline_misses_hrt, env.deadline_misses_be, time_per_iteration, time_per_exploration, eta))
        if g_stop_flag.value:
            break
        episode_queue.put((slast, total_reward))
        for u in range(config.update_iteration):
            batches_queue.put(replay_buffer.sample(config.batch_size))
    print('\nFine\n')

def update_worker(g_stop_flag, config, batches_queue, update_queue, episode_queue):
    utils.set_seed(config.seed)
    torch.set_num_threads(getattr(config, "torch_threads", 1))
    num_layers = config.num_layers
    policy_type = getattr(config, 'policy_type', 'gru')
    actor_input_size = 4 + config.num_sas
    if config.hrt and config.hrt_flag_provided_to_the_network:
        actor_input_size += 1
    if getattr(config, 'num_tenants', 0) > 0 and getattr(config, 'enable_balancing_reward', False):
        actor_input_size += 2
    actor_output_size = 1 + config.num_sas
    critic_input_size = actor_input_size + actor_output_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor = NetworkFactory.create_actor(actor_input_size, config.actor_hidden_size, actor_output_size, num_layers=num_layers, use_packing=config.use_packing, type=policy_type).to(device)
    critic = NetworkFactory.create_critic(critic_input_size, config.critic_hidden_size, num_layers=num_layers, use_packing=config.use_packing, type=policy_type).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr)
    actor_target = NetworkFactory.create_actor(actor_input_size, config.actor_hidden_size, actor_output_size, num_layers=num_layers, use_packing=config.use_packing, type=policy_type).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target = NetworkFactory.create_critic(critic_input_size, config.critic_hidden_size, num_layers=num_layers, use_packing=config.use_packing, type=policy_type).to(device)
    critic_target.load_state_dict(critic.state_dict())
    current_datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    writer = SummaryWriter(os.path.join(config.runs_path, current_datetime_str + config.run_suffix), flush_secs=20)
    utils.copyfile(config.config_path, writer.log_dir)
    update_queue.put({k: v.detach().cpu().numpy().copy() for k, v in actor.state_dict().items()})
    max_slasr = float('-inf')
    max_reward = float('-inf')
    num_saved_actors = 0
    print(f'Update Worker Setup Finish')
    for i in range(config.num_episodes):
        (slasr, total_reward) = episode_queue.get()
        writer.add_scalar('Total reward', total_reward, i)
        writer.add_scalar('SLA Satisfaction Rate', slasr, i)
        if slasr > max_slasr:
            max_slasr = slasr
            num_saved_actors += 1
            torch.save(actor.state_dict(), f'{writer.log_dir}/actor_best_{num_saved_actors}.pth')
        if total_reward > max_reward:
            max_reward = total_reward
            num_saved_actors += 1
            torch.save(actor.state_dict(), f'{writer.log_dir}/actor_best_{num_saved_actors}r.pth')
        if i % config.save_freq == 0 or i + 1 == config.num_episodes:
            torch.save(actor.state_dict(), writer.log_dir + '/actor_last.pth')
            checkpoint = {'actor': actor.state_dict(), 'critic': critic.state_dict(), 'actor_optimizer': actor_optimizer.state_dict(), 'critic_optimizer': critic_optimizer.state_dict(), 'epoch': i}
            torch.save(checkpoint, writer.log_dir + '/checkpoint.pth')
        for u in range(config.update_iteration):
            (cur_state, next_state, action, reward, num_sjs, next_num_sjs, dones) = batches_queue.get()
            cur_state = torch.from_numpy(cur_state).to(device, non_blocking=True)
            action = torch.from_numpy(action).to(device, non_blocking=True)
            next_state = torch.from_numpy(next_state).to(device, non_blocking=True)
            reward = torch.tensor(reward, dtype=torch.float32).to(device, non_blocking=True)
            dones = torch.tensor(1 - dones, dtype=torch.int8).to(device, non_blocking=True)
            next_num_sjs = torch.from_numpy(next_num_sjs).to(device, non_blocking=True)
            num_sjs = torch.from_numpy(num_sjs).to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                with torch.no_grad():
                    actor_target_out = actor_target(next_state, next_num_sjs)
                    target_Q = critic_target(next_state, actor_target_out, next_num_sjs)
                    target_Q = reward + (dones * config.gamma * target_Q).detach()
                current_Q = critic(cur_state, action, num_sjs)
                critic_loss = F.mse_loss(current_Q, target_Q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                actor_out = actor(cur_state, num_sjs)
                actor_loss = -critic(cur_state, actor_out, num_sjs).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            for (param, target_param) in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)
            for (param, target_param) in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)
            writer.add_scalar('Critic Loss', critic_loss.item(), i * config.update_iteration + u)
            writer.add_scalar('Actor Loss', actor_loss.item(), i * config.update_iteration + u)
            if g_stop_flag.value:
                break
        # Persist the updated final policy, including for a one-episode trial.
        torch.save(actor.state_dict(), writer.log_dir + '/actor_last.pth')
        checkpoint = {'actor': actor.state_dict(), 'critic': critic.state_dict(),
                      'actor_optimizer': actor_optimizer.state_dict(),
                      'critic_optimizer': critic_optimizer.state_dict(), 'epoch': i}
        torch.save(checkpoint, writer.log_dir + '/checkpoint.pth')
        if i + 1 < config.num_episodes:
            update_queue.put({k: v.detach().cpu().numpy().copy() for k, v in actor.state_dict().items()})
    writer.close()
    print('\nFine\n')

def train(config):
    """Run exploration and optimization; propagate worker failures to the CLI."""
    multiprocessing.set_start_method('spawn', force=True)
    stop = Value('b', False)
    batches, updates, episodes = Queue(config.update_iteration), Queue(2), Queue(2)
    workers = [Process(target=exploration_worker, args=(stop, config, batches, updates, episodes)),
               Process(target=update_worker, args=(stop, config, batches, updates, episodes))]
    try:
        for worker in workers:
            worker.start()
        while any(worker.is_alive() for worker in workers):
            for worker in workers:
                worker.join(timeout=0.2)
                if worker.exitcode not in (None, 0):
                    raise RuntimeError(f"Training worker failed (exit {worker.exitcode})")
    finally:
        stop.value = True
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join()
        for queue in (batches, updates, episodes):
            queue.close()
            queue.cancel_join_thread()
