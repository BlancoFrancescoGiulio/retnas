from tqdm.auto import trange
from tqdm import tqdm
import torch
from retnas.networks import *
from retnas.environment import *
from retnas.buffer import *
from retnas.job import *
from retnas.simulation import *

def test(config):
    num_layers = config.num_layers
    policy_type = getattr(config, 'policy_type', 'gru')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.next_state_mode = 'nonewjobs'
    actor_input_size = 4 + config.num_sas
    if config.hrt and config.hrt_flag_provided_to_the_network:
        actor_input_size += 1
    if getattr(config, 'num_tenants', 0) > 0 and getattr(config, 'enable_balancing_reward', False):
        actor_input_size += 2
    actor_output_size = 1 + config.num_sas
    actor = SchedulerNetworkActorRNN(actor_input_size, config.actor_hidden_size, actor_output_size, num_layers=num_layers, use_packing=config.use_packing, type=policy_type).to(device)
    actor.load_state_dict(torch.load(config.test_model_path, map_location=device))
    actor.eval()
    env = Environment(config, seed=config.seed)
    if config.hrt:
        env.hrt_step_mode = 'deployment'
    state = env.reset()
    tmacs = []
    rq_lengths = []
    rq_residual_lengths = []
    rq_arrival_lengths = []
    for i in trange(config.num_steps):
        with torch.no_grad():
            input_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            num_sjs = torch.tensor([input_state.shape[1]])
            action = actor(input_state, num_sjs).squeeze(0).cpu().data.numpy()
            (state, rb_state, reward, done, hrt_feasible) = env.step(action)
            rq_lengths.append(state.shape[0])
            rq_residual_lengths.append(rb_state.shape[0])
            rq_arrival_lengths.append(rq_lengths[-1] - rq_residual_lengths[-1])
            if config.count_ops:
                from thop import profile
                input_state = torch.randn((1, 1, actor_input_size), dtype=torch.float32, device=device)
                (macs, params) = profile(actor, inputs=(input_state, num_sjs), verbose=False, report_missing=False)
                tmacs.append(macs)
        total_deadline_hits = env.deadline_hits_be + env.deadline_hits_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        slast = total_deadline_hits / total_arrived_jobs if total_arrived_jobs > 0 else 0
        slast_be = env.deadline_hits_be / env.total_arrived_jobs_be if env.total_arrived_jobs_be > 0 else 0
        slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 0
        total_deadline_miss = env.deadline_misses_be + env.deadline_misses_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        percentage_wcet_relies = env.hrt_wcet_win / (env.hrt_wcet_win + env.hrt_rl_win) if env.hrt_wcet_win + env.hrt_rl_win > 0 else 0
        if env.deadline_misses_hrt:
            print('[PROBLEM] THIS CAN NEVER HAPPEN: NO DEADLINE MISSES TOLERATED IN DEPLOYMENT PHASE -> ', env.deadline_misses_hrt)
            quit(1)
        tqdm.write('Step: \t{} \t Slast: \t{:0.2f}% \t Slast_BE: \t{:0.2f}% \t Slast_HRT: \t{:0.2f}% \t HRT/BE MISS: {},{} \t WCET won {:0.2f}% times'.format(i + 1, slast * 100, slast_be * 100, slast_hrt * 100, env.deadline_misses_hrt, env.deadline_misses_be, percentage_wcet_relies * 100))
    print(f'SLA Satisfaction Rate: {slast * 100:.2f}%')
    print(f'SLA Satisfaction Rate BE: {slast_be * 100:.2f}% ({env.deadline_hits_be}/{env.total_arrived_jobs_be})')
    print(f'SLA Satisfaction Rate HRT: {slast_hrt * 100:.2f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    open_hrt = env.total_arrived_jobs_hrt - env.deadline_hits_hrt - env.deadline_misses_hrt
    print(f'HRT jobs still open at test end: {open_hrt}')
    print('HRT fallback reasons:', env.hrt_guardian_rejections)
    print('Throughput:', env.executed_sjs * 1000000000.0 / env.current_time)
    print(f'Mean RQ Length: {mean(rq_lengths)} \t Max RQ Length: {max(rq_lengths)}')
    print(f'Mean RQ Residual Length: {mean(rq_residual_lengths)}')
    print(f'Mean RQ Arrival Length: {mean(rq_arrival_lengths)}')
    if config.count_ops:
        print('Total Scheduler MAC Ops:', sum(tmacs))
        print('Average Scheduler MAC Ops per arrival:', sum(tmacs) / config.num_steps)
        print('Max Scheduler MAC Ops per arrival:', max(tmacs))
        print('Scheduler Params:', params)
    if env.num_tenants > 0:
        slasr_per_tenant = [env.deadline_hits_per_tenant[i] / max(1, env.arrived_jobs_per_tenant[i]) for i in range(env.num_tenants)]
        print('SLASR per tenant:', slasr_per_tenant)
        print('SLASR per tenant std:', np.std(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant mad:', np.median(np.absolute(slasr_per_tenant - np.median(slasr_per_tenant))) * 100, '%')
        print('SLASR per tenant min:', np.min(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant max:', np.max(slasr_per_tenant) * 100, '%')
        if len(env.tenant_targets) > 1:
            target_actual = np.array(env.tenant_targets) - np.array(slasr_per_tenant)
            num_satisfied_slas = np.sum(target_actual <= 0)
            print(f'Happy tenants: {num_satisfied_slas} ({100 * num_satisfied_slas / env.num_tenants:.2f} %)')
            print('SLASR Diff:', target_actual.tolist())
            print('SLASR Mean:', mean(target_actual))

def mean(l):
    return sum(l) / len(l)
