import numpy as np
from tqdm.auto import trange
from tqdm import tqdm
from retnas.environment import *

def prema(config):
    env = Environment(config, seed=config.seed, use_sa_assignement_heuristic=True)
    env.prepend_sa_remaining_time = False
    env.reset()
    for i in trange(config.num_steps):
        action = []
        for sj in env.rq:
            priority1 = -(env.current_time - sj.arrival_time) / config.all_jobs[sj.job_id].min_latency
            priority2 = sum(config.all_jobs[sj.job_id].layer_latencies[sj.layer_id]) / config.num_sas
            sa_choice_feat = np.random.rand(config.num_sas - 1)
            action.append(np.concatenate(([priority1, priority2], sa_choice_feat)))
        action = np.array(action)
        env.prema_step(action)
        total_deadline_hits = env.deadline_hits_be + env.deadline_hits_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        slast = total_deadline_hits / total_arrived_jobs if total_arrived_jobs > 0 else 0
        slast_be = env.deadline_hits_be / env.total_arrived_jobs_be if env.total_arrived_jobs_be > 0 else 0
        slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 0
        total_deadline_miss = env.deadline_misses_be + env.deadline_misses_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        percentage_wcet_relies = env.hrt_wcet_win / (env.hrt_wcet_win + env.hrt_rl_win) if env.hrt_wcet_win + env.hrt_rl_win > 0 else 0
        if env.deadline_misses_hrt:
            tqdm.write(f'[WARN] HRT deadline misses so far: {env.deadline_misses_hrt} (PREMA cannot guarantee HRT deadlines)')
        tqdm.write('Step: \t{} \t Slast: \t{:0.2f}% \t Slast_BE: \t{:0.2f}% \t Slast_HRT: \t{:0.2f}% \t HRT/BE MISS: {},{} \t WCET won {:0.2f}% times'.format(i + 1, slast * 100, slast_be * 100, slast_hrt * 100, env.deadline_misses_hrt, env.deadline_misses_be, percentage_wcet_relies * 100))
    print('SLA Satisfaction Rate:', slast)
    print('Throughput:', env.executed_sjs * 1000000000.0 / env.current_time)
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
            print('SLASR Mean:', np.mean(target_actual))
