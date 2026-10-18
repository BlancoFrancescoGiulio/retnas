class Job:
    """Profiled layer execution costs for one DNN on the configured accelerators."""

    def __init__(self, name, min_latency, layer_latencies):
        self.name = name
        self.min_latency = min_latency
        self.layer_latencies = layer_latencies
        self.num_layers = len(layer_latencies)

class SubJob:

    def __init__(self, job_instance_id, job_id, layer_id, relative_deadline, absolute_deadline, arrival_time, dependencies=[], children=[], is_last=False, start_time=None, finish_time=None, sa_id=None, owner=0, target_slasr=1, is_hrt=False, hrt_customer_id=None) -> None:
        self.job_instance_id = job_instance_id
        self.job_id = job_id
        self.layer_id = layer_id
        self.relative_deadline = relative_deadline
        self.absolute_deadline = absolute_deadline
        self.dependencies = dependencies
        self.children = children
        self.is_last = is_last
        self.arrival_time = arrival_time
        self.start_time = start_time
        self.finish_time = finish_time
        self.sa_id = sa_id
        self.owner = owner
        self.target_slasr = target_slasr
        self.is_hrt = is_hrt
        self.hrt_customer_id = hrt_customer_id
