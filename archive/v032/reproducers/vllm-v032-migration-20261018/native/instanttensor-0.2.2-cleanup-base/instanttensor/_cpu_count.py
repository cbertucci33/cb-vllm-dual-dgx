import os


_CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_CPU_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_CPU_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def cpu_count():
    limits = [os.cpu_count() or 1]

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            limits.append(len(get_affinity(0)))
        except NotImplementedError:
            pass

    cgroup_limit = _cgroup_cpu_limit()
    if cgroup_limit is not None:
        limits.append(cgroup_limit)

    return max(min(limits), 1)


def _cgroup_cpu_limit():
    quota_period = _read_cgroup_quota_period()
    if quota_period is None:
        return None

    quota, period = quota_period
    if quota == "max":
        return None

    quota = int(quota)
    period = int(period)
    if quota <= 0 or period <= 0:
        return None
    return (quota + period - 1) // period


def _read_cgroup_quota_period():
    if os.path.exists(_CGROUP_V2_CPU_MAX):
        with open(_CGROUP_V2_CPU_MAX) as cpu_max_file:
            values = cpu_max_file.read().split()
        if len(values) == 2:
            return values

    if not (
        os.path.exists(_CGROUP_V1_CPU_QUOTA)
        and os.path.exists(_CGROUP_V1_CPU_PERIOD)
    ):
        return None

    with open(_CGROUP_V1_CPU_QUOTA) as quota_file:
        quota = quota_file.read().strip()
    with open(_CGROUP_V1_CPU_PERIOD) as period_file:
        period = period_file.read().strip()
    return quota, period
