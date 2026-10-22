import numpy as np
import pytest

from qtraffic import FixedTimePolicy, MaxQueuePolicy, SCENARIOS, TrafficSimulator, make_scenario


def test_grid_topology():
    net = make_scenario("off_peak", 2, 3)
    assert net.n == 6
    assert net.downstream_ew(0) == 1 and net.downstream_ew(2) is None
    assert net.downstream_ns(0) == 3 and net.downstream_ns(3) is None
    assert net.upstream_ew(1) == 0 and net.upstream_ns(4) == 1
    assert len(net.links()) == 2 * 2 + 3 * 1
    assert net.label(4) == "B2"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_build(name):
    net = make_scenario(name, 3, 3, seed=1)
    assert net.arrival_ns.shape == (9,) and (net.arrival_ns > 0).all()


def test_vehicle_conservation():
    net = make_scenario("rush_hour", 2, 2)
    sim = TrafficSimulator(net, seed=3)
    start = sim.initial_state()
    res = sim.run(MaxQueuePolicy(), n_slots=20, state=start)
    final_queue = res.queues[-1].sum()
    assert np.isclose(start.total_queue + res.arrivals, res.exited + final_queue)


def test_actuated_beats_fixed_time_under_imbalance():
    net = make_scenario("rush_hour", 2, 2)
    fixed = TrafficSimulator(net, seed=0).run(FixedTimePolicy(), n_slots=60)
    actuated = TrafficSimulator(net, seed=0).run(MaxQueuePolicy(), n_slots=60)
    assert actuated.avg_wait_s < fixed.avg_wait_s
