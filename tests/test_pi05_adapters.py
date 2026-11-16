import numpy as np
import pytest
from integration_pi05.actions import resample, ActionAdapter
from detent_bottle.core import DetentBottleSim


def test_resampling_preserves_displacement():
    chunk = np.zeros((15, 8)); chunk[:, :7] = np.arange(15)[:, None] / 15
    output = np.array(list(resample(chunk, 20)))
    np.testing.assert_allclose(output[:, :7].sum(0), chunk[:, :7].sum(0))
    with pytest.raises(ValueError): list(resample(np.zeros((10, 8)), 4))


def test_fk_does_not_write_physics():
    sim = DetentBottleSim()
    try:
        before = sim.data.qpos.copy()
        adapter = ActionAdapter(sim)
        adapter.apply(np.zeros(8))
        np.testing.assert_array_equal(sim.data.qpos, before)
        assert sim.grip_targets[0] == .045
        adapter.apply(np.r_[np.zeros(7), 1])
        assert sim.grip_targets[0] == 0
    finally: sim.close()


@pytest.mark.parametrize('ticks', [0, -1, 21, 1.5, True])
def test_resampling_rejects_invalid_horizon(ticks):
    with pytest.raises(ValueError): list(resample(np.zeros((15, 8)), ticks))


def test_fractional_tick_convention():
    chunk = np.zeros((15, 8)); chunk[:, 0] = 1
    cmds = np.array(list(resample(chunk, 4)))
    np.testing.assert_allclose(cmds[:, 0], .75)
    # Four 50ms ticks represent three normalized 1/15s increments.
    assert cmds[:, 0].sum() == pytest.approx(3)


def test_observation_closure_and_privacy():
    from integration_pi05.observations import request
    obs = {'agentview': np.zeros((224,224,3), dtype=np.uint8),
           'left_wrist': np.zeros((224,224,3), dtype=np.uint8),
           'joint_pos': np.arange(14), 'gripper_qpos': np.array([.0225,.0225,.045,.045]),
           'privileged': {'bottle_pos': [0,0,1]}}
    payload = request(obs, 'Open the bottle')
    assert payload['observation/gripper_position'][0] == pytest.approx(.5)
    assert len(payload) == 5
    np.testing.assert_array_equal(payload['observation/joint_position'], np.arange(7))


def test_adapter_diagnostics_match_fk_and_are_snapshots():
    import mujoco
    from detent_bottle.core import WORKSPACE
    sim = DetentBottleSim()
    try:
        adapter = ActionAdapter(sim)
        assert adapter.last_diagnostics is None
        before = sim.data.qpos.copy()
        command = np.r_[np.linspace(-.5, .5, 7), .4]
        expected_joints = before[sim.qids[0]] + .2 * command[:7] * .5
        scratch = mujoco.MjData(sim.model)
        scratch.qpos[:] = before
        scratch.qpos[sim.qids[0]] = np.clip(expected_joints, *sim.model.jnt_range[sim.ids[0]].T)
        mujoco.mj_forward(sim.model, scratch)
        result = adapter.apply(command, fraction=.5)
        diag = adapter.last_diagnostics
        np.testing.assert_array_equal(sim.data.qpos, before)
        np.testing.assert_array_equal(diag['pre_joint_pos'], before[sim.qids[0]])
        np.testing.assert_array_equal(diag['raw_joint_target'], expected_joints)
        np.testing.assert_array_equal(diag['fk_pos'], scratch.site_xpos[sim.tips[0]])
        np.testing.assert_array_equal(diag['fk_rotation'], scratch.site_xmat[sim.tips[0]].reshape(3, 3))
        np.testing.assert_array_equal(result, np.clip(diag['fk_pos'], *WORKSPACE))
        np.testing.assert_array_equal(result, sim.targets[0])
        assert diag['gripper_gap'] == pytest.approx(.054)
        assert not diag['gripper_clipped']
        saved_rotation = diag['fk_rotation'].copy()
        command[:] = 99
        result[:] = 99
        adapter.apply(np.zeros(8))
        assert diag['command'][7] == .4
        assert np.all(diag['bounded_target'] != 99)
        np.testing.assert_array_equal(diag['fk_rotation'], saved_rotation)
    finally:
        sim.close()


@pytest.mark.parametrize('gripper,closure', [(-.5, 0.), (1.5, 1.)])
def test_adapter_records_distinct_clipping_stages(monkeypatch, gripper, closure):
    from detent_bottle.core import WORKSPACE
    sim = DetentBottleSim()
    try:
        # Force a joint limit independently of command saturation.
        sim.data.qpos[sim.qids[0][0]] = sim.model.jnt_range[sim.ids[0][0], 1] - .01
        adapter = ActionAdapter(sim)
        command = np.r_[2., np.zeros(6), gripper]
        adapter.apply(command)
        fk_pos = adapter.last_diagnostics['fk_pos']
        narrowed = WORKSPACE.copy()
        if fk_pos[0] > WORKSPACE[0, 0] + .001:
            narrowed[1, 0] = WORKSPACE[0, 0] + .001
        else:
            narrowed[0, 0] = WORKSPACE[1, 0] - .001
        monkeypatch.setattr('integration_pi05.actions.WORKSPACE', narrowed)
        before = sim.data.qpos.copy()
        previous_clips = adapter.clips
        result = adapter.apply(command)
        diag = adapter.last_diagnostics
        assert diag['joint_action_clipped'].tolist() == [True] + [False] * 6
        assert diag['joint_limit_clipped'][0]
        assert diag['workspace_clipped'][0]
        assert adapter.clips == previous_clips + 1
        assert diag['gripper_raw'] == gripper
        assert diag['gripper_clipped']
        assert diag['gripper_closure'] == closure
        np.testing.assert_array_equal(result, np.clip(diag['fk_pos'], *narrowed))
        np.testing.assert_array_equal(sim.data.qpos, before)
    finally:
        sim.close()


def test_resampling_boundary_blends_joint_increments_but_holds_gripper():
    chunk = np.zeros((15, 8))
    chunk[0] = np.r_[np.ones(7), .2]
    chunk[1] = np.r_[-np.ones(7), .8]
    output = np.array(list(resample(chunk, 3)))
    np.testing.assert_allclose(output[:, 0], [.75, -.25, -.5])
    np.testing.assert_array_equal(output[:, 7], [.2, .2, .8])
