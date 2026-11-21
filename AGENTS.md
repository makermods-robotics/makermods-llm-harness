# Working on makermods-llm-harness

- Layout: `src/makermods_llm_harness/`: `actions.py` (actions, observations, tool schemas) and `backends/base.py`
  (the `Backend` protocol) are the contracts; `harness.py` runs actions through `safety.py` (envelope, table
  calibration) and `arm_model.py` (SDK kinematics, planning); `backends/` has `sim.py`, `mujoco.py`, `real.py`;
  `session.py`, `cli.py`, `mcp_server.py` and `agent.py` are the entry points; `robot.py` and `log.py` handle robot
  files and episode logs. `docs/`, `examples/`, `tests/`.
- Tests: `pip install -e ".[dev,mujoco,mcp,agent]"`, then `python -m pytest -q` (one file: `python -m pytest -q
  tests/test_harness.py`). Without an extra its tests skip.
- Tests run on simulation only: the simulators, injected fakes and the SDKs' `testing.simulated_arm`. A test never
  opens a CAN adapter, serial port or camera, never loads the real SDK rig file, and keeps `LLM_HARNESS_HOME` in a
  temporary directory (`tests/conftest.py` does this for every test). Code must run on Linux, macOS and Windows.
- Use only the SDKs' public API. Optional dependencies (`mujoco`, `mcp`, `anthropic`) are imported inside the module
  that needs them.
- Writing: comments, docstrings and docs describe what the code does, once, briefly, in plain words. No history or
  comparisons with earlier versions, no banner or stacked comments, no narration of obvious code. Keep docs and the
  numbers in them consistent with the code (`SafetyConfig` holds the envelope defaults), and only show commands that
  exist exactly as written.
