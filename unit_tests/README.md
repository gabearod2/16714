# SPARK compatibility checks

Run these tests from the `16714_spark` repository root whenever the SPARK
checkout changes:

```bash
conda activate spark_course
python -m unittest discover -s unit_tests -v
python -m scripts.generate_results --all
python -m scripts.validate_results
python scripts/check_repository.py
```

The supported SPARK baseline is the current `master` branch. Install it with
the `mujoco` profile and `--dev`; the course does not require learned-policy,
Isaac, ROS, or hardware SDK extras.

The suite checks:

- public SPARK imports required by the lectures;
- robot-config-owned dynamics and course-owned numerical model stepping;
- LQR, MPC, and iLQR construction from the expected robot configurations;
- estimator, MRAC, and ILC migration boundaries;
- lecture package/configuration structure;
- deterministic result summaries against compact baselines.

A future SPARK commit is compatible if this suite and the result validation
both pass. Do not add course-specific behavior to SPARK to satisfy this suite.
