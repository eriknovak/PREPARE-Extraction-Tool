from app.training.gliner_trainer import GLiNERFinetuner


def test_compute_eval_steps_targets_about_15_to_20_points():
    f = GLiNERFinetuner(run_id=1, base_model_path="x", training_data=[{}])
    # 800 total steps -> ~16-20 eval points -> eval_steps in [40, 53]
    assert 40 <= f._compute_eval_steps(total_steps=800) <= 53
    # tiny runs still evaluate at least once
    assert f._compute_eval_steps(total_steps=3) >= 1


def test_total_steps_formula():
    f = GLiNERFinetuner(
        run_id=1,
        base_model_path="x",
        training_data=[{}],
        num_epochs=4,
        train_batch_size=8,
    )
    # 100 train items, batch 8 -> 13 steps/epoch * 4 = 52
    assert f._compute_total_steps(train_size=100) == 52
