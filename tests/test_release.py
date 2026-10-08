"""Offline checks; no datasets or pretrained weights are required."""

import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

from configuration import load_configuration

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_dataset_dispatch(self):
        import trainers.run as run

        for dataset in ("mosi", "mosei", "sims"):
            with self.subTest(dataset=dataset):
                params = load_configuration(ROOT / "configs" / f"{dataset}.json")
                params.update(seed=1, model_save_path="unused")
                with patch.object(run, "EnRun") as english, patch.object(
                    run, "ChRun"
                ) as chinese:
                    run.main(Namespace(**params))
                    self.assertEqual(
                        (chinese if dataset == "sims" else english).call_count, 1
                    )

    def test_unknown_option_rejected(self):
        defaults = (ROOT / "configs/defaults.json").read_text(encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=[defaults, '{"typo": true}']):
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                load_configuration("unused.json")


class FusionTests(unittest.TestCase):
    def test_forward_backward_with_empty_masks(self):
        import torch
        from models.fusion import TriModalFusionEncoder

        torch.set_num_threads(2)
        for empty in (False, True):
            with self.subTest(empty=empty):
                torch.manual_seed(4)
                model = TriModalFusionEncoder(
                    hidden_size=12,
                    num_layers=2,
                    num_heads=3,
                    dropout=0,
                    ta_to_a_only=True,
                    tv_cross=True,
                )
                streams = [torch.randn(2, 3, 12, requires_grad=True) for _ in range(3)]
                mask = torch.zeros(2, 3) if empty else torch.ones(2, 3)
                output = model(streams[0], mask, streams[1], mask, streams[2], mask)
                self.assertEqual(output.shape, (2, 48))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(all(torch.isfinite(x.grad).all() for x in streams))


class MissingOutputTests(unittest.TestCase):
    def test_post_training_outputs_are_per_run_records(self):
        from contextlib import ExitStack
        import train_missing as missing

        args = Namespace(
            dataset="mosi",
            model_seed=1,
            selection_split="valid",
            selection_rate=0.5,
            selection_mask_seed=3,
            mask_seed=4,
            train_missing_seed=2,
            train_clean_probability=0.5,
            audio_block_size=320,
            epochs=1,
            early_stop=8,
            weight_decay=0.01,
            init_checkpoint=None,
            save_dir="runs/offline-output-check",
            skip_post_eval=False,
            keep_sims_train_test_duplicate=False,
            rates=[0.0, 0.5],
        )
        config = Namespace(train_mode="regression", dataset_name="mosi")
        tokenizer = Namespace(all_special_ids=[], unk_token_id=0)
        metric_keys = ["MAE", "Corr", "Has0_acc_2"]
        result = {"MAE": 0.5, "Corr": 0.5, "Has0_acc_2": 0.5, "Loss": 0.5}
        with ExitStack() as stack:
            for name in (
                "build_model",
                "freeze_audio_frontend",
                "EnTrainer",
                "build_training_optimizer",
                "save_training_checkpoint",
                "LNLNTrainMissingLoader",
                "MetricsTop",
                "print_result",
            ):
                stack.enter_context(patch.object(missing, name))
            stack.enter_context(
                patch.object(missing, "train_one_epoch", return_value=0.5)
            )
            stack.enter_context(
                patch.object(
                    missing,
                    "evaluate_setting",
                    side_effect=lambda **kwargs: dict(result),
                )
            )
            stack.enter_context(
                patch.object(
                    missing,
                    "selection_spec",
                    return_value={"Has0_acc_2": "max", "MAE": "min"},
                )
            )
            stack.enter_context(
                patch.object(missing, "load_checkpoint", return_value={"epoch": 1})
            )
            stack.enter_context(patch.object(Path, "mkdir"))
            saved_metadata = stack.enter_context(patch.object(Path, "write_text"))
            stack.enter_context(patch("builtins.print"))
            output = stack.enter_context(patch.object(missing, "write_outputs"))
            missing.run_training(
                args, config, [], [], [], tokenizer, "cpu", metric_keys
            )

        output.assert_called_once()
        rows, csv_path, json_path, actual_metrics = output.call_args.args
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["selection_key"] for row in rows}, {"Has0_acc_2", "MAE"})
        self.assertEqual(csv_path.name, "missing_eval_seed1.csv")
        self.assertEqual(json_path.name, "missing_eval_seed1.json")
        self.assertEqual(actual_metrics, metric_keys)
        self.assertEqual(saved_metadata.call_count, 2)


if __name__ == "__main__":
    unittest.main()
