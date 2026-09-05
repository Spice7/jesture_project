"""합성 텐서로 LSTM 계약을 검증합니다. 실제 데이터 학습이나 성능 평가가 아닙니다."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch
from torch import nn

if __package__:
    from . import models
else:
    import models


class LSTMClassifierTests(unittest.TestCase):
    def setUp(self):
        # 재현성 설정은 모델이 아닌 테스트에 둡니다. 종료 시 CPU RNG 상태를 복원합니다.
        state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, state)
        torch.manual_seed(42)

    def test_default_shape_and_finite_output(self):
        model = models.LSTMClassifier()
        x = torch.randn(4, 32, 66)
        original = x.clone()
        logits = model(x)
        self.assertEqual(logits.shape, (4, 3))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.device.type, "cpu")
        self.assertTrue(torch.isfinite(logits).all().item())
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_single_batch_variable_lengths_and_layers(self):
        for layers in (1, 2):
            model = models.LSTMClassifier(num_layers=layers)
            for length in (1, 7, 32, 45):
                with self.subTest(layers=layers, length=length):
                    out = model(torch.randn(1, length, 66))
                    self.assertEqual(out.shape, (1, 3))
                    self.assertTrue(torch.isfinite(out).all().item())

    def test_constructor_overrides(self):
        model = models.LSTMClassifier(input_size=10, hidden_size=8,
                                      num_layers=2, num_classes=4, dropout=0)
        self.assertEqual(model(torch.randn(2, 5, 10)).shape, (2, 4))
        self.assertTrue(model.lstm.batch_first)
        self.assertFalse(model.lstm.bidirectional)

    def test_dropout_placement(self):
        for layers in (1, 2):
            model = models.LSTMClassifier(num_layers=layers, dropout=.3)
            self.assertEqual(model.lstm.dropout, .3 if layers > 1 else 0.0)
            self.assertEqual(model.dropout.p, .3)
            model.train()
            self.assertTrue(model.dropout.training)
            model.eval()
            self.assertFalse(model.dropout.training)

    def test_last_layer_hidden_and_raw_logits(self):
        model = models.LSTMClassifier(hidden_size=4, num_layers=2, dropout=0).eval()
        hidden = torch.stack((torch.full((2, 4), 9.0), torch.full((2, 4), 2.0)))
        with torch.no_grad():
            model.classifier.weight.fill_(1)
            model.classifier.bias.fill_(-10)
        # 모의 출력으로 최상위 층 h_n[-1]의 선택과 softmax 생략을 직접 확인합니다.
        with mock.patch.object(model.lstm, "forward", return_value=(
            torch.zeros(2, 5, 4), (hidden, torch.zeros_like(hidden))
        )):
            logits = model(torch.zeros(2, 5, 66))
        torch.testing.assert_close(logits, torch.full((2, 3), -2.0), rtol=0, atol=0)

    def test_invalid_inputs(self):
        model = models.LSTMClassifier()
        for value in ([1, 2], torch.ones(2, 3, 66, dtype=torch.int64),
                      torch.ones(2, 3, 66, dtype=torch.bool),
                      torch.ones(2, 3, 66, dtype=torch.complex64)):
            with self.subTest(type=str(type(value))):
                with self.assertRaises(TypeError):
                    model(value)
        for shape in ((32, 66), (1, 2, 3, 66), (2, 32, 65), (0, 32, 66), (2, 0, 66)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    model(torch.empty(shape))

    def test_invalid_constructor_settings(self):
        for name in ("input_size", "hidden_size", "num_layers", "num_classes"):
            for value in (0, -1, True, 1.5, "2", None):
                with self.subTest(name=name, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        models.LSTMClassifier(**{name: value})
        for value in (-.1, 1, 1.1, float("nan"), float("inf"), True, "0.2", None):
            with self.subTest(dropout=value):
                with self.assertRaises((TypeError, ValueError)):
                    models.LSTMClassifier(dropout=value)

    def test_loss_backward_finite_gradients(self):
        for layers in (1, 2):
            model = models.LSTMClassifier(num_layers=layers)
            logits = model(torch.randn(4, 32, 66))
            labels = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
            loss = nn.CrossEntropyLoss()(logits, labels)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()  # 기울기 계산만 검사합니다. 가중치 갱신이나 실제 데이터 학습은 하지 않습니다.
            for name, parameter in model.named_parameters():
                with self.subTest(layers=layers, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_eval_reproducibility_and_no_cross_batch_state(self):
        model = models.LSTMClassifier(num_layers=2).eval()
        x = torch.randn(2, 32, 66)
        with torch.no_grad():
            first = model(x)
            second = model(x)
            model(torch.randn(5, 9, 66))
            after_other_batch = model(x)
            alone = model(x[:1])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, after_other_batch, rtol=0, atol=0)
        torch.testing.assert_close(first[:1], alone)

    def test_stationary_input_is_valid(self):
        model = models.LSTMClassifier().eval()
        # 시간에 따라 동일한 특징을 반복하는 정지 시퀀스도 거부하지 않습니다.
        x = torch.randn(2, 1, 66).repeat(1, 32, 1)
        out = model(x)
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_state_dict_roundtrip(self):
        config = dict(input_size=66, hidden_size=12, num_layers=2, num_classes=3, dropout=.3)
        model = models.LSTMClassifier(**config).eval()
        x = torch.randn(3, 13, 66)
        with torch.no_grad():
            expected = model(x)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model_config": config, "state_dict": model.state_dict()}, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            restored = models.LSTMClassifier(**checkpoint["model_config"])
            restored.load_state_dict(checkpoint["state_dict"])
            restored.eval()
            with torch.no_grad():
                actual = restored(x)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_caller_controls_dtype(self):
        model = models.LSTMClassifier().double()
        x = torch.randn(2, 4, 66, dtype=torch.float64)
        self.assertEqual(model(x).dtype, torch.float64)
        with self.assertRaises((ValueError, RuntimeError)):
            model(x.float())  # 모델이 몰래 float64로 바꿔 주지 않아야 합니다.

    def test_import_and_construction_do_not_reseed(self):
        state = torch.get_rng_state().clone()
        with mock.patch.object(torch, "manual_seed", side_effect=AssertionError("unexpected seed reset")):
            importlib.reload(models)
            torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
            models.LSTMClassifier()
        # 생성 시 파라미터 초기화가 RNG를 소비하는 것은 정상입니다. seed 재설정만 금지합니다.


if __name__ == "__main__":
    unittest.main()
