import unittest

import torch

from model import LSTMAttentionVancomycin


def _build_mock_batch(batch_size=2, seq_len=6, vocab_size=50, meds_per_visit=5):
    cont = torch.randn(batch_size, seq_len, 40)
    cat = torch.randint(0, vocab_size, (batch_size, seq_len, meds_per_visit))
    labels = torch.zeros(batch_size, seq_len)
    doses = torch.rand(batch_size, seq_len)
    tdiff = torch.rand(batch_size, seq_len)
    v_tensor = torch.rand(batch_size, seq_len)
    vanco_el = torch.rand(batch_size, seq_len)
    pt_ids = torch.arange(batch_size, dtype=torch.float32)
    lengths = torch.tensor([seq_len, seq_len - 2], dtype=torch.float32)
    labels[0, : seq_len // 2] = torch.rand(seq_len // 2)
    labels[1, : (seq_len - 2) // 2] = torch.rand((seq_len - 2) // 2)
    cat = cat.to(torch.long)
    batch = (
        cont,
        cat,
        labels,
        doses,
        tdiff,
        v_tensor,
        vanco_el,
        pt_ids,
        lengths,
    )
    return batch


class LSTMAttentionVancomycinTest(unittest.TestCase):
    def test_forward_runs_and_shapes(self):
        model = LSTMAttentionVancomycin(device="cpu")
        batch = _build_mock_batch()
        predictions, loss = model(batch)
        self.assertEqual(predictions.shape, batch[2].shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.last_attention)
        heads = model.last_attention.shape[1]
        self.assertGreater(heads, 0)


if __name__ == "__main__":
    unittest.main()
