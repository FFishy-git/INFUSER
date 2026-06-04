"""Unit tests for the seeding fixture.

Does not exercise GPU / vLLM — those require a real accelerator. Instead
we verify:
  * ``seed_all`` is idempotent and actually seeds the RNGs it promises to.
  * ``derive_sampling_seed`` is deterministic, non-negative, fits in int64,
    and different inputs produce different outputs.
  * CPU torch RNG is reproducible after ``seed_all``.
"""

from __future__ import annotations

import os
import random
import unittest

import numpy as np


class SeedAllTest(unittest.TestCase):
    def test_seed_all_is_reproducible_cpu(self):
        from verl_inf_evolve.utils.seeding import seed_all

        seed_all(1234)
        a_random = [random.random() for _ in range(5)]
        a_numpy = np.random.rand(5).tolist()

        seed_all(1234)
        b_random = [random.random() for _ in range(5)]
        b_numpy = np.random.rand(5).tolist()

        self.assertEqual(a_random, b_random)
        self.assertEqual(a_numpy, b_numpy)

    def test_seed_all_different_seeds_diverge(self):
        from verl_inf_evolve.utils.seeding import seed_all

        seed_all(1)
        a = np.random.rand(5).tolist()
        seed_all(2)
        b = np.random.rand(5).tolist()

        self.assertNotEqual(a, b)

    def test_seed_all_sets_expected_env(self):
        from verl_inf_evolve.utils.seeding import seed_all

        seed_all(7)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "7")
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")

    def test_seed_all_rank_offset(self):
        from verl_inf_evolve.utils.seeding import seed_all

        eff_a = seed_all(100, rank=0)
        eff_b = seed_all(100, rank=3)
        self.assertEqual(eff_a, 100)
        self.assertEqual(eff_b, 103)

    def test_torch_cpu_rng_reproducible(self):
        import torch

        from verl_inf_evolve.utils.seeding import seed_all

        seed_all(42)
        a = torch.rand(4).tolist()
        seed_all(42)
        b = torch.rand(4).tolist()
        self.assertEqual(a, b)


class DeriveSamplingSeedTest(unittest.TestCase):
    def test_deterministic(self):
        from verl_inf_evolve.utils.seeding import derive_sampling_seed

        self.assertEqual(
            derive_sampling_seed(42, 1, 2, 3),
            derive_sampling_seed(42, 1, 2, 3),
        )

    def test_nonneg_and_fits_int64(self):
        from verl_inf_evolve.utils.seeding import derive_sampling_seed

        for args in [(0,), (1, 2), (123, 456, 789, 0)]:
            s = derive_sampling_seed(42, *args)
            self.assertGreaterEqual(s, 0)
            self.assertLess(s, 2**63)

    def test_different_inputs_differ(self):
        from verl_inf_evolve.utils.seeding import derive_sampling_seed

        a = derive_sampling_seed(42, 0, 0, 0)
        b = derive_sampling_seed(42, 0, 0, 1)
        c = derive_sampling_seed(43, 0, 0, 0)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(b, c)

    def test_base_seed_affects_all_derivations(self):
        """Different base seeds must produce different downstream seeds."""
        from verl_inf_evolve.utils.seeding import derive_sampling_seed

        seeds_a = {derive_sampling_seed(42, i) for i in range(32)}
        seeds_b = {derive_sampling_seed(123, i) for i in range(32)}
        self.assertEqual(len(seeds_a & seeds_b), 0)


if __name__ == "__main__":
    unittest.main()
