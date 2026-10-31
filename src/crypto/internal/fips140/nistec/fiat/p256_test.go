// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64

package fiat

import (
	"testing"
)

func TestP256Mul(t *testing.T) {
	t.Run("CompareAssemblyWithGo", func(t *testing.T) {
		testCases := []struct {
			name string
			a, b p256MontgomeryDomainFieldElement
		}{
			// Basic cases
			{
				name: "one_times_one",
				a:    p256MontgomeryDomainFieldElement{1, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{1, 0, 0, 0},
			},
			{
				name: "max_low_times_two",
				a:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{2, 0, 0, 0},
			},
			{
				name: "random_large_values",
				a:    p256MontgomeryDomainFieldElement{0x123456789abcdef0, 0x0fedcba987654321, 0x1111111111111111, 0x2222222222222222},
				b:    p256MontgomeryDomainFieldElement{0xaaaaaaaaaaaaaaaa, 0xbbbbbbbbbbbbbbbb, 0xcccccccccccccccc, 0xdddddddddddddddd},
			},
			// Zero cases
			{
				name: "zero_times_zero",
				a:    p256MontgomeryDomainFieldElement{0, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{0, 0, 0, 0},
			},
			{
				name: "zero_times_one",
				a:    p256MontgomeryDomainFieldElement{0, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{1, 0, 0, 0},
			},
			{
				name: "one_times_zero",
				a:    p256MontgomeryDomainFieldElement{1, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{0, 0, 0, 0},
			},
			{
				name: "zero_times_max",
				a:    p256MontgomeryDomainFieldElement{0, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff},
			},
			// Unit element cases
			{
				name: "one_times_max",
				a:    p256MontgomeryDomainFieldElement{1, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff},
			},
			{
				name: "max_times_one",
				a:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff},
				b:    p256MontgomeryDomainFieldElement{1, 0, 0, 0},
			},
			// Maximum values
			{
				name: "max_times_max",
				a:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff},
				b:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff},
			},
			// Boundary values - P256 prime modulus components
			// P256 = 2^256 - 2^224 + 2^192 + 2^96 - 1
			// In Montgomery domain representation
			{
				name: "p256_prime_like_values",
				a:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0x00000000ffffffff, 0x0000000000000000, 0xffffffff00000001},
				b:    p256MontgomeryDomainFieldElement{0x0000000000000001, 0xffffffff00000000, 0xffffffffffffffff, 0x00000000fffffffe},
			},
			// Special bit patterns
			{
				name: "alternating_bits",
				a:    p256MontgomeryDomainFieldElement{0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x5555555555555555, 0xaaaaaaaaaaaaaaaa},
				b:    p256MontgomeryDomainFieldElement{0xaaaaaaaaaaaaaaaa, 0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x5555555555555555},
			},
			{
				name: "all_ones_pattern",
				a:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff},
				b:    p256MontgomeryDomainFieldElement{0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x3333333333333333, 0xcccccccccccccccc},
			},
			// High bit set cases
			{
				name: "high_bit_set",
				a:    p256MontgomeryDomainFieldElement{0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000},
				b:    p256MontgomeryDomainFieldElement{0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000},
			},
			// Low limb only
			{
				name: "low_limb_only",
				a:    p256MontgomeryDomainFieldElement{0x1234567890abcdef, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{0xfedcba0987654321, 0, 0, 0},
			},
			// High limb only
			{
				name: "high_limb_only",
				a:    p256MontgomeryDomainFieldElement{0, 0, 0, 0x1234567890abcdef},
				b:    p256MontgomeryDomainFieldElement{0, 0, 0, 0xfedcba0987654321},
			},
			// Middle limbs
			{
				name: "middle_limbs",
				a:    p256MontgomeryDomainFieldElement{0, 0x1111111111111111, 0x2222222222222222, 0},
				b:    p256MontgomeryDomainFieldElement{0, 0x3333333333333333, 0x4444444444444444, 0},
			},
			// Small values
			{
				name: "small_values",
				a:    p256MontgomeryDomainFieldElement{2, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{3, 0, 0, 0},
			},
			{
				name: "small_times_large",
				a:    p256MontgomeryDomainFieldElement{2, 0, 0, 0},
				b:    p256MontgomeryDomainFieldElement{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff},
			},
			// More random-like values
			{
				name: "random_pattern_1",
				a:    p256MontgomeryDomainFieldElement{0xdeadbeefcafebabe, 0x1234567890abcdef, 0xfedcba0987654321, 0xabcdef0123456789},
				b:    p256MontgomeryDomainFieldElement{0x1122334455667788, 0x99aabbccddeeff00, 0x0011223344556677, 0x8899aabbccddeeff},
			},
			{
				name: "random_pattern_2",
				a:    p256MontgomeryDomainFieldElement{0x0123456789abcdef, 0xfedcba9876543210, 0x13579bdf2468ace0, 0xfdb97531eca86420},
				b:    p256MontgomeryDomainFieldElement{0xf0e1d2c3b4a59687, 0x78695a4b3c2d1e0f, 0x0f1e2d3c4b5a6978, 0x8796a5b4c3d2e1f0},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				var outGo, outAsm p256MontgomeryDomainFieldElement
				p256MulGeneric(&outGo, &tc.a, &tc.b)
				p256Mul(&outAsm, &tc.a, &tc.b)

				if outGo != outAsm {
					t.Errorf("Mismatch:\n"+
						"  Arg1: %016x %016x %016x %016x\n"+
						"  Arg2: %016x %016x %016x %016x\n"+
						"  Go:   %016x %016x %016x %016x\n"+
						"  Asm:  %016x %016x %016x %016x\n",
						tc.a[0], tc.a[1], tc.a[2], tc.a[3],
						tc.b[0], tc.b[1], tc.b[2], tc.b[3],
						outGo[0], outGo[1], outGo[2], outGo[3],
						outAsm[0], outAsm[1], outAsm[2], outAsm[3])
				}
			})
		}
	})
}
