// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math/big"
	"testing"
)

func TestMagicExhaustive8(t *testing.T) {
	testMagicExhaustive(t, 8)
}
func TestMagicExhaustive8U(t *testing.T) {
	testMagicExhaustiveU(t, 8)
}
func TestMagicExhaustive16(t *testing.T) {
	if testing.Short() {
		t.Skip("slow test; skipping")
	}
	testMagicExhaustive(t, 16)
}
func TestMagicExhaustive16U(t *testing.T) {
	if testing.Short() {
		t.Skip("slow test; skipping")
	}
	testMagicExhaustiveU(t, 16)
}

// exhaustive test of magic for n bits
func testMagicExhaustive(t *testing.T, n uint) {
	min := -int64(1) << (n - 1)
	max := int64(1) << (n - 1)
	for c := int64(1); c < max; c++ {
		if !smagicOK(n, int64(c)) {
			continue
		}
		m := int64(smagic(n, c).m)
		s := smagic(n, c).s
		for i := min; i < max; i++ {
			want := i / c
			got := (i * m) >> (n + uint(s))
			if i < 0 {
				got++
			}
			if want != got {
				t.Errorf("signed magic wrong for %d / %d: got %d, want %d (m=%d,s=%d)\n", i, c, got, want, m, s)
			}
		}
	}
}
func testMagicExhaustiveU(t *testing.T, n uint) {
	max := uint64(1) << n
	for c := uint64(1); c < max; c++ {
		if !umagicOK(n, int64(c)) {
			continue
		}
		m := umagic(n, int64(c)).m
		s := umagic(n, int64(c)).s
		for i := uint64(0); i < max; i++ {
			want := i / c
			got := (i * (max + m)) >> (n + uint(s))
			if want != got {
				t.Errorf("unsigned magic wrong for %d / %d: got %d, want %d (m=%d,s=%d)\n", i, c, got, want, m, s)
			}
		}
	}
}

func TestMagicUnsigned(t *testing.T) {
	One := new(big.Int).SetUint64(1)
	for _, n := range [...]uint{8, 16, 32, 64} {
		TwoN := new(big.Int).Lsh(One, n)
		Max := new(big.Int).Sub(TwoN, One)
		for _, c := range [...]uint64{
			3,
			5,
			6,
			7,
			9,
			10,
			11,
			12,
			13,
			14,
			15,
			17,
			1<<8 - 1,
			1<<8 + 1,
			1<<16 - 1,
			1<<16 + 1,
			1<<32 - 1,
			1<<32 + 1,
			1<<64 - 1,
		} {
			if c>>n != 0 {
				continue // not appropriate for the given n.
			}
			if !umagicOK(n, int64(c)) {
				t.Errorf("expected n=%d c=%d to pass\n", n, c)
			}
			m := umagic(n, int64(c)).m
			s := umagic(n, int64(c)).s

			C := new(big.Int).SetUint64(c)
			M := new(big.Int).SetUint64(m)
			M.Add(M, TwoN)

			// Find largest multiple of c.
			Mul := new(big.Int).Div(Max, C)
			Mul.Mul(Mul, C)
			mul := Mul.Uint64()

			// Try some input values, mostly around multiples of c.
			for _, x := range [...]uint64{0, 1,
				c - 1, c, c + 1,
				2*c - 1, 2 * c, 2*c + 1,
				mul - 1, mul, mul + 1,
				uint64(1)<<n - 1,
			} {
				X := new(big.Int).SetUint64(x)
				if X.Cmp(Max) > 0 {
					continue
				}
				Want := new(big.Int).Quo(X, C)
				Got := new(big.Int).Mul(X, M)
				Got.Rsh(Got, n+uint(s))
				if Want.Cmp(Got) != 0 {
					t.Errorf("umagic for %d/%d n=%d doesn't work, got=%s, want %s\n", x, c, n, Got, Want)
				}
			}
		}
	}
}

func TestMagicSigned(t *testing.T) {
	One := new(big.Int).SetInt64(1)
	for _, n := range [...]uint{8, 16, 32, 64} {
		TwoNMinusOne := new(big.Int).Lsh(One, n-1)
		Max := new(big.Int).Sub(TwoNMinusOne, One)
		Min := new(big.Int).Neg(TwoNMinusOne)
		for _, c := range [...]int64{
			3,
			5,
			6,
			7,
			9,
			10,
			11,
			12,
			13,
			14,
			15,
			17,
			1<<7 - 1,
			1<<7 + 1,
			1<<15 - 1,
			1<<15 + 1,
			1<<31 - 1,
			1<<31 + 1,
			1<<63 - 1,
		} {
			if c>>(n-1) != 0 {
				continue // not appropriate for the given n.
			}
			if !smagicOK(n, int64(c)) {
				t.Errorf("expected n=%d c=%d to pass\n", n, c)
			}
			m := smagic(n, int64(c)).m
			s := smagic(n, int64(c)).s

			C := new(big.Int).SetInt64(c)
			M := new(big.Int).SetUint64(m)

			// Find largest multiple of c.
			Mul := new(big.Int).Div(Max, C)
			Mul.Mul(Mul, C)
			mul := Mul.Int64()

			// Try some input values, mostly around multiples of c.
			for _, x := range [...]int64{
				-1, 1,
				-c - 1, -c, -c + 1, c - 1, c, c + 1,
				-2*c - 1, -2 * c, -2*c + 1, 2*c - 1, 2 * c, 2*c + 1,
				-mul - 1, -mul, -mul + 1, mul - 1, mul, mul + 1,
				int64(1)<<n - 1, -int64(1)<<n + 1,
			} {
				X := new(big.Int).SetInt64(x)
				if X.Cmp(Min) < 0 || X.Cmp(Max) > 0 {
					continue
				}
				Want := new(big.Int).Quo(X, C)
				Got := new(big.Int).Mul(X, M)
				Got.Rsh(Got, n+uint(s))
				if x < 0 {
					Got.Add(Got, One)
				}
				if Want.Cmp(Got) != 0 {
					t.Errorf("smagic for %d/%d n=%d doesn't work, got=%s, want %s\n", x, c, n, Got, Want)
				}
			}
		}
	}
}
