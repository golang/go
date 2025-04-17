// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle

import (
	"bytes"
	"crypto/internal/fips140deps/byteorder"
	"math/rand/v2"
	"testing"
	"time"
)

func TestConstantTimeLessOrEqBytes(t *testing.T) {
	seed := make([]byte, 32)
	byteorder.BEPutUint64(seed, uint64(time.Now().UnixNano()))
	r := rand.NewChaCha8([32]byte(seed))
	for l := range 20 {
		a := make([]byte, l)
		b := make([]byte, l)
		empty := make([]byte, l)
		r.Read(a)
		r.Read(b)
		exp := 0
		if bytes.Compare(a, b) <= 0 {
			exp = 1
		}
		if got := ConstantTimeLessOrEqBytes(a, b); got != exp {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want %d", a, b, got, exp)
		}
		exp = 0
		if bytes.Compare(b, a) <= 0 {
			exp = 1
		}
		if got := ConstantTimeLessOrEqBytes(b, a); got != exp {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want %d", b, a, got, exp)
		}
		if got := ConstantTimeLessOrEqBytes(empty, a); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", empty, a, got)
		}
		if got := ConstantTimeLessOrEqBytes(empty, b); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", empty, b, got)
		}
		if got := ConstantTimeLessOrEqBytes(a, a); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", a, a, got)
		}
		if got := ConstantTimeLessOrEqBytes(b, b); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", b, b, got)
		}
		if got := ConstantTimeLessOrEqBytes(empty, empty); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", empty, empty, got)
		}
		if l == 0 {
			continue
		}
		max := make([]byte, l)
		for i := range max {
			max[i] = 0xff
		}
		if got := ConstantTimeLessOrEqBytes(a, max); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", a, max, got)
		}
		if got := ConstantTimeLessOrEqBytes(b, max); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", b, max, got)
		}
		if got := ConstantTimeLessOrEqBytes(empty, max); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", empty, max, got)
		}
		if got := ConstantTimeLessOrEqBytes(max, max); got != 1 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", max, max, got)
		}
		aPlusOne := make([]byte, l)
		copy(aPlusOne, a)
		for i := l - 1; i >= 0; i-- {
			if aPlusOne[i] == 0xff {
				aPlusOne[i] = 0
				continue
			}
			aPlusOne[i]++
			if got := ConstantTimeLessOrEqBytes(a, aPlusOne); got != 1 {
				t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 1", a, aPlusOne, got)
			}
			if got := ConstantTimeLessOrEqBytes(aPlusOne, a); got != 0 {
				t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 0", aPlusOne, a, got)
			}
			break
		}
		shorter := make([]byte, l-1)
		copy(shorter, a)
		if got := ConstantTimeLessOrEqBytes(a, shorter); got != 0 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 0", a, shorter, got)
		}
		if got := ConstantTimeLessOrEqBytes(shorter, a); got != 0 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 0", shorter, a, got)
		}
		if got := ConstantTimeLessOrEqBytes(b, shorter); got != 0 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 0", b, shorter, got)
		}
		if got := ConstantTimeLessOrEqBytes(shorter, b); got != 0 {
			t.Errorf("ConstantTimeLessOrEqBytes(%x, %x) = %d, want 0", shorter, b, got)
		}
	}
}
