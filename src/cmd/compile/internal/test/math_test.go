// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"testing"
)

var Output int

func BenchmarkDiv64UnsignedSmall(b *testing.B) {
	q := uint64(1)
	for i := 1; i <= b.N; i++ {
		q = (q + uint64(i)) / uint64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64Small(b *testing.B) {
	q := int64(1)
	for i := 1; i <= b.N; i++ {
		q = (q + int64(i)) / int64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64SmallNegDivisor(b *testing.B) {
	q := int64(-1)
	for i := 1; i <= b.N; i++ {
		q = (int64(i) - q) / -int64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64SmallNegDividend(b *testing.B) {
	q := int64(-1)
	for i := 1; i <= b.N; i++ {
		q = -(int64(i) - q) / int64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64SmallNegBoth(b *testing.B) {
	q := int64(1)
	for i := 1; i <= b.N; i++ {
		q = -(int64(i) + q) / -int64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64Unsigned(b *testing.B) {
	q := uint64(1)
	for i := 1; i <= b.N; i++ {
		q = (uint64(0x7fffffffffffffff) - uint64(i) - (q & 1)) / uint64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64(b *testing.B) {
	q := int64(1)
	for i := 1; i <= b.N; i++ {
		q = (int64(0x7fffffffffffffff) - int64(i) - (q & 1)) / int64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64NegDivisor(b *testing.B) {
	q := int64(-1)
	for i := 1; i <= b.N; i++ {
		q = (int64(0x7fffffffffffffff) - int64(i) - (q & 1)) / -int64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64NegDividend(b *testing.B) {
	q := int64(-1)
	for i := 1; i <= b.N; i++ {
		q = -(int64(0x7fffffffffffffff) - int64(i) - (q & 1)) / int64(i)
	}
	Output = int(q)
}

func BenchmarkDiv64NegBoth(b *testing.B) {
	q := int64(-1)
	for i := 1; i <= b.N; i++ {
		q = -(int64(0x7fffffffffffffff) - int64(i) - (q & 1)) / -int64(i)
	}
	Output = int(q)
}

func BenchmarkMod64UnsignedSmall(b *testing.B) {
	r := uint64(1)
	for i := 1; i <= b.N; i++ {
		r = (uint64(i) + r) % uint64(i)
	}
	Output = int(r)
}

func BenchmarkMod64Small(b *testing.B) {
	r := int64(1)
	for i := 1; i <= b.N; i++ {
		r = (int64(i) + r) % int64(i)
	}
	Output = int(r)
}

func BenchmarkMod64SmallNegDivisor(b *testing.B) {
	r := int64(-1)
	for i := 1; i <= b.N; i++ {
		r = (int64(i) - r) % -int64(i)
	}
	Output = int(r)
}

func BenchmarkMod64SmallNegDividend(b *testing.B) {
	r := int64(-1)
	for i := 1; i <= b.N; i++ {
		r = -(int64(i) - r) % int64(i)
	}
	Output = int(r)
}

func BenchmarkMod64SmallNegBoth(b *testing.B) {
	r := int64(1)
	for i := 1; i <= b.N; i++ {
		r = -(int64(i) + r) % -int64(i)
	}
	Output = int(r)
}

func BenchmarkMod64Unsigned(b *testing.B) {
	r := uint64(1)
	for i := 1; i <= b.N; i++ {
		r = (uint64(0x7fffffffffffffff) - uint64(i) - (r & 1)) % uint64(i)
	}
	Output = int(r)
}

func BenchmarkMod64(b *testing.B) {
	r := int64(1)
	for i := 1; i <= b.N; i++ {
		r = (int64(0x7fffffffffffffff) - int64(i) - (r & 1)) % int64(i)
	}
	Output = int(r)
}

func BenchmarkMod64NegDivisor(b *testing.B) {
	r := int64(-1)
	for i := 1; i <= b.N; i++ {
		r = (int64(0x7fffffffffffffff) - int64(i) - (r & 1)) % -int64(i)
	}
	Output = int(r)
}

func BenchmarkMod64NegDividend(b *testing.B) {
	r := int64(-1)
	for i := 1; i <= b.N; i++ {
		r = -(int64(0x7fffffffffffffff) - int64(i) - (r & 1)) % int64(i)
	}
	Output = int(r)
}

func BenchmarkMod64NegBoth(b *testing.B) {
	r := int64(1)
	for i := 1; i <= b.N; i++ {
		r = -(int64(0x7fffffffffffffff) - int64(i) - (r & 1)) % -int64(i)
	}
	Output = int(r)
}
