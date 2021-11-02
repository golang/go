// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trie

import (
	"math/rand"
	"testing"
)

func TestMask(t *testing.T) {
	for _, c := range []struct {
		p    prefix
		b    bitpos
		want prefix
	}{
		{
			p:    0b00001000,
			b:    0b00000100,
			want: 0b00001011,
		}, {
			p:    0b01011011,
			b:    0b00000000,
			want: ^prefix(0),
		}, {
			p:    0b01011011,
			b:    0b00000001,
			want: 0b01011010,
		}, {
			p:    0b01011011,
			b:    0b00000010,
			want: 0b01011001,
		}, {
			p:    0b01011011,
			b:    0b00000100,
			want: 0b01011011,
		}, {
			p:    0b01011011,
			b:    0b00001000,
			want: 0b01010111,
		}, {
			p:    0b01011011,
			b:    0b00010000,
			want: 0b01001111,
		}, {
			p:    0b01011011,
			b:    0b00100000,
			want: 0b01011111,
		}, {
			p:    0b01011011,
			b:    0b01000000,
			want: 0b00111111,
		}, {
			p:    0b01011011,
			b:    0b01000000,
			want: 0b00111111,
		}, {
			p:    0b01011011,
			b:    0b10000000,
			want: 0b01111111,
		},
	} {
		if got := mask(c.p, c.b); got != c.want {
			t.Errorf("mask(%#b,%#b) got %#b. want %#b", c.p, c.b, got, c.want)
		}
	}
}

func TestMaskImpotent(t *testing.T) {
	// test mask(mask(p, b), b) == mask(p,b)
	for _, p := range []prefix{
		0b0, 0b1, 0b100, ^prefix(0b0), ^prefix(0b10),
	} {
		for _, b := range []bitpos{
			0, 0b1, 1 << 2, 1 << 63,
		} {
			once := mask(p, b)
			twice := mask(once, b)
			if once != twice {
				t.Errorf("mask(mask(%#b,%#b), %#b) != mask(%#b,%#b) got %#b. want %#b",
					p, b, b, p, b, twice, once)
			}
		}
	}
}

func TestMatchPrefix(t *testing.T) {
	for _, c := range []struct {
		k prefix
		p prefix
		b bitpos
	}{
		{
			k: 0b1000,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b1001,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b1010,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b1011,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b1100,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b1101,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b1110,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b1111,
			p: 0b1011,
			b: 0b0100,
		},
	} {
		if !matchPrefix(c.k, c.p, c.b) {
			t.Errorf("matchPrefix(%#b, %#b,%#b) should be true", c.k, c.p, c.b)
		}
	}
}

func TestNotMatchPrefix(t *testing.T) {
	for _, c := range []struct {
		k prefix
		p prefix
		b bitpos
	}{
		{
			k: 0b0000,
			p: 0b1011,
			b: 0b0100,
		}, {
			k: 0b0010,
			p: 0b1011,
			b: 0b0100,
		},
	} {
		if matchPrefix(c.k, c.p, c.b) {
			t.Errorf("matchPrefix(%#b, %#b,%#b) should be false", c.k, c.p, c.b)
		}
	}
}

func TestBranchingBit(t *testing.T) {
	for _, c := range []struct {
		x    prefix
		y    prefix
		want bitpos
	}{
		{
			x:    0b0000,
			y:    0b1011,
			want: 0b1000,
		}, {
			x:    0b1010,
			y:    0b1011,
			want: 0b0001,
		}, {
			x:    0b1011,
			y:    0b1111,
			want: 0b0100,
		}, {
			x:    0b1011,
			y:    0b1001,
			want: 0b0010,
		},
	} {
		if got := branchingBit(c.x, c.y); got != c.want {
			t.Errorf("branchingBit(%#b, %#b,) is not expected value. got %#b want %#b",
				c.x, c.y, got, c.want)
		}
	}
}

func TestZeroBit(t *testing.T) {
	for _, c := range []struct {
		k prefix
		b bitpos
	}{
		{
			k: 0b1000,
			b: 0b0100,
		}, {
			k: 0b1001,
			b: 0b0100,
		}, {
			k: 0b1010,
			b: 0b0100,
		},
	} {
		if !zeroBit(c.k, c.b) {
			t.Errorf("zeroBit(%#b, %#b) should be true", c.k, c.b)
		}
	}
}
func TestZeroBitFails(t *testing.T) {
	for _, c := range []struct {
		k prefix
		b bitpos
	}{
		{
			k: 0b1000,
			b: 0b1000,
		}, {
			k: 0b1001,
			b: 0b0001,
		}, {
			k: 0b1010,
			b: 0b0010,
		}, {
			k: 0b1011,
			b: 0b0001,
		},
	} {
		if zeroBit(c.k, c.b) {
			t.Errorf("zeroBit(%#b, %#b) should be false", c.k, c.b)
		}
	}
}

func TestOrd(t *testing.T) {
	a := bitpos(0b0010)
	b := bitpos(0b1000)
	if ord(a, b) {
		t.Errorf("ord(%#b, %#b) should be false", a, b)
	}
	if !ord(b, a) {
		t.Errorf("ord(%#b, %#b) should be true", b, a)
	}
	if ord(a, a) {
		t.Errorf("ord(%#b, %#b) should be false", a, a)
	}
	if !ord(a, 0) {
		t.Errorf("ord(%#b, %#b) should be true", a, 0)
	}
}

func TestPrefixesOverlapLemma(t *testing.T) {
	// test
	//   mask(p, fbb) == mask(q, fbb)
	// iff
	//   m > n && matchPrefix(q, p, m) or  (note: big endian encoding)
	//   m < n && matchPrefix(p, q, n) or  (note: big endian encoding)
	//   m ==n && p == q

	// Case 1: mask(p, fbb) == mask(q, fbb) => m > n && matchPrefix(q, p, m)
	m, n := bitpos(1<<2), bitpos(1<<1)
	p, q := mask(0b100, m), mask(0b010, n)
	if !(prefixesOverlap(p, m, q, n) && m > n && matchPrefix(q, p, m)) {
		t.Errorf("prefixesOverlap(%#b, %#b, %#b, %#b) lemma does not hold",
			p, m, q, n)
	}
	// Case 2: mask(p, fbb) == mask(q, fbb) => m < n && matchPrefix(p, q, n)
	m, n = bitpos(1<<2), bitpos(1<<3)
	p, q = mask(0b100, m), mask(0b1000, n)
	if !(prefixesOverlap(p, m, q, n) && m < n && matchPrefix(p, q, n)) {
		t.Errorf("prefixesOverlap(%#b, %#b, %#b, %#b) lemma does not hold",
			p, m, q, n)
	}
	// Case 3: mask(p, fbb) == mask(q, fbb) => m < n && matchPrefix(p, q, n)
	m, n = bitpos(1<<2), bitpos(1<<2)
	p, q = mask(0b100, m), mask(0b001, n)
	if !(prefixesOverlap(p, m, q, n) && m == n && p == q) {
		t.Errorf("prefixesOverlap(%#b, %#b, %#b, %#b) lemma does not hold",
			p, m, q, n)
	}
	// Case 4: mask(p, fbb) != mask(q, fbb)
	m, n = bitpos(1<<1), bitpos(1<<1)
	p, q = mask(0b100, m), mask(0b001, n)
	if prefixesOverlap(p, m, q, n) ||
		(m > n && matchPrefix(q, p, m)) ||
		(m < n && matchPrefix(p, q, n)) ||
		(m == n && p == q) {
		t.Errorf("prefixesOverlap(%#b, %#b, %#b, %#b) lemma does not hold",
			p, m, q, n)
	}

	// Do a few more random cases
	r := rand.New(rand.NewSource(123))
	N := 2000
	for i := 0; i < N; i++ {
		m := bitpos(1 << (r.Uint64() % (64 + 1)))
		n := bitpos(1 << (r.Uint64() % (64 + 1)))

		p := mask(prefix(r.Uint64()), m)
		q := mask(prefix(r.Uint64()), n)

		lhs := prefixesOverlap(p, m, q, n)
		rhs := (m > n && matchPrefix(q, p, m)) ||
			(m < n && matchPrefix(p, q, n)) ||
			(m == n && p == q)

		if lhs != rhs {
			t.Errorf("prefixesOverlap(%#b, %#b, %#b, %#b) != <lemma> got %v. want %v",
				p, m, q, n, lhs, rhs)
		}
	}
}
