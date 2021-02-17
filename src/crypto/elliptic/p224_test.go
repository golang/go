// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"math/big"
	"math/bits"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

var toFromBigTests = []string{
	"0",
	"1",
	"23",
	"b70e0cb46bb4bf7f321390b94a03c1d356c01122343280d6105c1d21",
	"706a46d476dcb76798e6046d89474788d164c18032d268fd10704fa6",
}

func p224AlternativeToBig(in *p224FieldElement) *big.Int {
	ret := new(big.Int)
	tmp := new(big.Int)

	for i := len(in) - 1; i >= 0; i-- {
		ret.Lsh(ret, 28)
		tmp.SetInt64(int64(in[i]))
		ret.Add(ret, tmp)
	}
	ret.Mod(ret, P224().Params().P)
	return ret
}

func TestP224ToFromBig(t *testing.T) {
	for i, test := range toFromBigTests {
		n, _ := new(big.Int).SetString(test, 16)
		var x p224FieldElement
		p224FromBig(&x, n)
		m := p224ToBig(&x)
		if n.Cmp(m) != 0 {
			t.Errorf("#%d: %x != %x", i, n, m)
		}
		q := p224AlternativeToBig(&x)
		if n.Cmp(q) != 0 {
			t.Errorf("#%d: %x != %x (alternative)", i, n, q)
		}
	}
}

// quickCheckConfig32 will make each quickcheck test run (32 * -quickchecks)
// times. The default value of -quickchecks is 100.
var quickCheckConfig32 = &quick.Config{MaxCountScale: 32}

// weirdLimbs can be combined to generate a range of edge-case field elements.
var weirdLimbs = [...]uint32{
	0, 1, (1 << 29) - 1,
	(1 << 12), (1 << 12) - 1,
	(1 << 28), (1 << 28) - 1,
}

func generateLimb(rand *rand.Rand) uint32 {
	const bottom29Bits = 0x1fffffff
	n := rand.Intn(len(weirdLimbs) + 3)
	switch n {
	case len(weirdLimbs):
		// Random value.
		return uint32(rand.Int31n(1 << 29))
	case len(weirdLimbs) + 1:
		// Sum of two values.
		k := generateLimb(rand) + generateLimb(rand)
		return k & bottom29Bits
	case len(weirdLimbs) + 2:
		// Difference of two values.
		k := generateLimb(rand) - generateLimb(rand)
		return k & bottom29Bits
	default:
		return weirdLimbs[n]
	}
}

func (p224FieldElement) Generate(rand *rand.Rand, size int) reflect.Value {
	return reflect.ValueOf(p224FieldElement{
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
	})
}

func isInBounds(x *p224FieldElement) bool {
	return bits.Len32(x[0]) <= 29 &&
		bits.Len32(x[1]) <= 29 &&
		bits.Len32(x[2]) <= 29 &&
		bits.Len32(x[3]) <= 29 &&
		bits.Len32(x[4]) <= 29 &&
		bits.Len32(x[5]) <= 29 &&
		bits.Len32(x[6]) <= 29 &&
		bits.Len32(x[7]) <= 29
}

func TestP224Mul(t *testing.T) {
	mulMatchesBigInt := func(a, b, out p224FieldElement) bool {
		var tmp p224LargeFieldElement
		p224Mul(&out, &a, &b, &tmp)

		exp := new(big.Int).Mul(p224AlternativeToBig(&a), p224AlternativeToBig(&b))
		exp.Mod(exp, P224().Params().P)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 || !isInBounds(&out) {
			t.Logf("a = %x", a)
			t.Logf("b = %x", b)
			t.Logf("p224Mul(a, b) = %x = %v", out, got)
			t.Logf("a * b = %v", exp)
			return false
		}

		return true
	}

	a := p224FieldElement{0xfffffff, 0xfffffff, 0xf00ffff, 0x20f, 0x0, 0x0, 0x0, 0x0}
	b := p224FieldElement{1, 0, 0, 0, 0, 0, 0, 0}
	if !mulMatchesBigInt(a, b, p224FieldElement{}) {
		t.Fail()
	}

	if err := quick.Check(mulMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Square(t *testing.T) {
	squareMatchesBigInt := func(a, out p224FieldElement) bool {
		var tmp p224LargeFieldElement
		p224Square(&out, &a, &tmp)

		exp := p224AlternativeToBig(&a)
		exp.Mul(exp, exp)
		exp.Mod(exp, P224().Params().P)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 || !isInBounds(&out) {
			t.Logf("a = %x", a)
			t.Logf("p224Square(a, b) = %x = %v", out, got)
			t.Logf("a * a = %v", exp)
			return false
		}

		return true
	}

	if err := quick.Check(squareMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Add(t *testing.T) {
	addMatchesBigInt := func(a, b, out p224FieldElement) bool {
		p224Add(&out, &a, &b)

		exp := new(big.Int).Add(p224AlternativeToBig(&a), p224AlternativeToBig(&b))
		exp.Mod(exp, P224().Params().P)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 {
			t.Logf("a = %x", a)
			t.Logf("b = %x", b)
			t.Logf("p224Add(a, b) = %x = %v", out, got)
			t.Logf("a + b = %v", exp)
			return false
		}

		return true
	}

	if err := quick.Check(addMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Reduce(t *testing.T) {
	reduceMatchesBigInt := func(a p224FieldElement) bool {
		out := a
		// TODO: generate higher values for functions like p224Reduce that are
		// expected to work with higher input bounds.
		p224Reduce(&out)

		exp := p224AlternativeToBig(&a)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 || !isInBounds(&out) {
			t.Logf("a = %x = %v", a, exp)
			t.Logf("p224Reduce(a) = %x = %v", out, got)
			return false
		}

		return true
	}

	if err := quick.Check(reduceMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Contract(t *testing.T) {
	contractMatchesBigInt := func(a, out p224FieldElement) bool {
		p224Contract(&out, &a)

		exp := p224AlternativeToBig(&a)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 {
			t.Logf("a = %x = %v", a, exp)
			t.Logf("p224Contract(a) = %x = %v", out, got)
			return false
		}

		// Check that out < P.
		for i := range p224P {
			k := 8 - i - 1
			if out[k] > p224P[k] {
				t.Logf("p224Contract(a) = %x", out)
				return false
			}
			if out[k] < p224P[k] {
				return true
			}
		}
		t.Logf("p224Contract(a) = %x", out)
		return false
	}

	if !contractMatchesBigInt(p224P, p224FieldElement{}) {
		t.Error("p224Contract(p) is broken")
	}
	pMinus1 := p224FieldElement{0, 0, 0, 0xffff000, 0xfffffff, 0xfffffff, 0xfffffff, 0xfffffff}
	if !contractMatchesBigInt(pMinus1, p224FieldElement{}) {
		t.Error("p224Contract(p - 1) is broken")
	}
	// Check that we can handle input above p, but lowest limb zero.
	a := p224FieldElement{0, 1, 0, 0xffff000, 0xfffffff, 0xfffffff, 0xfffffff, 0xfffffff}
	if !contractMatchesBigInt(a, p224FieldElement{}) {
		t.Error("p224Contract(p + 2²⁸) is broken")
	}
	// Check that we can handle input above p, but lowest three limbs zero.
	b := p224FieldElement{0, 0, 0, 0xffff001, 0xfffffff, 0xfffffff, 0xfffffff, 0xfffffff}
	if !contractMatchesBigInt(b, p224FieldElement{}) {
		t.Error("p224Contract(p + 2⁸⁴) is broken")
	}

	if err := quick.Check(contractMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224IsZero(t *testing.T) {
	if got := p224IsZero(&p224FieldElement{}); got != 1 {
		t.Errorf("p224IsZero(0) = %d, expected 1", got)
	}
	if got := p224IsZero((*p224FieldElement)(&p224P)); got != 1 {
		t.Errorf("p224IsZero(p) = %d, expected 1", got)
	}
	if got := p224IsZero(&p224FieldElement{1}); got != 0 {
		t.Errorf("p224IsZero(1) = %d, expected 0", got)
	}

	isZeroMatchesBigInt := func(a p224FieldElement) bool {
		isZero := p224IsZero(&a)

		big := p224AlternativeToBig(&a)
		if big.Sign() == 0 && isZero != 1 {
			return false
		}
		if big.Sign() != 0 && isZero != 0 {
			return false
		}
		return true
	}

	if err := quick.Check(isZeroMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Invert(t *testing.T) {
	var out p224FieldElement

	p224Invert(&out, &p224FieldElement{})
	if got := p224IsZero(&out); got != 1 {
		t.Errorf("p224Invert(0) = %x, expected 0", out)
	}

	p224Invert(&out, (*p224FieldElement)(&p224P))
	if got := p224IsZero(&out); got != 1 {
		t.Errorf("p224Invert(p) = %x, expected 0", out)
	}

	p224Invert(&out, &p224FieldElement{1})
	p224Contract(&out, &out)
	if out != (p224FieldElement{1}) {
		t.Errorf("p224Invert(1) = %x, expected 1", out)
	}

	var tmp p224LargeFieldElement
	a := p224FieldElement{1, 2, 3, 4, 5, 6, 7, 8}
	p224Invert(&out, &a)
	p224Mul(&out, &out, &a, &tmp)
	p224Contract(&out, &out)
	if out != (p224FieldElement{1}) {
		t.Errorf("p224Invert(a) * a = %x, expected 1", out)
	}
}
