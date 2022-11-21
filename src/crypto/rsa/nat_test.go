// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bytes"
	"math/big"
	"math/bits"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

// Generate generates an even nat. It's used by testing/quick to produce random
// *nat values for quick.Check invocations.
func (*nat) Generate(r *rand.Rand, size int) reflect.Value {
	limbs := make([]uint, size)
	for i := 0; i < size; i++ {
		limbs[i] = uint(r.Uint64()) & ((1 << _W) - 2)
	}
	return reflect.ValueOf(&nat{limbs})
}

func testModAddCommutative(a *nat, b *nat) bool {
	mLimbs := make([]uint, len(a.limbs))
	for i := 0; i < len(mLimbs); i++ {
		mLimbs[i] = _MASK
	}
	m := modulusFromNat(&nat{mLimbs})
	aPlusB := new(nat).set(a)
	aPlusB.modAdd(b, m)
	bPlusA := new(nat).set(b)
	bPlusA.modAdd(a, m)
	return aPlusB.cmpEq(bPlusA) == 1
}

func TestModAddCommutative(t *testing.T) {
	err := quick.Check(testModAddCommutative, &quick.Config{})
	if err != nil {
		t.Error(err)
	}
}

func testModSubThenAddIdentity(a *nat, b *nat) bool {
	mLimbs := make([]uint, len(a.limbs))
	for i := 0; i < len(mLimbs); i++ {
		mLimbs[i] = _MASK
	}
	m := modulusFromNat(&nat{mLimbs})
	original := new(nat).set(a)
	a.modSub(b, m)
	a.modAdd(b, m)
	return a.cmpEq(original) == 1
}

func TestModSubThenAddIdentity(t *testing.T) {
	err := quick.Check(testModSubThenAddIdentity, &quick.Config{})
	if err != nil {
		t.Error(err)
	}
}

func testMontgomeryRoundtrip(a *nat) bool {
	one := &nat{make([]uint, len(a.limbs))}
	one.limbs[0] = 1
	aPlusOne := new(nat).set(a)
	aPlusOne.add(1, one)
	m := modulusFromNat(aPlusOne)
	monty := new(nat).set(a)
	monty.montgomeryRepresentation(m)
	aAgain := new(nat).set(monty)
	aAgain.montgomeryMul(monty, one, m)
	return a.cmpEq(aAgain) == 1
}

func TestMontgomeryRoundtrip(t *testing.T) {
	err := quick.Check(testMontgomeryRoundtrip, &quick.Config{})
	if err != nil {
		t.Error(err)
	}
}

func TestFromBig(t *testing.T) {
	expected := []byte{0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
	theBig := new(big.Int).SetBytes(expected)
	actual := new(nat).setBig(theBig).fillBytes(make([]byte, len(expected)))
	if !bytes.Equal(actual, expected) {
		t.Errorf("%+x != %+x", actual, expected)
	}
}

func TestFillBytes(t *testing.T) {
	xBytes := []byte{0xAA, 0xFF, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}
	x := new(nat).setBytes(xBytes)
	for l := 20; l >= len(xBytes); l-- {
		buf := make([]byte, l)
		rand.Read(buf)
		actual := x.fillBytes(buf)
		expected := make([]byte, l)
		copy(expected[l-len(xBytes):], xBytes)
		if !bytes.Equal(actual, expected) {
			t.Errorf("%d: %+v != %+v", l, actual, expected)
		}
	}
	for l := len(xBytes) - 1; l >= 0; l-- {
		(func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%d: expected panic", l)
				}
			}()
			x.fillBytes(make([]byte, l))
		})()
	}
}

func TestFromBytes(t *testing.T) {
	f := func(xBytes []byte) bool {
		if len(xBytes) == 0 {
			return true
		}
		actual := new(nat).setBytes(xBytes).fillBytes(make([]byte, len(xBytes)))
		if !bytes.Equal(actual, xBytes) {
			t.Errorf("%+x != %+x", actual, xBytes)
			return false
		}
		return true
	}

	err := quick.Check(f, &quick.Config{})
	if err != nil {
		t.Error(err)
	}

	f([]byte{0xFF, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88})
	f(bytes.Repeat([]byte{0xFF}, _W))
}

func TestShiftIn(t *testing.T) {
	if bits.UintSize != 64 {
		t.Skip("examples are only valid in 64 bit")
	}
	examples := []struct {
		m, x, expected []byte
		y              uint64
	}{{
		m:        []byte{13},
		x:        []byte{0},
		y:        0x7FFF_FFFF_FFFF_FFFF,
		expected: []byte{7},
	}, {
		m:        []byte{13},
		x:        []byte{7},
		y:        0x7FFF_FFFF_FFFF_FFFF,
		expected: []byte{11},
	}, {
		m:        []byte{0x06, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d},
		x:        make([]byte, 9),
		y:        0x7FFF_FFFF_FFFF_FFFF,
		expected: []byte{0x00, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
	}, {
		m:        []byte{0x06, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d},
		x:        []byte{0x00, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		y:        0,
		expected: []byte{0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08},
	}}

	for i, tt := range examples {
		m := modulusFromNat(new(nat).setBytes(tt.m))
		got := new(nat).setBytes(tt.x).expandFor(m).shiftIn(uint(tt.y), m)
		if got.cmpEq(new(nat).setBytes(tt.expected).expandFor(m)) != 1 {
			t.Errorf("%d: got %x, expected %x", i, got, tt.expected)
		}
	}
}

func TestModulusAndNatSizes(t *testing.T) {
	// These are 126 bit (2 * _W on 64-bit architectures) values, serialized as
	// 128 bits worth of bytes. If leading zeroes are stripped, they fit in two
	// limbs, if they are not, they fit in three. This can be a problem because
	// modulus strips leading zeroes and nat does not.
	m := modulusFromNat(new(nat).setBytes([]byte{
		0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}))
	x := new(nat).setBytes([]byte{
		0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe})
	x.expandFor(m) // must not panic for shrinking
}

func TestExpand(t *testing.T) {
	sliced := []uint{1, 2, 3, 4}
	examples := []struct {
		in  []uint
		n   int
		out []uint
	}{{
		[]uint{1, 2},
		4,
		[]uint{1, 2, 0, 0},
	}, {
		sliced[:2],
		4,
		[]uint{1, 2, 0, 0},
	}, {
		[]uint{1, 2},
		2,
		[]uint{1, 2},
	}, {
		[]uint{1, 2, 0},
		2,
		[]uint{1, 2},
	}}

	for i, tt := range examples {
		got := (&nat{tt.in}).expand(tt.n)
		if len(got.limbs) != len(tt.out) || got.cmpEq(&nat{tt.out}) != 1 {
			t.Errorf("%d: got %x, expected %x", i, got, tt.out)
		}
	}
}

func TestMod(t *testing.T) {
	m := modulusFromNat(new(nat).setBytes([]byte{0x06, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d}))
	x := new(nat).setBytes([]byte{0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01})
	out := new(nat)
	out.mod(x, m)
	expected := new(nat).setBytes([]byte{0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09})
	if out.cmpEq(expected) != 1 {
		t.Errorf("%+v != %+v", out, expected)
	}
}

func TestModSub(t *testing.T) {
	m := modulusFromNat(&nat{[]uint{13}})
	x := &nat{[]uint{6}}
	y := &nat{[]uint{7}}
	x.modSub(y, m)
	expected := &nat{[]uint{12}}
	if x.cmpEq(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
	x.modSub(y, m)
	expected = &nat{[]uint{5}}
	if x.cmpEq(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
}

func TestModAdd(t *testing.T) {
	m := modulusFromNat(&nat{[]uint{13}})
	x := &nat{[]uint{6}}
	y := &nat{[]uint{7}}
	x.modAdd(y, m)
	expected := &nat{[]uint{0}}
	if x.cmpEq(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
	x.modAdd(y, m)
	expected = &nat{[]uint{7}}
	if x.cmpEq(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
}

func TestExp(t *testing.T) {
	m := modulusFromNat(&nat{[]uint{13}})
	x := &nat{[]uint{3}}
	out := &nat{[]uint{0}}
	out.exp(x, []byte{12}, m)
	expected := &nat{[]uint{1}}
	if out.cmpEq(expected) != 1 {
		t.Errorf("%+v != %+v", out, expected)
	}
}

func makeBenchmarkModulus() *modulus {
	m := make([]uint, 32)
	for i := 0; i < 32; i++ {
		m[i] = _MASK
	}
	return modulusFromNat(&nat{limbs: m})
}

func makeBenchmarkValue() *nat {
	x := make([]uint, 32)
	for i := 0; i < 32; i++ {
		x[i] = _MASK - 1
	}
	return &nat{limbs: x}
}

func makeBenchmarkExponent() []byte {
	e := make([]byte, 256)
	for i := 0; i < 32; i++ {
		e[i] = 0xFF
	}
	return e
}

func BenchmarkModAdd(b *testing.B) {
	x := makeBenchmarkValue()
	y := makeBenchmarkValue()
	m := makeBenchmarkModulus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.modAdd(y, m)
	}
}

func BenchmarkModSub(b *testing.B) {
	x := makeBenchmarkValue()
	y := makeBenchmarkValue()
	m := makeBenchmarkModulus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.modSub(y, m)
	}
}

func BenchmarkMontgomeryRepr(b *testing.B) {
	x := makeBenchmarkValue()
	m := makeBenchmarkModulus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.montgomeryRepresentation(m)
	}
}

func BenchmarkMontgomeryMul(b *testing.B) {
	x := makeBenchmarkValue()
	y := makeBenchmarkValue()
	out := makeBenchmarkValue()
	m := makeBenchmarkModulus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out.montgomeryMul(x, y, m)
	}
}

func BenchmarkModMul(b *testing.B) {
	x := makeBenchmarkValue()
	y := makeBenchmarkValue()
	m := makeBenchmarkModulus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.modMul(y, m)
	}
}

func BenchmarkExpBig(b *testing.B) {
	out := new(big.Int)
	exponentBytes := makeBenchmarkExponent()
	x := new(big.Int).SetBytes(exponentBytes)
	e := new(big.Int).SetBytes(exponentBytes)
	n := new(big.Int).SetBytes(exponentBytes)
	one := new(big.Int).SetUint64(1)
	n.Add(n, one)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out.Exp(x, e, n)
	}
}

func BenchmarkExp(b *testing.B) {
	x := makeBenchmarkValue()
	e := makeBenchmarkExponent()
	out := makeBenchmarkValue()
	m := makeBenchmarkModulus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out.exp(x, e, m)
	}
}
