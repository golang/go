// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bigmod

import (
	"math/big"
	"math/bits"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

// Generate generates an even nat. It's used by testing/quick to produce random
// *nat values for quick.Check invocations.
func (*Nat) Generate(r *rand.Rand, size int) reflect.Value {
	limbs := make([]uint, size)
	for i := 0; i < size; i++ {
		limbs[i] = uint(r.Uint64()) & ((1 << _W) - 2)
	}
	return reflect.ValueOf(&Nat{limbs})
}

func testModAddCommutative(a *Nat, b *Nat) bool {
	m := maxModulus(uint(len(a.limbs)))
	aPlusB := new(Nat).set(a)
	aPlusB.Add(b, m)
	bPlusA := new(Nat).set(b)
	bPlusA.Add(a, m)
	return aPlusB.Equal(bPlusA) == 1
}

func TestModAddCommutative(t *testing.T) {
	err := quick.Check(testModAddCommutative, &quick.Config{})
	if err != nil {
		t.Error(err)
	}
}

func testModSubThenAddIdentity(a *Nat, b *Nat) bool {
	m := maxModulus(uint(len(a.limbs)))
	original := new(Nat).set(a)
	a.Sub(b, m)
	a.Add(b, m)
	return a.Equal(original) == 1
}

func TestModSubThenAddIdentity(t *testing.T) {
	err := quick.Check(testModSubThenAddIdentity, &quick.Config{})
	if err != nil {
		t.Error(err)
	}
}

func testMontgomeryRoundtrip(a *Nat) bool {
	one := &Nat{make([]uint, len(a.limbs))}
	one.limbs[0] = 1
	aPlusOne := new(big.Int).SetBytes(natBytes(a))
	aPlusOne.Add(aPlusOne, big.NewInt(1))
	m := NewModulusFromBig(aPlusOne)
	monty := new(Nat).set(a)
	monty.montgomeryRepresentation(m)
	aAgain := new(Nat).set(monty)
	aAgain.montgomeryMul(monty, one, m)
	return a.Equal(aAgain) == 1
}

func TestMontgomeryRoundtrip(t *testing.T) {
	err := quick.Check(testMontgomeryRoundtrip, &quick.Config{})
	if err != nil {
		t.Error(err)
	}
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
		m := modulusFromBytes(tt.m)
		got := natFromBytes(tt.x).ExpandFor(m).shiftIn(uint(tt.y), m)
		if got.Equal(natFromBytes(tt.expected).ExpandFor(m)) != 1 {
			t.Errorf("%d: got %x, expected %x", i, got, tt.expected)
		}
	}
}

func TestModulusAndNatSizes(t *testing.T) {
	// These are 126 bit (2 * _W on 64-bit architectures) values, serialized as
	// 128 bits worth of bytes. If leading zeroes are stripped, they fit in two
	// limbs, if they are not, they fit in three. This can be a problem because
	// modulus strips leading zeroes and nat does not.
	m := modulusFromBytes([]byte{
		0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff})
	xb := []byte{0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe}
	natFromBytes(xb).ExpandFor(m) // must not panic for shrinking
	NewNat().SetBytes(xb, m)
}

func TestSetBytes(t *testing.T) {
	tests := []struct {
		m, b []byte
		fail bool
	}{{
		m: []byte{0xff, 0xff},
		b: []byte{0x00, 0x01},
	}, {
		m:    []byte{0xff, 0xff},
		b:    []byte{0xff, 0xff},
		fail: true,
	}, {
		m: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		b: []byte{0x00, 0x01},
	}, {
		m: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		b: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe},
	}, {
		m:    []byte{0xff, 0xff},
		b:    []byte{0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		fail: true,
	}, {
		m:    []byte{0xff, 0xff},
		b:    []byte{0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		fail: true,
	}, {
		m: []byte{0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		b: []byte{0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe},
	}, {
		m:    []byte{0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		b:    []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe},
		fail: true,
	}, {
		m:    []byte{0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		b:    []byte{0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		fail: true,
	}, {
		m:    []byte{0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		b:    []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe},
		fail: true,
	}, {
		m:    []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd},
		b:    []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		fail: true,
	}}

	for i, tt := range tests {
		m := modulusFromBytes(tt.m)
		got, err := NewNat().SetBytes(tt.b, m)
		if err != nil {
			if !tt.fail {
				t.Errorf("%d: unexpected error: %v", i, err)
			}
			continue
		}
		if tt.fail {
			t.Errorf("%d: unexpected success", i)
			continue
		}
		if expected := natFromBytes(tt.b).ExpandFor(m); got.Equal(expected) != yes {
			t.Errorf("%d: got %x, expected %x", i, got, expected)
		}
	}

	f := func(xBytes []byte) bool {
		m := maxModulus(uint(len(xBytes)*8/_W + 1))
		got, err := NewNat().SetBytes(xBytes, m)
		if err != nil {
			return false
		}
		return got.Equal(natFromBytes(xBytes).ExpandFor(m)) == yes
	}

	err := quick.Check(f, &quick.Config{})
	if err != nil {
		t.Error(err)
	}
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
	}}

	for i, tt := range examples {
		got := (&Nat{tt.in}).expand(tt.n)
		if len(got.limbs) != len(tt.out) || got.Equal(&Nat{tt.out}) != 1 {
			t.Errorf("%d: got %x, expected %x", i, got, tt.out)
		}
	}
}

func TestMod(t *testing.T) {
	m := modulusFromBytes([]byte{0x06, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d})
	x := natFromBytes([]byte{0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01})
	out := new(Nat)
	out.Mod(x, m)
	expected := natFromBytes([]byte{0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09})
	if out.Equal(expected) != 1 {
		t.Errorf("%+v != %+v", out, expected)
	}
}

func TestModSub(t *testing.T) {
	m := modulusFromBytes([]byte{13})
	x := &Nat{[]uint{6}}
	y := &Nat{[]uint{7}}
	x.Sub(y, m)
	expected := &Nat{[]uint{12}}
	if x.Equal(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
	x.Sub(y, m)
	expected = &Nat{[]uint{5}}
	if x.Equal(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
}

func TestModAdd(t *testing.T) {
	m := modulusFromBytes([]byte{13})
	x := &Nat{[]uint{6}}
	y := &Nat{[]uint{7}}
	x.Add(y, m)
	expected := &Nat{[]uint{0}}
	if x.Equal(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
	x.Add(y, m)
	expected = &Nat{[]uint{7}}
	if x.Equal(expected) != 1 {
		t.Errorf("%+v != %+v", x, expected)
	}
}

func TestExp(t *testing.T) {
	m := modulusFromBytes([]byte{13})
	x := &Nat{[]uint{3}}
	out := &Nat{[]uint{0}}
	out.Exp(x, []byte{12}, m)
	expected := &Nat{[]uint{1}}
	if out.Equal(expected) != 1 {
		t.Errorf("%+v != %+v", out, expected)
	}
}

func natBytes(n *Nat) []byte {
	return n.Bytes(maxModulus(uint(len(n.limbs))))
}

func natFromBytes(b []byte) *Nat {
	bb := new(big.Int).SetBytes(b)
	return NewNat().setBig(bb)
}

func modulusFromBytes(b []byte) *Modulus {
	bb := new(big.Int).SetBytes(b)
	return NewModulusFromBig(bb)
}

// maxModulus returns the biggest modulus that can fit in n limbs.
func maxModulus(n uint) *Modulus {
	m := big.NewInt(1)
	m.Lsh(m, n*_W)
	m.Sub(m, big.NewInt(1))
	return NewModulusFromBig(m)
}

func makeBenchmarkModulus() *Modulus {
	return maxModulus(32)
}

func makeBenchmarkValue() *Nat {
	x := make([]uint, 32)
	for i := 0; i < 32; i++ {
		x[i] = _MASK - 1
	}
	return &Nat{limbs: x}
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
		x.Add(y, m)
	}
}

func BenchmarkModSub(b *testing.B) {
	x := makeBenchmarkValue()
	y := makeBenchmarkValue()
	m := makeBenchmarkModulus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Sub(y, m)
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
		x.Mul(y, m)
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
		out.Exp(x, e, m)
	}
}
