// Copyright (c) 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package edwards25519

import (
	"bytes"
	"encoding/hex"
	"math/big"
	mathrand "math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

// quickCheckConfig returns a quick.Config that scales the max count by the
// given factor if the -short flag is not set.
func quickCheckConfig(slowScale int) *quick.Config {
	cfg := new(quick.Config)
	if !testing.Short() {
		cfg.MaxCountScale = float64(slowScale)
	}
	return cfg
}

var scOneBytes = [32]byte{1}
var scOne, _ = new(Scalar).SetCanonicalBytes(scOneBytes[:])
var scMinusOne = new(Scalar).Subtract(new(Scalar), scOne)

// Generate returns a valid (reduced modulo l) Scalar with a distribution
// weighted towards high, low, and edge values.
func (Scalar) Generate(rand *mathrand.Rand, size int) reflect.Value {
	var s [32]byte
	diceRoll := rand.Intn(100)
	switch {
	case diceRoll == 0:
	case diceRoll == 1:
		s = scOneBytes
	case diceRoll == 2:
		s = [32]byte(scMinusOne.Bytes())
	case diceRoll < 5:
		// Generate a low scalar in [0, 2^125).
		rand.Read(s[:16])
		s[15] &= (1 << 5) - 1
	case diceRoll < 10:
		// Generate a high scalar in [2^252, 2^252 + 2^124).
		s[31] = 1 << 4
		rand.Read(s[:16])
		s[15] &= (1 << 4) - 1
	default:
		// Generate a valid scalar in [0, l) by returning [0, 2^252) which has a
		// negligibly different distribution (the former has a 2^-127.6 chance
		// of being out of the latter range).
		rand.Read(s[:])
		s[31] &= (1 << 4) - 1
	}

	val := Scalar{}
	fiatScalarFromBytes((*[4]uint64)(&val.s), &s)
	fiatScalarToMontgomery(&val.s, (*fiatScalarNonMontgomeryDomainFieldElement)(&val.s))

	return reflect.ValueOf(val)
}

func TestScalarGenerate(t *testing.T) {
	f := func(sc Scalar) bool {
		return isReduced(sc.Bytes())
	}
	if err := quick.Check(f, quickCheckConfig(1024)); err != nil {
		t.Errorf("generated unreduced scalar: %v", err)
	}
}

func TestScalarSetCanonicalBytes(t *testing.T) {
	f1 := func(in [32]byte, sc Scalar) bool {
		// Mask out top 4 bits to guarantee value falls in [0, l).
		in[len(in)-1] &= (1 << 4) - 1
		if _, err := sc.SetCanonicalBytes(in[:]); err != nil {
			return false
		}
		repr := sc.Bytes()
		return bytes.Equal(in[:], repr) && isReduced(repr)
	}
	if err := quick.Check(f1, quickCheckConfig(1024)); err != nil {
		t.Errorf("failed bytes->scalar->bytes round-trip: %v", err)
	}

	f2 := func(sc1, sc2 Scalar) bool {
		if _, err := sc2.SetCanonicalBytes(sc1.Bytes()); err != nil {
			return false
		}
		return sc1 == sc2
	}
	if err := quick.Check(f2, quickCheckConfig(1024)); err != nil {
		t.Errorf("failed scalar->bytes->scalar round-trip: %v", err)
	}

	expectReject := func(b []byte) {
		t.Helper()
		s := scOne
		if out, err := s.SetCanonicalBytes(b[:]); err == nil {
			t.Errorf("SetCanonicalBytes worked on a non-canonical value")
		} else if s != scOne {
			t.Errorf("SetCanonicalBytes modified its receiver")
		} else if out != nil {
			t.Errorf("SetCanonicalBytes did not return nil with an error")
		}
	}

	b := scMinusOne.Bytes()
	b[0] += 1
	expectReject(b)

	b = scMinusOne.Bytes()
	b[31] += 1
	expectReject(b)

	b = scMinusOne.Bytes()
	b[31] |= 0b1000_0000
	expectReject(b)
}

func TestScalarSetUniformBytes(t *testing.T) {
	mod, _ := new(big.Int).SetString("27742317777372353535851937790883648493", 10)
	mod.Add(mod, new(big.Int).Lsh(big.NewInt(1), 252))
	f := func(in [64]byte, sc Scalar) bool {
		sc.SetUniformBytes(in[:])
		repr := sc.Bytes()
		if !isReduced(repr) {
			return false
		}
		scBig := bigIntFromLittleEndianBytes(repr[:])
		inBig := bigIntFromLittleEndianBytes(in[:])
		return inBig.Mod(inBig, mod).Cmp(scBig) == 0
	}
	if err := quick.Check(f, quickCheckConfig(1024)); err != nil {
		t.Error(err)
	}
}

func TestScalarSetBytesWithClamping(t *testing.T) {
	// Generated with libsodium.js 1.0.18 crypto_scalarmult_ed25519_base.

	random := "633d368491364dc9cd4c1bf891b1d59460face1644813240a313e61f2c88216e"
	s, _ := new(Scalar).SetBytesWithClamping(decodeHex(random))
	p := new(Point).ScalarBaseMult(s)
	want := "1d87a9026fd0126a5736fe1628c95dd419172b5b618457e041c9c861b2494a94"
	if got := hex.EncodeToString(p.Bytes()); got != want {
		t.Errorf("random: got %q, want %q", got, want)
	}

	zero := "0000000000000000000000000000000000000000000000000000000000000000"
	s, _ = new(Scalar).SetBytesWithClamping(decodeHex(zero))
	p = new(Point).ScalarBaseMult(s)
	want = "693e47972caf527c7883ad1b39822f026f47db2ab0e1919955b8993aa04411d1"
	if got := hex.EncodeToString(p.Bytes()); got != want {
		t.Errorf("zero: got %q, want %q", got, want)
	}

	one := "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
	s, _ = new(Scalar).SetBytesWithClamping(decodeHex(one))
	p = new(Point).ScalarBaseMult(s)
	want = "12e9a68b73fd5aacdbcaf3e88c46fea6ebedb1aa84eed1842f07f8edab65e3a7"
	if got := hex.EncodeToString(p.Bytes()); got != want {
		t.Errorf("one: got %q, want %q", got, want)
	}
}

func bigIntFromLittleEndianBytes(b []byte) *big.Int {
	bb := make([]byte, len(b))
	for i := range b {
		bb[i] = b[len(b)-i-1]
	}
	return new(big.Int).SetBytes(bb)
}

func TestScalarMultiplyDistributesOverAdd(t *testing.T) {
	multiplyDistributesOverAdd := func(x, y, z Scalar) bool {
		// Compute t1 = (x+y)*z
		var t1 Scalar
		t1.Add(&x, &y)
		t1.Multiply(&t1, &z)

		// Compute t2 = x*z + y*z
		var t2 Scalar
		var t3 Scalar
		t2.Multiply(&x, &z)
		t3.Multiply(&y, &z)
		t2.Add(&t2, &t3)

		reprT1, reprT2 := t1.Bytes(), t2.Bytes()

		return t1 == t2 && isReduced(reprT1) && isReduced(reprT2)
	}

	if err := quick.Check(multiplyDistributesOverAdd, quickCheckConfig(1024)); err != nil {
		t.Error(err)
	}
}

func TestScalarAddLikeSubNeg(t *testing.T) {
	addLikeSubNeg := func(x, y Scalar) bool {
		// Compute t1 = x - y
		var t1 Scalar
		t1.Subtract(&x, &y)

		// Compute t2 = -y + x
		var t2 Scalar
		t2.Negate(&y)
		t2.Add(&t2, &x)

		return t1 == t2 && isReduced(t1.Bytes())
	}

	if err := quick.Check(addLikeSubNeg, quickCheckConfig(1024)); err != nil {
		t.Error(err)
	}
}

func TestScalarNonAdjacentForm(t *testing.T) {
	s, _ := (&Scalar{}).SetCanonicalBytes([]byte{
		0x1a, 0x0e, 0x97, 0x8a, 0x90, 0xf6, 0x62, 0x2d,
		0x37, 0x47, 0x02, 0x3f, 0x8a, 0xd8, 0x26, 0x4d,
		0xa7, 0x58, 0xaa, 0x1b, 0x88, 0xe0, 0x40, 0xd1,
		0x58, 0x9e, 0x7b, 0x7f, 0x23, 0x76, 0xef, 0x09,
	})

	expectedNaf := [256]int8{
		0, 13, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, -9, 0, 0, 0, 0, -11, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 9, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 11, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0,
		-9, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 9, 0,
		0, 0, 0, -15, 0, 0, 0, 0, -7, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, -3, 0,
		0, 0, 0, -11, 0, 0, 0, 0, -7, 0, 0, 0, 0, -13, 0, 0, 0, 0, 11, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, -15, 0, 0, 0, 0, 1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 13, 0, 0, 0,
		0, 0, 0, 11, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 7,
		0, 0, 0, 0, 0, -15, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
	}

	sNaf := s.nonAdjacentForm(5)

	for i := 0; i < 256; i++ {
		if expectedNaf[i] != sNaf[i] {
			t.Errorf("Wrong digit at position %d, got %d, expected %d", i, sNaf[i], expectedNaf[i])
		}
	}
}

type notZeroScalar Scalar

func (notZeroScalar) Generate(rand *mathrand.Rand, size int) reflect.Value {
	var s Scalar
	var isNonZero uint64
	for isNonZero == 0 {
		s = Scalar{}.Generate(rand, size).Interface().(Scalar)
		fiatScalarNonzero(&isNonZero, (*[4]uint64)(&s.s))
	}
	return reflect.ValueOf(notZeroScalar(s))
}

func TestScalarEqual(t *testing.T) {
	if scOne.Equal(scMinusOne) == 1 {
		t.Errorf("scOne.Equal(&scMinusOne) is true")
	}
	if scMinusOne.Equal(scMinusOne) == 0 {
		t.Errorf("scMinusOne.Equal(&scMinusOne) is false")
	}
}
