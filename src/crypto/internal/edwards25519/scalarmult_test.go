// Copyright (c) 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package edwards25519

import (
	"testing"
	"testing/quick"
)

var (
	// a random scalar generated using dalek.
	dalekScalar, _ = (&Scalar{}).SetCanonicalBytes([]byte{219, 106, 114, 9, 174, 249, 155, 89, 69, 203, 201, 93, 92, 116, 234, 187, 78, 115, 103, 172, 182, 98, 62, 103, 187, 136, 13, 100, 248, 110, 12, 4})
	// the above, times the edwards25519 basepoint.
	dalekScalarBasepoint, _ = new(Point).SetBytes([]byte{0xf4, 0xef, 0x7c, 0xa, 0x34, 0x55, 0x7b, 0x9f, 0x72, 0x3b, 0xb6, 0x1e, 0xf9, 0x46, 0x9, 0x91, 0x1c, 0xb9, 0xc0, 0x6c, 0x17, 0x28, 0x2d, 0x8b, 0x43, 0x2b, 0x5, 0x18, 0x6a, 0x54, 0x3e, 0x48})
)

func TestScalarMultSmallScalars(t *testing.T) {
	var z Scalar
	var p Point
	p.ScalarMult(&z, B)
	if I.Equal(&p) != 1 {
		t.Error("0*B != 0")
	}
	checkOnCurve(t, &p)

	scEight, _ := (&Scalar{}).SetCanonicalBytes([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	p.ScalarMult(scEight, B)
	if B.Equal(&p) != 1 {
		t.Error("1*B != 1")
	}
	checkOnCurve(t, &p)
}

func TestScalarMultVsDalek(t *testing.T) {
	var p Point
	p.ScalarMult(dalekScalar, B)
	if dalekScalarBasepoint.Equal(&p) != 1 {
		t.Error("Scalar mul does not match dalek")
	}
	checkOnCurve(t, &p)
}

func TestBaseMultVsDalek(t *testing.T) {
	var p Point
	p.ScalarBaseMult(dalekScalar)
	if dalekScalarBasepoint.Equal(&p) != 1 {
		t.Error("Scalar mul does not match dalek")
	}
	checkOnCurve(t, &p)
}

func TestVarTimeDoubleBaseMultVsDalek(t *testing.T) {
	var p Point
	var z Scalar
	p.VarTimeDoubleScalarBaseMult(dalekScalar, B, &z)
	if dalekScalarBasepoint.Equal(&p) != 1 {
		t.Error("VarTimeDoubleScalarBaseMult fails with b=0")
	}
	checkOnCurve(t, &p)
	p.VarTimeDoubleScalarBaseMult(&z, B, dalekScalar)
	if dalekScalarBasepoint.Equal(&p) != 1 {
		t.Error("VarTimeDoubleScalarBaseMult fails with a=0")
	}
	checkOnCurve(t, &p)
}

func TestScalarMultDistributesOverAdd(t *testing.T) {
	scalarMultDistributesOverAdd := func(x, y Scalar) bool {
		var z Scalar
		z.Add(&x, &y)
		var p, q, r, check Point
		p.ScalarMult(&x, B)
		q.ScalarMult(&y, B)
		r.ScalarMult(&z, B)
		check.Add(&p, &q)
		checkOnCurve(t, &p, &q, &r, &check)
		return check.Equal(&r) == 1
	}

	if err := quick.Check(scalarMultDistributesOverAdd, quickCheckConfig(32)); err != nil {
		t.Error(err)
	}
}

func TestScalarMultNonIdentityPoint(t *testing.T) {
	// Check whether p.ScalarMult and q.ScalaBaseMult give the same,
	// when p and q are originally set to the base point.

	scalarMultNonIdentityPoint := func(x Scalar) bool {
		var p, q Point
		p.Set(B)
		q.Set(B)

		p.ScalarMult(&x, B)
		q.ScalarBaseMult(&x)

		checkOnCurve(t, &p, &q)

		return p.Equal(&q) == 1
	}

	if err := quick.Check(scalarMultNonIdentityPoint, quickCheckConfig(32)); err != nil {
		t.Error(err)
	}
}

func TestBasepointTableGeneration(t *testing.T) {
	// The basepoint table is 32 affineLookupTables,
	// corresponding to (16^2i)*B for table i.
	basepointTable := basepointTable()

	tmp1 := &projP1xP1{}
	tmp2 := &projP2{}
	tmp3 := &Point{}
	tmp3.Set(B)
	table := make([]affineLookupTable, 32)
	for i := 0; i < 32; i++ {
		// Build the table
		table[i].FromP3(tmp3)
		// Assert equality with the hardcoded one
		if table[i] != basepointTable[i] {
			t.Errorf("Basepoint table %d does not match", i)
		}

		// Set p = (16^2)*p = 256*p = 2^8*p
		tmp2.FromP3(tmp3)
		for j := 0; j < 7; j++ {
			tmp1.Double(tmp2)
			tmp2.FromP1xP1(tmp1)
		}
		tmp1.Double(tmp2)
		tmp3.fromP1xP1(tmp1)
		checkOnCurve(t, tmp3)
	}
}

func TestScalarMultMatchesBaseMult(t *testing.T) {
	scalarMultMatchesBaseMult := func(x Scalar) bool {
		var p, q Point
		p.ScalarMult(&x, B)
		q.ScalarBaseMult(&x)
		checkOnCurve(t, &p, &q)
		return p.Equal(&q) == 1
	}

	if err := quick.Check(scalarMultMatchesBaseMult, quickCheckConfig(32)); err != nil {
		t.Error(err)
	}
}

func TestBasepointNafTableGeneration(t *testing.T) {
	var table nafLookupTable8
	table.FromP3(B)

	if table != *basepointNafTable() {
		t.Error("BasepointNafTable does not match")
	}
}

func TestVarTimeDoubleBaseMultMatchesBaseMult(t *testing.T) {
	varTimeDoubleBaseMultMatchesBaseMult := func(x, y Scalar) bool {
		var p, q1, q2, check Point

		p.VarTimeDoubleScalarBaseMult(&x, B, &y)

		q1.ScalarBaseMult(&x)
		q2.ScalarBaseMult(&y)
		check.Add(&q1, &q2)

		checkOnCurve(t, &p, &check, &q1, &q2)
		return p.Equal(&check) == 1
	}

	if err := quick.Check(varTimeDoubleBaseMultMatchesBaseMult, quickCheckConfig(32)); err != nil {
		t.Error(err)
	}
}

// Benchmarks.

func BenchmarkScalarBaseMult(b *testing.B) {
	var p Point

	for i := 0; i < b.N; i++ {
		p.ScalarBaseMult(dalekScalar)
	}
}

func BenchmarkScalarMult(b *testing.B) {
	var p Point

	for i := 0; i < b.N; i++ {
		p.ScalarMult(dalekScalar, B)
	}
}

func BenchmarkVarTimeDoubleScalarBaseMult(b *testing.B) {
	var p Point

	for i := 0; i < b.N; i++ {
		p.VarTimeDoubleScalarBaseMult(dalekScalar, B, dalekScalar)
	}
}
