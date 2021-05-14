// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"math/big"
	"testing"
)

// genericParamsForCurve returns the dereferenced CurveParams for
// the specified curve. This is used to avoid the logic for
// upgrading a curve to it's specific implementation, forcing
// usage of the generic implementation. This is only relevant
// for the P224, P256, and P521 curves.
func genericParamsForCurve(c Curve) *CurveParams {
	d := *(c.Params())
	return &d
}

func testAllCurves(t *testing.T, f func(*testing.T, Curve)) {
	tests := []struct {
		name  string
		curve Curve
	}{
		{"P256", P256()},
		{"P256/Params", genericParamsForCurve(P256())},
		{"P224", P224()},
		{"P224/Params", genericParamsForCurve(P224())},
		{"P384", P384()},
		{"P384/Params", genericParamsForCurve(P384())},
		{"P521", P521()},
		{"P521/Params", genericParamsForCurve(P521())},
	}
	if testing.Short() {
		tests = tests[:1]
	}
	for _, test := range tests {
		curve := test.curve
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			f(t, curve)
		})
	}
}

func TestOnCurve(t *testing.T) {
	testAllCurves(t, func(t *testing.T, curve Curve) {
		if !curve.IsOnCurve(curve.Params().Gx, curve.Params().Gy) {
			t.Error("basepoint is not on the curve")
		}
	})
}

func TestOffCurve(t *testing.T) {
	testAllCurves(t, func(t *testing.T, curve Curve) {
		x, y := new(big.Int).SetInt64(1), new(big.Int).SetInt64(1)
		if curve.IsOnCurve(x, y) {
			t.Errorf("point off curve is claimed to be on the curve")
		}
		b := Marshal(curve, x, y)
		x1, y1 := Unmarshal(curve, b)
		if x1 != nil || y1 != nil {
			t.Errorf("unmarshaling a point not on the curve succeeded")
		}
	})
}

func TestInfinity(t *testing.T) {
	testAllCurves(t, testInfinity)
}

func testInfinity(t *testing.T, curve Curve) {
	_, x, y, _ := GenerateKey(curve, rand.Reader)
	x, y = curve.ScalarMult(x, y, curve.Params().N.Bytes())
	if x.Sign() != 0 || y.Sign() != 0 {
		t.Errorf("x^q != ∞")
	}

	x, y = curve.ScalarBaseMult([]byte{0})
	if x.Sign() != 0 || y.Sign() != 0 {
		t.Errorf("b^0 != ∞")
		x.SetInt64(0)
		y.SetInt64(0)
	}

	x2, y2 := curve.Double(x, y)
	if x2.Sign() != 0 || y2.Sign() != 0 {
		t.Errorf("2∞ != ∞")
	}

	baseX := curve.Params().Gx
	baseY := curve.Params().Gy

	x3, y3 := curve.Add(baseX, baseY, x, y)
	if x3.Cmp(baseX) != 0 || y3.Cmp(baseY) != 0 {
		t.Errorf("x+∞ != x")
	}

	x4, y4 := curve.Add(x, y, baseX, baseY)
	if x4.Cmp(baseX) != 0 || y4.Cmp(baseY) != 0 {
		t.Errorf("∞+x != x")
	}

	if curve.IsOnCurve(x, y) {
		t.Errorf("IsOnCurve(∞) == true")
	}
}

func TestMarshal(t *testing.T) {
	testAllCurves(t, func(t *testing.T, curve Curve) {
		_, x, y, err := GenerateKey(curve, rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		serialized := Marshal(curve, x, y)
		xx, yy := Unmarshal(curve, serialized)
		if xx == nil {
			t.Fatal("failed to unmarshal")
		}
		if xx.Cmp(x) != 0 || yy.Cmp(y) != 0 {
			t.Fatal("unmarshal returned different values")
		}
	})
}

func TestUnmarshalToLargeCoordinates(t *testing.T) {
	// See https://golang.org/issues/20482.
	testAllCurves(t, testUnmarshalToLargeCoordinates)
}

func testUnmarshalToLargeCoordinates(t *testing.T, curve Curve) {
	p := curve.Params().P
	byteLen := (p.BitLen() + 7) / 8

	// Set x to be greater than curve's parameter P – specifically, to P+5.
	// Set y to mod_sqrt(x^3 - 3x + B)) so that (x mod P = 5 , y) is on the
	// curve.
	x := new(big.Int).Add(p, big.NewInt(5))
	y := curve.Params().polynomial(x)
	y.ModSqrt(y, p)

	invalid := make([]byte, byteLen*2+1)
	invalid[0] = 4 // uncompressed encoding
	x.FillBytes(invalid[1 : 1+byteLen])
	y.FillBytes(invalid[1+byteLen:])

	if X, Y := Unmarshal(curve, invalid); X != nil || Y != nil {
		t.Errorf("Unmarshal accepts invalid X coordinate")
	}

	if curve == p256 {
		// This is a point on the curve with a small y value, small enough that
		// we can add p and still be within 32 bytes.
		x, _ = new(big.Int).SetString("31931927535157963707678568152204072984517581467226068221761862915403492091210", 10)
		y, _ = new(big.Int).SetString("5208467867388784005506817585327037698770365050895731383201516607147", 10)
		y.Add(y, p)

		if p.Cmp(y) > 0 || y.BitLen() != 256 {
			t.Fatal("y not within expected range")
		}

		// marshal
		x.FillBytes(invalid[1 : 1+byteLen])
		y.FillBytes(invalid[1+byteLen:])

		if X, Y := Unmarshal(curve, invalid); X != nil || Y != nil {
			t.Errorf("Unmarshal accepts invalid Y coordinate")
		}
	}
}

func TestMarshalCompressed(t *testing.T) {
	t.Run("P-256/03", func(t *testing.T) {
		data, _ := hex.DecodeString("031e3987d9f9ea9d7dd7155a56a86b2009e1e0ab332f962d10d8beb6406ab1ad79")
		x, _ := new(big.Int).SetString("13671033352574878777044637384712060483119675368076128232297328793087057702265", 10)
		y, _ := new(big.Int).SetString("66200849279091436748794323380043701364391950689352563629885086590854940586447", 10)
		testMarshalCompressed(t, P256(), x, y, data)
	})
	t.Run("P-256/02", func(t *testing.T) {
		data, _ := hex.DecodeString("021e3987d9f9ea9d7dd7155a56a86b2009e1e0ab332f962d10d8beb6406ab1ad79")
		x, _ := new(big.Int).SetString("13671033352574878777044637384712060483119675368076128232297328793087057702265", 10)
		y, _ := new(big.Int).SetString("49591239931264812013903123569363872165694192725937750565648544718012157267504", 10)
		testMarshalCompressed(t, P256(), x, y, data)
	})

	t.Run("Invalid", func(t *testing.T) {
		data, _ := hex.DecodeString("02fd4bf61763b46581fd9174d623516cf3c81edd40e29ffa2777fb6cb0ae3ce535")
		X, Y := UnmarshalCompressed(P256(), data)
		if X != nil || Y != nil {
			t.Error("expected an error for invalid encoding")
		}
	})

	if testing.Short() {
		t.Skip("skipping other curves on short test")
	}

	testAllCurves(t, func(t *testing.T, curve Curve) {
		_, x, y, err := GenerateKey(curve, rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		testMarshalCompressed(t, curve, x, y, nil)
	})

}

func testMarshalCompressed(t *testing.T, curve Curve, x, y *big.Int, want []byte) {
	if !curve.IsOnCurve(x, y) {
		t.Fatal("invalid test point")
	}
	got := MarshalCompressed(curve, x, y)
	if want != nil && !bytes.Equal(got, want) {
		t.Errorf("got unexpected MarshalCompressed result: got %x, want %x", got, want)
	}

	X, Y := UnmarshalCompressed(curve, got)
	if X == nil || Y == nil {
		t.Fatalf("UnmarshalCompressed failed unexpectedly")
	}

	if !curve.IsOnCurve(X, Y) {
		t.Error("UnmarshalCompressed returned a point not on the curve")
	}
	if X.Cmp(x) != 0 || Y.Cmp(y) != 0 {
		t.Errorf("point did not round-trip correctly: got (%v, %v), want (%v, %v)", X, Y, x, y)
	}
}

func benchmarkAllCurves(t *testing.B, f func(*testing.B, Curve)) {
	tests := []struct {
		name  string
		curve Curve
	}{
		{"P256", P256()},
		{"P224", P224()},
		{"P384", P384()},
		{"P521", P521()},
	}
	for _, test := range tests {
		curve := test.curve
		t.Run(test.name, func(t *testing.B) {
			f(t, curve)
		})
	}
}

func BenchmarkScalarBaseMult(b *testing.B) {
	benchmarkAllCurves(b, func(b *testing.B, curve Curve) {
		priv, _, _, _ := GenerateKey(curve, rand.Reader)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			x, _ := curve.ScalarBaseMult(priv)
			// Prevent the compiler from optimizing out the operation.
			priv[0] ^= byte(x.Bits()[0])
		}
	})
}

func BenchmarkScalarMult(b *testing.B) {
	benchmarkAllCurves(b, func(b *testing.B, curve Curve) {
		_, x, y, _ := GenerateKey(curve, rand.Reader)
		priv, _, _, _ := GenerateKey(curve, rand.Reader)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			x, y = curve.ScalarMult(x, y, priv)
		}
	})
}
