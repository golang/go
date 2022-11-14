// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"math/big"
	"testing"
)

type scalarMultTest struct {
	k          string
	xIn, yIn   string
	xOut, yOut string
}

var p256MultTests = []scalarMultTest{
	{
		"2a265f8bcbdcaf94d58519141e578124cb40d64a501fba9c11847b28965bc737",
		"023819813ac969847059028ea88a1f30dfbcde03fc791d3a252c6b41211882ea",
		"f93e4ae433cc12cf2a43fc0ef26400c0e125508224cdb649380f25479148a4ad",
		"4d4de80f1534850d261075997e3049321a0864082d24a917863366c0724f5ae3",
		"a22d2b7f7818a3563e0f7a76c9bf0921ac55e06e2e4d11795b233824b1db8cc0",
	},
	{
		"313f72ff9fe811bf573176231b286a3bdb6f1b14e05c40146590727a71c3bccd",
		"cc11887b2d66cbae8f4d306627192522932146b42f01d3c6f92bd5c8ba739b06",
		"a2f08a029cd06b46183085bae9248b0ed15b70280c7ef13a457f5af382426031",
		"831c3f6b5f762d2f461901577af41354ac5f228c2591f84f8a6e51e2e3f17991",
		"93f90934cd0ef2c698cc471c60a93524e87ab31ca2412252337f364513e43684",
	},
}

func TestP256BaseMult(t *testing.T) {
	p256 := P256()
	p256Generic := genericParamsForCurve(p256)

	scalars := make([]*big.Int, 0, len(p224BaseMultTests)+1)
	for _, e := range p224BaseMultTests {
		k, _ := new(big.Int).SetString(e.k, 10)
		scalars = append(scalars, k)
	}
	k := new(big.Int).SetInt64(1)
	k.Lsh(k, 500)
	scalars = append(scalars, k)

	for i, k := range scalars {
		x, y := p256.ScalarBaseMult(k.Bytes())
		x2, y2 := p256Generic.ScalarBaseMult(k.Bytes())
		if x.Cmp(x2) != 0 || y.Cmp(y2) != 0 {
			t.Errorf("#%d: got (%x, %x), want (%x, %x)", i, x, y, x2, y2)
		}

		if testing.Short() && i > 5 {
			break
		}
	}
}

func TestP256Mult(t *testing.T) {
	p256 := P256()
	for i, e := range p256MultTests {
		x, _ := new(big.Int).SetString(e.xIn, 16)
		y, _ := new(big.Int).SetString(e.yIn, 16)
		k, _ := new(big.Int).SetString(e.k, 16)
		expectedX, _ := new(big.Int).SetString(e.xOut, 16)
		expectedY, _ := new(big.Int).SetString(e.yOut, 16)

		xx, yy := p256.ScalarMult(x, y, k.Bytes())
		if xx.Cmp(expectedX) != 0 || yy.Cmp(expectedY) != 0 {
			t.Errorf("#%d: got (%x, %x), want (%x, %x)", i, xx, yy, expectedX, expectedY)
		}
	}
}

type synthCombinedMult struct {
	Curve
}

func (s synthCombinedMult) CombinedMult(bigX, bigY *big.Int, baseScalar, scalar []byte) (x, y *big.Int) {
	x1, y1 := s.ScalarBaseMult(baseScalar)
	x2, y2 := s.ScalarMult(bigX, bigY, scalar)
	return s.Add(x1, y1, x2, y2)
}

func TestP256CombinedMult(t *testing.T) {
	type combinedMult interface {
		Curve
		CombinedMult(bigX, bigY *big.Int, baseScalar, scalar []byte) (x, y *big.Int)
	}

	p256, ok := P256().(combinedMult)
	if !ok {
		p256 = &synthCombinedMult{P256()}
	}

	gx := p256.Params().Gx
	gy := p256.Params().Gy

	zero := make([]byte, 32)
	one := make([]byte, 32)
	one[31] = 1
	two := make([]byte, 32)
	two[31] = 2

	// 0×G + 0×G = ∞
	x, y := p256.CombinedMult(gx, gy, zero, zero)
	if x.Sign() != 0 || y.Sign() != 0 {
		t.Errorf("0×G + 0×G = (%d, %d), should be ∞", x, y)
	}

	// 1×G + 0×G = G
	x, y = p256.CombinedMult(gx, gy, one, zero)
	if x.Cmp(gx) != 0 || y.Cmp(gy) != 0 {
		t.Errorf("1×G + 0×G = (%d, %d), should be (%d, %d)", x, y, gx, gy)
	}

	// 0×G + 1×G = G
	x, y = p256.CombinedMult(gx, gy, zero, one)
	if x.Cmp(gx) != 0 || y.Cmp(gy) != 0 {
		t.Errorf("0×G + 1×G = (%d, %d), should be (%d, %d)", x, y, gx, gy)
	}

	// 1×G + 1×G = 2×G
	x, y = p256.CombinedMult(gx, gy, one, one)
	ggx, ggy := p256.ScalarBaseMult(two)
	if x.Cmp(ggx) != 0 || y.Cmp(ggy) != 0 {
		t.Errorf("1×G + 1×G = (%d, %d), should be (%d, %d)", x, y, ggx, ggy)
	}

	minusOne := new(big.Int).Sub(p256.Params().N, big.NewInt(1))
	// 1×G + (-1)×G = ∞
	x, y = p256.CombinedMult(gx, gy, one, minusOne.Bytes())
	if x.Sign() != 0 || y.Sign() != 0 {
		t.Errorf("1×G + (-1)×G = (%d, %d), should be ∞", x, y)
	}
}

func TestIssue52075(t *testing.T) {
	Gx, Gy := P256().Params().Gx, P256().Params().Gy
	scalar := make([]byte, 33)
	scalar[32] = 1
	x, y := P256().ScalarBaseMult(scalar)
	if x.Cmp(Gx) != 0 || y.Cmp(Gy) != 0 {
		t.Errorf("unexpected output (%v,%v)", x, y)
	}
	x, y = P256().ScalarMult(Gx, Gy, scalar)
	if x.Cmp(Gx) != 0 || y.Cmp(Gy) != 0 {
		t.Errorf("unexpected output (%v,%v)", x, y)
	}
}
