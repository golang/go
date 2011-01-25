// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dsa

import (
	"big"
	"crypto/rand"
	"testing"
)

func testSignAndVerify(t *testing.T, i int, priv *PrivateKey) {
	hashed := []byte("testing")
	r, s, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("%d: error signing: %s", i, err)
		return
	}

	if !Verify(&priv.PublicKey, hashed, r, s) {
		t.Errorf("%d: Verify failed", i)
	}
}

func testParameterGeneration(t *testing.T, sizes ParameterSizes, L, N int) {
	var priv PrivateKey
	params := &priv.Parameters

	err := GenerateParameters(params, rand.Reader, sizes)
	if err != nil {
		t.Errorf("%d: %s", int(sizes), err)
		return
	}

	if params.P.BitLen() != L {
		t.Errorf("%d: params.BitLen got:%d want:%d", int(sizes), params.P.BitLen(), L)
	}

	if params.Q.BitLen() != N {
		t.Errorf("%d: q.BitLen got:%d want:%d", int(sizes), params.Q.BitLen(), L)
	}

	one := new(big.Int)
	one.SetInt64(1)
	pm1 := new(big.Int).Sub(params.P, one)
	quo, rem := new(big.Int).DivMod(pm1, params.Q, new(big.Int))
	if rem.Sign() != 0 {
		t.Errorf("%d: p-1 mod q != 0", int(sizes))
	}
	x := new(big.Int).Exp(params.G, quo, params.P)
	if x.Cmp(one) == 0 {
		t.Errorf("%d: invalid generator", int(sizes))
	}

	err = GenerateKey(&priv, rand.Reader)
	if err != nil {
		t.Errorf("error generating key: %s", err)
		return
	}

	testSignAndVerify(t, int(sizes), &priv)
}

func TestParameterGeneration(t *testing.T) {
	// This test is too slow to run all the time.
	return

	testParameterGeneration(t, L1024N160, 1024, 160)
	testParameterGeneration(t, L2048N224, 2048, 224)
	testParameterGeneration(t, L2048N256, 2048, 256)
	testParameterGeneration(t, L3072N256, 3072, 256)
}

func TestSignAndVerify(t *testing.T) {
	var priv PrivateKey
	priv.P, _ = new(big.Int).SetString("A9B5B793FB4785793D246BAE77E8FF63CA52F442DA763C440259919FE1BC1D6065A9350637A04F75A2F039401D49F08E066C4D275A5A65DA5684BC563C14289D7AB8A67163BFBF79D85972619AD2CFF55AB0EE77A9002B0EF96293BDD0F42685EBB2C66C327079F6C98000FBCB79AACDE1BC6F9D5C7B1A97E3D9D54ED7951FEF", 16)
	priv.Q, _ = new(big.Int).SetString("E1D3391245933D68A0714ED34BBCB7A1F422B9C1", 16)
	priv.G, _ = new(big.Int).SetString("634364FC25248933D01D1993ECABD0657CC0CB2CEED7ED2E3E8AECDFCDC4A25C3B15E9E3B163ACA2984B5539181F3EFF1A5E8903D71D5B95DA4F27202B77D2C44B430BB53741A8D59A8F86887525C9F2A6A5980A195EAA7F2FF910064301DEF89D3AA213E1FAC7768D89365318E370AF54A112EFBA9246D9158386BA1B4EEFDA", 16)
	priv.Y, _ = new(big.Int).SetString("32969E5780CFE1C849A1C276D7AEB4F38A23B591739AA2FE197349AEEBD31366AEE5EB7E6C6DDB7C57D02432B30DB5AA66D9884299FAA72568944E4EEDC92EA3FBC6F39F53412FBCC563208F7C15B737AC8910DBC2D9C9B8C001E72FDC40EB694AB1F06A5A2DBD18D9E36C66F31F566742F11EC0A52E9F7B89355C02FB5D32D2", 16)
	priv.X, _ = new(big.Int).SetString("5078D4D29795CBE76D3AACFE48C9AF0BCDBEE91A", 16)

	testSignAndVerify(t, 0, &priv)
}
