// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"bufio"
	"bytes"
	"compress/bzip2"
	"crypto/elliptic"
	"crypto/internal/bigmod"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/hex"
	"hash"
	"io"
	"math/big"
	"os"
	"strings"
	"testing"
)

func testAllCurves(t *testing.T, f func(*testing.T, elliptic.Curve)) {
	tests := []struct {
		name  string
		curve elliptic.Curve
	}{
		{"P256", elliptic.P256()},
		{"P224", elliptic.P224()},
		{"P384", elliptic.P384()},
		{"P521", elliptic.P521()},
		{"P256/Generic", genericParamsForCurve(elliptic.P256())},
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

// genericParamsForCurve returns the dereferenced CurveParams for
// the specified curve. This is used to avoid the logic for
// upgrading a curve to its specific implementation, forcing
// usage of the generic implementation.
func genericParamsForCurve(c elliptic.Curve) *elliptic.CurveParams {
	d := *(c.Params())
	return &d
}

func TestKeyGeneration(t *testing.T) {
	testAllCurves(t, testKeyGeneration)
}

func testKeyGeneration(t *testing.T, c elliptic.Curve) {
	priv, err := GenerateKey(c, rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	if !c.IsOnCurve(priv.PublicKey.X, priv.PublicKey.Y) {
		t.Errorf("public key invalid: %s", err)
	}
}

func TestSignAndVerify(t *testing.T) {
	testAllCurves(t, testSignAndVerify)
}

func testSignAndVerify(t *testing.T, c elliptic.Curve) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	r, s, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("error signing: %s", err)
		return
	}

	if !Verify(&priv.PublicKey, hashed, r, s) {
		t.Errorf("Verify failed")
	}

	hashed[0] ^= 0xff
	if Verify(&priv.PublicKey, hashed, r, s) {
		t.Errorf("Verify always works!")
	}
}

func TestSignAndVerifyASN1(t *testing.T) {
	testAllCurves(t, testSignAndVerifyASN1)
}

func testSignAndVerifyASN1(t *testing.T, c elliptic.Curve) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	sig, err := SignASN1(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("error signing: %s", err)
		return
	}

	if !VerifyASN1(&priv.PublicKey, hashed, sig) {
		t.Errorf("VerifyASN1 failed")
	}

	hashed[0] ^= 0xff
	if VerifyASN1(&priv.PublicKey, hashed, sig) {
		t.Errorf("VerifyASN1 always works!")
	}
}

func TestNonceSafety(t *testing.T) {
	testAllCurves(t, testNonceSafety)
}

func testNonceSafety(t *testing.T, c elliptic.Curve) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	r0, s0, err := Sign(zeroReader, priv, hashed)
	if err != nil {
		t.Errorf("error signing: %s", err)
		return
	}

	hashed = []byte("testing...")
	r1, s1, err := Sign(zeroReader, priv, hashed)
	if err != nil {
		t.Errorf("error signing: %s", err)
		return
	}

	if s0.Cmp(s1) == 0 {
		// This should never happen.
		t.Errorf("the signatures on two different messages were the same")
	}

	if r0.Cmp(r1) == 0 {
		t.Errorf("the nonce used for two different messages was the same")
	}
}

func TestINDCCA(t *testing.T) {
	testAllCurves(t, testINDCCA)
}

func testINDCCA(t *testing.T, c elliptic.Curve) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	r0, s0, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("error signing: %s", err)
		return
	}

	r1, s1, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("error signing: %s", err)
		return
	}

	if s0.Cmp(s1) == 0 {
		t.Errorf("two signatures of the same message produced the same result")
	}

	if r0.Cmp(r1) == 0 {
		t.Errorf("two signatures of the same message produced the same nonce")
	}
}

func fromHex(s string) *big.Int {
	r, ok := new(big.Int).SetString(s, 16)
	if !ok {
		panic("bad hex")
	}
	return r
}

func TestVectors(t *testing.T) {
	// This test runs the full set of NIST test vectors from
	// https://csrc.nist.gov/groups/STM/cavp/documents/dss/186-3ecdsatestvectors.zip
	//
	// The SigVer.rsp file has been edited to remove test vectors for
	// unsupported algorithms and has been compressed.

	if testing.Short() {
		return
	}

	f, err := os.Open("testdata/SigVer.rsp.bz2")
	if err != nil {
		t.Fatal(err)
	}

	buf := bufio.NewReader(bzip2.NewReader(f))

	lineNo := 1
	var h hash.Hash
	var msg []byte
	var hashed []byte
	var r, s *big.Int
	pub := new(PublicKey)

	for {
		line, err := buf.ReadString('\n')
		if len(line) == 0 {
			if err == io.EOF {
				break
			}
			t.Fatalf("error reading from input: %s", err)
		}
		lineNo++
		// Need to remove \r\n from the end of the line.
		if !strings.HasSuffix(line, "\r\n") {
			t.Fatalf("bad line ending (expected \\r\\n) on line %d", lineNo)
		}
		line = line[:len(line)-2]

		if len(line) == 0 || line[0] == '#' {
			continue
		}

		if line[0] == '[' {
			line = line[1 : len(line)-1]
			curve, hash, _ := strings.Cut(line, ",")

			switch curve {
			case "P-224":
				pub.Curve = elliptic.P224()
			case "P-256":
				pub.Curve = elliptic.P256()
			case "P-384":
				pub.Curve = elliptic.P384()
			case "P-521":
				pub.Curve = elliptic.P521()
			default:
				pub.Curve = nil
			}

			switch hash {
			case "SHA-1":
				h = sha1.New()
			case "SHA-224":
				h = sha256.New224()
			case "SHA-256":
				h = sha256.New()
			case "SHA-384":
				h = sha512.New384()
			case "SHA-512":
				h = sha512.New()
			default:
				h = nil
			}

			continue
		}

		if h == nil || pub.Curve == nil {
			continue
		}

		switch {
		case strings.HasPrefix(line, "Msg = "):
			if msg, err = hex.DecodeString(line[6:]); err != nil {
				t.Fatalf("failed to decode message on line %d: %s", lineNo, err)
			}
		case strings.HasPrefix(line, "Qx = "):
			pub.X = fromHex(line[5:])
		case strings.HasPrefix(line, "Qy = "):
			pub.Y = fromHex(line[5:])
		case strings.HasPrefix(line, "R = "):
			r = fromHex(line[4:])
		case strings.HasPrefix(line, "S = "):
			s = fromHex(line[4:])
		case strings.HasPrefix(line, "Result = "):
			expected := line[9] == 'P'
			h.Reset()
			h.Write(msg)
			hashed := h.Sum(hashed[:0])
			if Verify(pub, hashed, r, s) != expected {
				t.Fatalf("incorrect result on line %d", lineNo)
			}
		default:
			t.Fatalf("unknown variable on line %d: %s", lineNo, line)
		}
	}
}

func TestNegativeInputs(t *testing.T) {
	testAllCurves(t, testNegativeInputs)
}

func testNegativeInputs(t *testing.T, curve elliptic.Curve) {
	key, err := GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Errorf("failed to generate key")
	}

	var hash [32]byte
	r := new(big.Int).SetInt64(1)
	r.Lsh(r, 550 /* larger than any supported curve */)
	r.Neg(r)

	if Verify(&key.PublicKey, hash[:], r, r) {
		t.Errorf("bogus signature accepted")
	}
}

func TestZeroHashSignature(t *testing.T) {
	testAllCurves(t, testZeroHashSignature)
}

func testZeroHashSignature(t *testing.T, curve elliptic.Curve) {
	zeroHash := make([]byte, 64)

	privKey, err := GenerateKey(curve, rand.Reader)
	if err != nil {
		panic(err)
	}

	// Sign a hash consisting of all zeros.
	r, s, err := Sign(rand.Reader, privKey, zeroHash)
	if err != nil {
		panic(err)
	}

	// Confirm that it can be verified.
	if !Verify(&privKey.PublicKey, zeroHash, r, s) {
		t.Errorf("zero hash signature verify failed for %T", curve)
	}
}

func TestRandomPoint(t *testing.T) {
	t.Run("P-224", func(t *testing.T) { testRandomPoint(t, p224()) })
	t.Run("P-256", func(t *testing.T) { testRandomPoint(t, p256()) })
	t.Run("P-384", func(t *testing.T) { testRandomPoint(t, p384()) })
	t.Run("P-521", func(t *testing.T) { testRandomPoint(t, p521()) })
}

func testRandomPoint[Point nistPoint[Point]](t *testing.T, c *nistCurve[Point]) {
	t.Cleanup(func() { testingOnlyRejectionSamplingLooped = nil })
	var loopCount int
	testingOnlyRejectionSamplingLooped = func() { loopCount++ }

	// A sequence of all ones will generate 2^N-1, which should be rejected.
	// (Unless, for example, we are masking too many bits.)
	r := io.MultiReader(bytes.NewReader(bytes.Repeat([]byte{0xff}, 100)), rand.Reader)
	if k, p, err := randomPoint(c, r); err != nil {
		t.Fatal(err)
	} else if k.IsZero() == 1 {
		t.Error("k is zero")
	} else if p.Bytes()[0] != 4 {
		t.Error("p is infinity")
	}
	if loopCount == 0 {
		t.Error("overflow was not rejected")
	}
	loopCount = 0

	// A sequence of all zeroes will generate zero, which should be rejected.
	r = io.MultiReader(bytes.NewReader(bytes.Repeat([]byte{0}, 100)), rand.Reader)
	if k, p, err := randomPoint(c, r); err != nil {
		t.Fatal(err)
	} else if k.IsZero() == 1 {
		t.Error("k is zero")
	} else if p.Bytes()[0] != 4 {
		t.Error("p is infinity")
	}
	if loopCount == 0 {
		t.Error("zero was not rejected")
	}
	loopCount = 0

	// P-256 has a 2⁻³² chance or randomly hitting a rejection. For P-224 it's
	// 2⁻¹¹², for P-384 it's 2⁻¹⁹⁴, and for P-521 it's 2⁻²⁶², so if we hit in
	// tests, something is horribly wrong. (For example, we are masking the
	// wrong bits.)
	if c.curve == elliptic.P256() {
		return
	}
	if k, p, err := randomPoint(c, rand.Reader); err != nil {
		t.Fatal(err)
	} else if k.IsZero() == 1 {
		t.Error("k is zero")
	} else if p.Bytes()[0] != 4 {
		t.Error("p is infinity")
	}
	if loopCount > 0 {
		t.Error("unexpected rejection")
	}
}

func TestHashToNat(t *testing.T) {
	t.Run("P-224", func(t *testing.T) { testHashToNat(t, p224()) })
	t.Run("P-256", func(t *testing.T) { testHashToNat(t, p256()) })
	t.Run("P-384", func(t *testing.T) { testHashToNat(t, p384()) })
	t.Run("P-521", func(t *testing.T) { testHashToNat(t, p521()) })
}

func testHashToNat[Point nistPoint[Point]](t *testing.T, c *nistCurve[Point]) {
	for l := 0; l < 600; l++ {
		h := bytes.Repeat([]byte{0xff}, l)
		hashToNat(c, bigmod.NewNat(), h)
	}
}

func TestZeroSignature(t *testing.T) {
	testAllCurves(t, testZeroSignature)
}

func testZeroSignature(t *testing.T, curve elliptic.Curve) {
	privKey, err := GenerateKey(curve, rand.Reader)
	if err != nil {
		panic(err)
	}

	if Verify(&privKey.PublicKey, make([]byte, 64), big.NewInt(0), big.NewInt(0)) {
		t.Errorf("Verify with r,s=0 succeeded: %T", curve)
	}
}

func TestNegativeSignature(t *testing.T) {
	testAllCurves(t, testNegativeSignature)
}

func testNegativeSignature(t *testing.T, curve elliptic.Curve) {
	zeroHash := make([]byte, 64)

	privKey, err := GenerateKey(curve, rand.Reader)
	if err != nil {
		panic(err)
	}
	r, s, err := Sign(rand.Reader, privKey, zeroHash)
	if err != nil {
		panic(err)
	}

	r = r.Neg(r)
	if Verify(&privKey.PublicKey, zeroHash, r, s) {
		t.Errorf("Verify with r=-r succeeded: %T", curve)
	}
}

func TestRPlusNSignature(t *testing.T) {
	testAllCurves(t, testRPlusNSignature)
}

func testRPlusNSignature(t *testing.T, curve elliptic.Curve) {
	zeroHash := make([]byte, 64)

	privKey, err := GenerateKey(curve, rand.Reader)
	if err != nil {
		panic(err)
	}
	r, s, err := Sign(rand.Reader, privKey, zeroHash)
	if err != nil {
		panic(err)
	}

	r = r.Add(r, curve.Params().N)
	if Verify(&privKey.PublicKey, zeroHash, r, s) {
		t.Errorf("Verify with r=r+n succeeded: %T", curve)
	}
}

func TestRMinusNSignature(t *testing.T) {
	testAllCurves(t, testRMinusNSignature)
}

func testRMinusNSignature(t *testing.T, curve elliptic.Curve) {
	zeroHash := make([]byte, 64)

	privKey, err := GenerateKey(curve, rand.Reader)
	if err != nil {
		panic(err)
	}
	r, s, err := Sign(rand.Reader, privKey, zeroHash)
	if err != nil {
		panic(err)
	}

	r = r.Sub(r, curve.Params().N)
	if Verify(&privKey.PublicKey, zeroHash, r, s) {
		t.Errorf("Verify with r=r-n succeeded: %T", curve)
	}
}

func randomPointForCurve(curve elliptic.Curve, rand io.Reader) error {
	switch curve.Params() {
	case elliptic.P224().Params():
		_, _, err := randomPoint(p224(), rand)
		return err
	case elliptic.P256().Params():
		_, _, err := randomPoint(p256(), rand)
		return err
	case elliptic.P384().Params():
		_, _, err := randomPoint(p384(), rand)
		return err
	case elliptic.P521().Params():
		_, _, err := randomPoint(p521(), rand)
		return err
	default:
		panic("unknown curve")
	}
}

func benchmarkAllCurves(b *testing.B, f func(*testing.B, elliptic.Curve)) {
	tests := []struct {
		name  string
		curve elliptic.Curve
	}{
		{"P256", elliptic.P256()},
		{"P384", elliptic.P384()},
		{"P521", elliptic.P521()},
	}
	for _, test := range tests {
		curve := test.curve
		b.Run(test.name, func(b *testing.B) {
			f(b, curve)
		})
	}
}

func BenchmarkSign(b *testing.B) {
	benchmarkAllCurves(b, func(b *testing.B, curve elliptic.Curve) {
		r := bufio.NewReaderSize(rand.Reader, 1<<15)
		priv, err := GenerateKey(curve, r)
		if err != nil {
			b.Fatal(err)
		}
		hashed := []byte("testing")

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sig, err := SignASN1(r, priv, hashed)
			if err != nil {
				b.Fatal(err)
			}
			// Prevent the compiler from optimizing out the operation.
			hashed[0] = sig[0]
		}
	})
}

func BenchmarkVerify(b *testing.B) {
	benchmarkAllCurves(b, func(b *testing.B, curve elliptic.Curve) {
		r := bufio.NewReaderSize(rand.Reader, 1<<15)
		priv, err := GenerateKey(curve, r)
		if err != nil {
			b.Fatal(err)
		}
		hashed := []byte("testing")
		sig, err := SignASN1(r, priv, hashed)
		if err != nil {
			b.Fatal(err)
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !VerifyASN1(&priv.PublicKey, hashed, sig) {
				b.Fatal("verify failed")
			}
		}
	})
}

func BenchmarkGenerateKey(b *testing.B) {
	benchmarkAllCurves(b, func(b *testing.B, curve elliptic.Curve) {
		r := bufio.NewReaderSize(rand.Reader, 1<<15)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := GenerateKey(curve, r); err != nil {
				b.Fatal(err)
			}
		}
	})
}
