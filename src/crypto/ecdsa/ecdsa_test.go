// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"bufio"
	"compress/bzip2"
	"crypto/elliptic"
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

func testKeyGeneration(t *testing.T, c elliptic.Curve, tag string) {
	priv, err := GenerateKey(c, rand.Reader)
	if err != nil {
		t.Errorf("%s: error: %s", tag, err)
		return
	}
	if !c.IsOnCurve(priv.PublicKey.X, priv.PublicKey.Y) {
		t.Errorf("%s: public key invalid: %s", tag, err)
	}
}

func TestKeyGeneration(t *testing.T) {
	testKeyGeneration(t, elliptic.P224(), "p224")
	if testing.Short() {
		return
	}
	testKeyGeneration(t, elliptic.P256(), "p256")
	testKeyGeneration(t, elliptic.P384(), "p384")
	testKeyGeneration(t, elliptic.P521(), "p521")
	testKeyGeneration(t, elliptic.P256k1(), "p256k1")
}

func BenchmarkSignP256(b *testing.B) {
	b.ResetTimer()
	p256 := elliptic.P256()
	hashed := []byte("testing")
	priv, _ := GenerateKey(p256, rand.Reader)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _, _ = Sign(rand.Reader, priv, hashed)
		}
	})
}

func BenchmarkSignP384(b *testing.B) {
	b.ResetTimer()
	p384 := elliptic.P384()
	hashed := []byte("testing")
	priv, _ := GenerateKey(p384, rand.Reader)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _, _ = Sign(rand.Reader, priv, hashed)
		}
	})
}

func BenchmarkVerifyP256(b *testing.B) {
	b.ResetTimer()
	p256 := elliptic.P256()
	hashed := []byte("testing")
	priv, _ := GenerateKey(p256, rand.Reader)
	r, s, _ := Sign(rand.Reader, priv, hashed)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Verify(&priv.PublicKey, hashed, r, s)
		}
	})
}

func BenchmarkKeyGeneration(b *testing.B) {
	b.ResetTimer()
	p256 := elliptic.P256()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			GenerateKey(p256, rand.Reader)
		}
	})
}

func testSignAndVerify(t *testing.T, c elliptic.Curve, tag string) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	r, s, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("%s: error signing: %s", tag, err)
		return
	}

	if !Verify(&priv.PublicKey, hashed, r, s) {
		t.Errorf("%s: Verify failed", tag)
	}

	hashed[0] ^= 0xff
	if Verify(&priv.PublicKey, hashed, r, s) {
		t.Errorf("%s: Verify always works!", tag)
	}
}

func TestSignAndVerify(t *testing.T) {
	testSignAndVerify(t, elliptic.P224(), "p224")
	if testing.Short() {
		return
	}
	testSignAndVerify(t, elliptic.P256(), "p256")
	testSignAndVerify(t, elliptic.P384(), "p384")
	testSignAndVerify(t, elliptic.P521(), "p521")
	testSignAndVerify(t, elliptic.P256k1(), "p256k1")
}

func testSignAndVerifyASN1(t *testing.T, c elliptic.Curve, tag string) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	sig, err := SignASN1(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("%s: error signing: %s", tag, err)
		return
	}

	if !VerifyASN1(&priv.PublicKey, hashed, sig) {
		t.Errorf("%s: VerifyASN1 failed", tag)
	}

	hashed[0] ^= 0xff
	if VerifyASN1(&priv.PublicKey, hashed, sig) {
		t.Errorf("%s: VerifyASN1 always works!", tag)
	}
}

func TestSignAndVerifyASN1(t *testing.T) {
	testSignAndVerifyASN1(t, elliptic.P224(), "p224")
	if testing.Short() {
		return
	}
	testSignAndVerifyASN1(t, elliptic.P256(), "p256")
	testSignAndVerifyASN1(t, elliptic.P384(), "p384")
	testSignAndVerifyASN1(t, elliptic.P521(), "p521")
	testSignAndVerifyASN1(t, elliptic.P256k1(), "p256k1")
}

func testNonceSafety(t *testing.T, c elliptic.Curve, tag string) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	r0, s0, err := Sign(zeroReader, priv, hashed)
	if err != nil {
		t.Errorf("%s: error signing: %s", tag, err)
		return
	}

	hashed = []byte("testing...")
	r1, s1, err := Sign(zeroReader, priv, hashed)
	if err != nil {
		t.Errorf("%s: error signing: %s", tag, err)
		return
	}

	if s0.Cmp(s1) == 0 {
		// This should never happen.
		t.Errorf("%s: the signatures on two different messages were the same", tag)
	}

	if r0.Cmp(r1) == 0 {
		t.Errorf("%s: the nonce used for two different messages was the same", tag)
	}
}

func TestNonceSafety(t *testing.T) {
	testNonceSafety(t, elliptic.P224(), "p224")
	if testing.Short() {
		return
	}
	testNonceSafety(t, elliptic.P256(), "p256")
	testNonceSafety(t, elliptic.P384(), "p384")
	testNonceSafety(t, elliptic.P521(), "p521")
	testNonceSafety(t, elliptic.P256k1(), "p256k1")
}

func testINDCCA(t *testing.T, c elliptic.Curve, tag string) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	r0, s0, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("%s: error signing: %s", tag, err)
		return
	}

	r1, s1, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("%s: error signing: %s", tag, err)
		return
	}

	if s0.Cmp(s1) == 0 {
		t.Errorf("%s: two signatures of the same message produced the same result", tag)
	}

	if r0.Cmp(r1) == 0 {
		t.Errorf("%s: two signatures of the same message produced the same nonce", tag)
	}
}

func TestINDCCA(t *testing.T) {
	testINDCCA(t, elliptic.P224(), "p224")
	if testing.Short() {
		return
	}
	testINDCCA(t, elliptic.P256(), "p256")
	testINDCCA(t, elliptic.P384(), "p384")
	testINDCCA(t, elliptic.P521(), "p521")
	testINDCCA(t, elliptic.P256k1(), "p256k1")
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
			parts := strings.SplitN(line, ",", 2)

			switch parts[0] {
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

			switch parts[1] {
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

func testNegativeInputs(t *testing.T, curve elliptic.Curve, tag string) {
	key, err := GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Errorf("failed to generate key for %q", tag)
	}

	var hash [32]byte
	r := new(big.Int).SetInt64(1)
	r.Lsh(r, 550 /* larger than any supported curve */)
	r.Neg(r)

	if Verify(&key.PublicKey, hash[:], r, r) {
		t.Errorf("bogus signature accepted for %q", tag)
	}
}

func TestNegativeInputs(t *testing.T) {
	testNegativeInputs(t, elliptic.P224(), "p224")
	testNegativeInputs(t, elliptic.P256(), "p256")
	testNegativeInputs(t, elliptic.P384(), "p384")
	testNegativeInputs(t, elliptic.P521(), "p521")
	testNegativeInputs(t, elliptic.P256k1(), "p256k1")
}

func TestZeroHashSignature(t *testing.T) {
	zeroHash := make([]byte, 64)

	for _, curve := range []elliptic.Curve{
		elliptic.P224(),
		elliptic.P256(),
		elliptic.P384(),
		elliptic.P521(),
		elliptic.P256k1(),
	} {
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
}

func TestP256k1Signature(t *testing.T) {
	// the first-ever bitcoin transaction: Satoshi transferred 10BTC to Hal Finney in block 170
	// See https://www.blockchain.com/btc/tx/f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16
	// use it as a test case to verify P256-k1 signature

	// raw transaction bytes
	// See https://blockchain.info/rawtx/f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16?format=hex
	rawTx := "0100000001c997a5e56e104102fa209c6a852dd90660a20b2d9c352423edce25857fcd3704000000004847304402204e45e16932b8af514961a1d3a1a25fdf3f4f7732e9d624c6c61548ab5fb8cd410220181522ec8eca07de4860a4acdd12909d831cc56cbbac4622082221a8768d1d0901ffffffff0200ca9a3b00000000434104ae1a62fe09c5f51b13905f07f06b99a2f7159b2225f374cd378d71302fa28414e7aab37397f554a7df5f142c21c1b7303b8a0626f1baded5c72a704f7e6cd84cac00286bee0000000043410411db93e1dcdb8a016b49840f8c53bc1eb68a382e97b1482ecad7b148a6909a5cb2e0eaddfb84ccf9744464f82e160bfa9b8b64f9d4c03f999b8643f656b412a3ac00000000"

	// decode public key
	// 0x41 = 65, followed by 65-byte uncompressed public key
	Pkscript, _ := hex.DecodeString("410411db93e1dcdb8a016b49840f8c53bc1eb68a382e97b1482ecad7b148a6909a5cb2e0eaddfb84ccf9744464f82e160bfa9b8b64f9d4c03f999b8643f656b412a3ac")
	p256k1 := elliptic.P256k1()
	x, y := elliptic.Unmarshal(p256k1, Pkscript[1:1+int(Pkscript[0])])
	if x == nil || y == nil {
		t.Error("failed to unmarshal public key")
	}
	pubkey := &PublicKey{
		Curve: p256k1,
		X:     x,
		Y:     y,
	}

	// decode (r, s) from signature
	Sigscript := "4847304402204e45e16932b8af514961a1d3a1a25fdf3f4f7732e9d624c6c61548ab5fb8cd410220181522ec8eca07de4860a4acdd12909d831cc56cbbac4622082221a8768d1d0901"
	SigscriptBytes, _ := hex.DecodeString(Sigscript)
	// bitcoin signature is DER-encoded:
	// 0x48: size of signature script
	// 0x47: length of data to follow
	// 0x30: marker of start
	// 0x44: length of remaining signature
	// 0x02: marker for r value
	// 0x20: length of r value
	// followed by : 32-byte r value
	r := new(big.Int).SetBytes(SigscriptBytes[6:38])
	// 0x02: marker for s value
	// 0x20: length of s value
	// followed by : 32-byte s value
	s := new(big.Int).SetBytes(SigscriptBytes[40:SigscriptBytes[0]])
	// last byte of Sigscript = 0x01, meaning the hash type = SIGHASH_ALL
	sigtype := SigscriptBytes[len(SigscriptBytes)-1]

	// construct the transaction input to calculate its hash
	// See https://en.bitcoin.it/wiki/OP_CHECKSIG for a detailed explanation
	pos := strings.Index(rawTx, Sigscript)
	afterSig, _ := hex.DecodeString(rawTx[pos+len(Sigscript):])
	txToHash, _ := hex.DecodeString(rawTx[:pos])
	// insert public key script
	txToHash = append(txToHash, byte(len(Pkscript)))
	txToHash = append(txToHash, Pkscript...)
	txToHash = append(txToHash, afterSig...)
	// append sigtype as 4-byte
	txToHash = append(txToHash, []byte{sigtype, 0, 0, 0}...)
	// SHA256 twice to get the hash
	hash := sha256.Sum256(txToHash)
	hash = sha256.Sum256(hash[:])

	// verify the signature
	if !Verify(pubkey, hash[:], r, s) {
		t.Error("failed to verify P256-k1 signature")
	}
}
