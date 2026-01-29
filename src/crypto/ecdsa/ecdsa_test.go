// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"bufio"
	"bytes"
	"compress/bzip2"
	"crypto"
	"crypto/elliptic"
	"crypto/internal/cryptotest"
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
		cryptotest.TestAllImplementations(t, "ecdsa", func(t *testing.T) {
			t.Run(test.name, func(t *testing.T) {
				t.Parallel()
				f(t, curve)
			})
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

type readerFunc func([]byte) (int, error)

func (f readerFunc) Read(b []byte) (int, error) { return f(b) }

var zeroReader = readerFunc(func(b []byte) (int, error) {
	clear(b)
	return len(b), nil
})

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
	cryptotest.TestAllImplementations(t, "ecdsa", testVectors)
}

func testVectors(t *testing.T) {
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

func TestRFC6979(t *testing.T) {
	t.Run("P-224", func(t *testing.T) {
		testRFC6979(t, elliptic.P224(),
			"F220266E1105BFE3083E03EC7A3A654651F45E37167E88600BF257C1",
			"00CF08DA5AD719E42707FA431292DEA11244D64FC51610D94B130D6C",
			"EEAB6F3DEBE455E3DBF85416F7030CBD94F34F2D6F232C69F3C1385A",
			"sample",
			"61AA3DA010E8E8406C656BC477A7A7189895E7E840CDFE8FF42307BA",
			"BC814050DAB5D23770879494F9E0A680DC1AF7161991BDE692B10101")
		testRFC6979(t, elliptic.P224(),
			"F220266E1105BFE3083E03EC7A3A654651F45E37167E88600BF257C1",
			"00CF08DA5AD719E42707FA431292DEA11244D64FC51610D94B130D6C",
			"EEAB6F3DEBE455E3DBF85416F7030CBD94F34F2D6F232C69F3C1385A",
			"test",
			"AD04DDE87B84747A243A631EA47A1BA6D1FAA059149AD2440DE6FBA6",
			"178D49B1AE90E3D8B629BE3DB5683915F4E8C99FDF6E666CF37ADCFD")
	})
	t.Run("P-256", func(t *testing.T) {
		// This vector was bruteforced to find a message that causes the
		// generation of k to loop. It was checked against
		// github.com/codahale/rfc6979 (https://go.dev/play/p/FK5-fmKf7eK),
		// OpenSSL 3.2.0 (https://github.com/openssl/openssl/pull/23130),
		// and python-ecdsa:
		//
		//    ecdsa.keys.SigningKey.from_secret_exponent(
		//        0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721,
		//        ecdsa.curves.curve_by_name("NIST256p"), hashlib.sha256).sign_deterministic(
		//        b"wv[vnX", hashlib.sha256, lambda r, s, order: print(hex(r), hex(s)))
		//
		testRFC6979(t, elliptic.P256(),
			"C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721",
			"60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6",
			"7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299",
			"wv[vnX",
			"EFD9073B652E76DA1B5A019C0E4A2E3FA529B035A6ABB91EF67F0ED7A1F21234",
			"3DB4706C9D9F4A4FE13BB5E08EF0FAB53A57DBAB2061C83A35FA411C68D2BA33")

		// The remaining vectors are from RFC 6979.
		testRFC6979(t, elliptic.P256(),
			"C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721",
			"60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6",
			"7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299",
			"sample",
			"EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
			"F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8")
		testRFC6979(t, elliptic.P256(),
			"C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721",
			"60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6",
			"7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299",
			"test",
			"F1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367",
			"019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083")
	})
	t.Run("P-384", func(t *testing.T) {
		testRFC6979(t, elliptic.P384(),
			"6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D896D5724E4C70A825F872C9EA60D2EDF5",
			"EC3A4E415B4E19A4568618029F427FA5DA9A8BC4AE92E02E06AAE5286B300C64DEF8F0EA9055866064A254515480BC13",
			"8015D9B72D7D57244EA8EF9AC0C621896708A59367F9DFB9F54CA84B3F1C9DB1288B231C3AE0D4FE7344FD2533264720",
			"sample",
			"21B13D1E013C7FA1392D03C5F99AF8B30C570C6F98D4EA8E354B63A21D3DAA33BDE1E888E63355D92FA2B3C36D8FB2CD",
			"F3AA443FB107745BF4BD77CB3891674632068A10CA67E3D45DB2266FA7D1FEEBEFDC63ECCD1AC42EC0CB8668A4FA0AB0")
		testRFC6979(t, elliptic.P384(),
			"6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D896D5724E4C70A825F872C9EA60D2EDF5",
			"EC3A4E415B4E19A4568618029F427FA5DA9A8BC4AE92E02E06AAE5286B300C64DEF8F0EA9055866064A254515480BC13",
			"8015D9B72D7D57244EA8EF9AC0C621896708A59367F9DFB9F54CA84B3F1C9DB1288B231C3AE0D4FE7344FD2533264720",
			"test",
			"6D6DEFAC9AB64DABAFE36C6BF510352A4CC27001263638E5B16D9BB51D451559F918EEDAF2293BE5B475CC8F0188636B",
			"2D46F3BECBCC523D5F1A1256BF0C9B024D879BA9E838144C8BA6BAEB4B53B47D51AB373F9845C0514EEFB14024787265")
	})
	t.Run("P-521", func(t *testing.T) {
		testRFC6979(t, elliptic.P521(),
			"0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538",
			"1894550D0785932E00EAA23B694F213F8C3121F86DC97A04E5A7167DB4E5BCD371123D46E45DB6B5D5370A7F20FB633155D38FFA16D2BD761DCAC474B9A2F5023A4",
			"0493101C962CD4D2FDDF782285E64584139C2F91B47F87FF82354D6630F746A28A0DB25741B5B34A828008B22ACC23F924FAAFBD4D33F81EA66956DFEAA2BFDFCF5",
			"sample",
			"1511BB4D675114FE266FC4372B87682BAECC01D3CC62CF2303C92B3526012659D16876E25C7C1E57648F23B73564D67F61C6F14D527D54972810421E7D87589E1A7",
			"04A171143A83163D6DF460AAF61522695F207A58B95C0644D87E52AA1A347916E4F7A72930B1BC06DBE22CE3F58264AFD23704CBB63B29B931F7DE6C9D949A7ECFC")
		testRFC6979(t, elliptic.P521(),
			"0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538",
			"1894550D0785932E00EAA23B694F213F8C3121F86DC97A04E5A7167DB4E5BCD371123D46E45DB6B5D5370A7F20FB633155D38FFA16D2BD761DCAC474B9A2F5023A4",
			"0493101C962CD4D2FDDF782285E64584139C2F91B47F87FF82354D6630F746A28A0DB25741B5B34A828008B22ACC23F924FAAFBD4D33F81EA66956DFEAA2BFDFCF5",
			"test",
			"00E871C4A14F993C6C7369501900C4BC1E9C7B0B4BA44E04868B30B41D8071042EB28C4C250411D0CE08CD197E4188EA4876F279F90B3D8D74A3C76E6F1E4656AA8",
			"0CD52DBAA33B063C3A6CD8058A1FB0A46A4754B034FCC644766CA14DA8CA5CA9FDE00E88C1AD60CCBA759025299079D7A427EC3CC5B619BFBC828E7769BCD694E86")
	})
}

func testRFC6979(t *testing.T, curve elliptic.Curve, D, X, Y, msg, r, s string) {
	priv := &PrivateKey{
		D: fromHex(D),
		PublicKey: PublicKey{
			Curve: curve,
			X:     fromHex(X),
			Y:     fromHex(Y),
		},
	}
	h := sha256.Sum256([]byte(msg))
	sig, err := priv.Sign(nil, h[:], crypto.SHA256)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := encodeSignature(fromHex(r).Bytes(), fromHex(s).Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(sig, expected) {
		t.Errorf("signature mismatch:\n got: %x\nwant: %x", sig, expected)
	}
}

func TestParseAndBytesRoundTrip(t *testing.T) {
	testAllCurves(t, testParseAndBytesRoundTrip)
}

func testParseAndBytesRoundTrip(t *testing.T, curve elliptic.Curve) {
	if strings.HasSuffix(t.Name(), "/Generic") {
		t.Skip("these methods don't support generic curves")
	}
	priv, _ := GenerateKey(curve, rand.Reader)

	b, err := priv.PublicKey.Bytes()
	if err != nil {
		t.Fatalf("failed to serialize private key's public key: %v", err)
	}
	if b[0] != 4 {
		t.Fatalf("public key bytes doesn't start with 0x04 (uncompressed format)")
	}
	p, err := ParseUncompressedPublicKey(curve, b)
	if err != nil {
		t.Fatalf("failed to parse private key's public key: %v", err)
	}
	if !priv.PublicKey.Equal(p) {
		t.Errorf("parsed private key's public key doesn't match original")
	}

	bk, err := priv.Bytes()
	if err != nil {
		t.Fatalf("failed to serialize private key: %v", err)
	}
	k, err := ParseRawPrivateKey(curve, bk)
	if err != nil {
		t.Fatalf("failed to parse private key: %v", err)
	}
	if !priv.Equal(k) {
		t.Errorf("parsed private key doesn't match original")
	}

	if curve != elliptic.P224() {
		privECDH, err := priv.ECDH()
		if err != nil {
			t.Fatalf("failed to convert private key to ECDH: %v", err)
		}

		pp, err := privECDH.Curve().NewPublicKey(b)
		if err != nil {
			t.Fatalf("failed to parse with ECDH: %v", err)
		}
		if !privECDH.PublicKey().Equal(pp) {
			t.Errorf("parsed ECDH public key doesn't match original")
		}
		if !bytes.Equal(b, pp.Bytes()) {
			t.Errorf("encoded ECDH public key doesn't match Bytes")
		}

		kk, err := privECDH.Curve().NewPrivateKey(bk)
		if err != nil {
			t.Fatalf("failed to parse with ECDH: %v", err)
		}
		if !privECDH.Equal(kk) {
			t.Errorf("parsed ECDH private key doesn't match original")
		}
		if !bytes.Equal(bk, kk.Bytes()) {
			t.Errorf("encoded ECDH private key doesn't match Bytes")
		}
	}
}

func TestInvalidPublicKeys(t *testing.T) {
	testAllCurves(t, testInvalidPublicKeys)
}

func testInvalidPublicKeys(t *testing.T, curve elliptic.Curve) {
	t.Run("Infinity", func(t *testing.T) {
		k := &PublicKey{Curve: curve, X: big.NewInt(0), Y: big.NewInt(0)}
		if _, err := k.Bytes(); err == nil {
			t.Errorf("PublicKey.Bytes accepted infinity")
		}

		b := []byte{0}
		if _, err := ParseUncompressedPublicKey(curve, b); err == nil {
			t.Errorf("ParseUncompressedPublicKey accepted infinity")
		}
		b = make([]byte, 1+2*(curve.Params().BitSize+7)/8)
		b[0] = 4
		if _, err := ParseUncompressedPublicKey(curve, b); err == nil {
			t.Errorf("ParseUncompressedPublicKey accepted infinity")
		}
	})
	t.Run("NotOnCurve", func(t *testing.T) {
		k, _ := GenerateKey(curve, rand.Reader)
		k.X = k.X.Add(k.X, big.NewInt(1))
		if _, err := k.Bytes(); err == nil {
			t.Errorf("PublicKey.Bytes accepted not on curve")
		}

		b := make([]byte, 1+2*(curve.Params().BitSize+7)/8)
		b[0] = 4
		k.X.FillBytes(b[1 : 1+len(b)/2])
		k.Y.FillBytes(b[1+len(b)/2:])
		if _, err := ParseUncompressedPublicKey(curve, b); err == nil {
			t.Errorf("ParseUncompressedPublicKey accepted not on curve")
		}
	})
	t.Run("Compressed", func(t *testing.T) {
		k, _ := GenerateKey(curve, rand.Reader)
		b := elliptic.MarshalCompressed(curve, k.X, k.Y)
		if _, err := ParseUncompressedPublicKey(curve, b); err == nil {
			t.Errorf("ParseUncompressedPublicKey accepted compressed key")
		}
	})
}

func TestInvalidPrivateKeys(t *testing.T) {
	testAllCurves(t, testInvalidPrivateKeys)
}

func testInvalidPrivateKeys(t *testing.T, curve elliptic.Curve) {
	t.Run("Zero", func(t *testing.T) {
		k := &PrivateKey{PublicKey{curve, big.NewInt(0), big.NewInt(0)}, big.NewInt(0)}
		if _, err := k.Bytes(); err == nil {
			t.Errorf("PrivateKey.Bytes accepted zero key")
		}

		b := make([]byte, (curve.Params().BitSize+7)/8)
		if _, err := ParseRawPrivateKey(curve, b); err == nil {
			t.Errorf("ParseRawPrivateKey accepted zero key")
		}
	})
	t.Run("Overflow", func(t *testing.T) {
		d := new(big.Int).Add(curve.Params().N, big.NewInt(5))
		x, y := curve.ScalarBaseMult(d.Bytes())
		k := &PrivateKey{PublicKey{curve, x, y}, d}
		if _, err := k.Bytes(); err == nil {
			t.Errorf("PrivateKey.Bytes accepted overflow key")
		}

		b := make([]byte, (curve.Params().BitSize+7)/8)
		k.D.FillBytes(b)
		if _, err := ParseRawPrivateKey(curve, b); err == nil {
			t.Errorf("ParseRawPrivateKey accepted overflow key")
		}
	})
	t.Run("Length", func(t *testing.T) {
		b := []byte{1, 2, 3}
		if _, err := ParseRawPrivateKey(curve, b); err == nil {
			t.Errorf("ParseRawPrivateKey accepted short key")
		}

		b = make([]byte, (curve.Params().BitSize+7)/8)
		b = append(b, []byte{1, 2, 3}...)
		if _, err := ParseRawPrivateKey(curve, b); err == nil {
			t.Errorf("ParseRawPrivateKey accepted long key")
		}
	})
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
