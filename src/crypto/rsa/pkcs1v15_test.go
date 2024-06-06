// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa_test

import (
	"bytes"
	"crypto"
	"crypto/rand"
	. "crypto/rsa"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"io"
	"math/big"
	"testing"
	"testing/quick"
)

func decodeBase64(in string) []byte {
	out := make([]byte, base64.StdEncoding.DecodedLen(len(in)))
	n, err := base64.StdEncoding.Decode(out, []byte(in))
	if err != nil {
		return nil
	}
	return out[0:n]
}

type DecryptPKCS1v15Test struct {
	in, out string
}

// These test vectors were generated with `openssl rsautl -pkcs -encrypt`
var decryptPKCS1v15Tests = []DecryptPKCS1v15Test{
	{
		"gIcUIoVkD6ATMBk/u/nlCZCCWRKdkfjCgFdo35VpRXLduiKXhNz1XupLLzTXAybEq15juc+EgY5o0DHv/nt3yg==",
		"x",
	},
	{
		"Y7TOCSqofGhkRb+jaVRLzK8xw2cSo1IVES19utzv6hwvx+M8kFsoWQm5DzBeJCZTCVDPkTpavUuEbgp8hnUGDw==",
		"testing.",
	},
	{
		"arReP9DJtEVyV2Dg3dDp4c/PSk1O6lxkoJ8HcFupoRorBZG+7+1fDAwT1olNddFnQMjmkb8vxwmNMoTAT/BFjQ==",
		"testing.\n",
	},
	{
		"WtaBXIoGC54+vH0NH0CHHE+dRDOsMc/6BrfFu2lEqcKL9+uDuWaf+Xj9mrbQCjjZcpQuX733zyok/jsnqe/Ftw==",
		"01234567890123456789012345678901234567890123456789012",
	},
}

func TestDecryptPKCS1v15(t *testing.T) {
	decryptionFuncs := []func([]byte) ([]byte, error){
		func(ciphertext []byte) (plaintext []byte, err error) {
			return DecryptPKCS1v15(nil, rsaPrivateKey, ciphertext)
		},
		func(ciphertext []byte) (plaintext []byte, err error) {
			return rsaPrivateKey.Decrypt(nil, ciphertext, nil)
		},
	}

	for _, decryptFunc := range decryptionFuncs {
		for i, test := range decryptPKCS1v15Tests {
			out, err := decryptFunc(decodeBase64(test.in))
			if err != nil {
				t.Errorf("#%d error decrypting: %v", i, err)
			}
			want := []byte(test.out)
			if !bytes.Equal(out, want) {
				t.Errorf("#%d got:%#v want:%#v", i, out, want)
			}
		}
	}
}

func TestEncryptPKCS1v15(t *testing.T) {
	random := rand.Reader
	k := (rsaPrivateKey.N.BitLen() + 7) / 8

	tryEncryptDecrypt := func(in []byte, blind bool) bool {
		if len(in) > k-11 {
			in = in[0 : k-11]
		}

		ciphertext, err := EncryptPKCS1v15(random, &rsaPrivateKey.PublicKey, in)
		if err != nil {
			t.Errorf("error encrypting: %s", err)
			return false
		}

		var rand io.Reader
		if !blind {
			rand = nil
		} else {
			rand = random
		}
		plaintext, err := DecryptPKCS1v15(rand, rsaPrivateKey, ciphertext)
		if err != nil {
			t.Errorf("error decrypting: %s", err)
			return false
		}

		if !bytes.Equal(plaintext, in) {
			t.Errorf("output mismatch: %#v %#v", plaintext, in)
			return false
		}
		return true
	}

	config := new(quick.Config)
	if testing.Short() {
		config.MaxCount = 10
	}
	quick.Check(tryEncryptDecrypt, config)
}

// These test vectors were generated with `openssl rsautl -pkcs -encrypt`
var decryptPKCS1v15SessionKeyTests = []DecryptPKCS1v15Test{
	{
		"e6ukkae6Gykq0fKzYwULpZehX+UPXYzMoB5mHQUDEiclRbOTqas4Y0E6nwns1BBpdvEJcilhl5zsox/6DtGsYg==",
		"1234",
	},
	{
		"Dtis4uk/q/LQGGqGk97P59K03hkCIVFMEFZRgVWOAAhxgYpCRG0MX2adptt92l67IqMki6iVQyyt0TtX3IdtEw==",
		"FAIL",
	},
	{
		"LIyFyCYCptPxrvTxpol8F3M7ZivlMsf53zs0vHRAv+rDIh2YsHS69ePMoPMe3TkOMZ3NupiL3takPxIs1sK+dw==",
		"abcd",
	},
	{
		"bafnobel46bKy76JzqU/RIVOH0uAYvzUtauKmIidKgM0sMlvobYVAVQPeUQ/oTGjbIZ1v/6Gyi5AO4DtHruGdw==",
		"FAIL",
	},
}

func TestEncryptPKCS1v15SessionKey(t *testing.T) {
	for i, test := range decryptPKCS1v15SessionKeyTests {
		key := []byte("FAIL")
		err := DecryptPKCS1v15SessionKey(nil, rsaPrivateKey, decodeBase64(test.in), key)
		if err != nil {
			t.Errorf("#%d error decrypting", i)
		}
		want := []byte(test.out)
		if !bytes.Equal(key, want) {
			t.Errorf("#%d got:%#v want:%#v", i, key, want)
		}
	}
}

func TestEncryptPKCS1v15DecrypterSessionKey(t *testing.T) {
	for i, test := range decryptPKCS1v15SessionKeyTests {
		plaintext, err := rsaPrivateKey.Decrypt(rand.Reader, decodeBase64(test.in), &PKCS1v15DecryptOptions{SessionKeyLen: 4})
		if err != nil {
			t.Fatalf("#%d: error decrypting: %s", i, err)
		}
		if len(plaintext) != 4 {
			t.Fatalf("#%d: incorrect length plaintext: got %d, want 4", i, len(plaintext))
		}

		if test.out != "FAIL" && !bytes.Equal(plaintext, []byte(test.out)) {
			t.Errorf("#%d: incorrect plaintext: got %x, want %x", i, plaintext, test.out)
		}
	}
}

func TestNonZeroRandomBytes(t *testing.T) {
	random := rand.Reader

	b := make([]byte, 512)
	err := NonZeroRandomBytes(b, random)
	if err != nil {
		t.Errorf("returned error: %s", err)
	}
	for _, b := range b {
		if b == 0 {
			t.Errorf("Zero octet found")
			return
		}
	}
}

type signPKCS1v15Test struct {
	in, out string
}

// These vectors have been tested with
//
//	`openssl rsautl -verify -inkey pk -in signature | hexdump -C`
var signPKCS1v15Tests = []signPKCS1v15Test{
	{"Test.\n", "a4f3fa6ea93bcdd0c57be020c1193ecbfd6f200a3d95c409769b029578fa0e336ad9a347600e40d3ae823b8c7e6bad88cc07c1d54c3a1523cbbb6d58efc362ae"},
}

func TestSignPKCS1v15(t *testing.T) {
	for i, test := range signPKCS1v15Tests {
		h := sha1.New()
		h.Write([]byte(test.in))
		digest := h.Sum(nil)

		s, err := SignPKCS1v15(nil, rsaPrivateKey, crypto.SHA1, digest)
		if err != nil {
			t.Errorf("#%d %s", i, err)
		}

		expected, _ := hex.DecodeString(test.out)
		if !bytes.Equal(s, expected) {
			t.Errorf("#%d got: %x want: %x", i, s, expected)
		}
	}
}

func TestSignPKCS1v15WithPublicKeySizeZero(t *testing.T) {
	h := sha1.New()
	h.Write([]byte("key"))
	digest := h.Sum(nil)
	_, err := SignPKCS1v15(nil,
		&PrivateKey{
			PublicKey: PublicKey{
				N: big.NewInt(0),
			},
		}, crypto.SHA1, digest)
	if err == nil {
		t.Error("expected error but got nil")
	}
	if err != nil && err.Error() != "crypto/rsa: public key size zero" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestVerifyPKCS1v15(t *testing.T) {
	for i, test := range signPKCS1v15Tests {
		h := sha1.New()
		h.Write([]byte(test.in))
		digest := h.Sum(nil)

		sig, _ := hex.DecodeString(test.out)

		err := VerifyPKCS1v15(&rsaPrivateKey.PublicKey, crypto.SHA1, digest, sig)
		if err != nil {
			t.Errorf("#%d %s", i, err)
		}
	}
}

func TestOverlongMessagePKCS1v15(t *testing.T) {
	ciphertext := decodeBase64("fjOVdirUzFoLlukv80dBllMLjXythIf22feqPrNo0YoIjzyzyoMFiLjAc/Y4krkeZ11XFThIrEvw\nkRiZcCq5ng==")
	_, err := DecryptPKCS1v15(nil, rsaPrivateKey, ciphertext)
	if err == nil {
		t.Error("RSA decrypted a message that was too long.")
	}
}

func TestUnpaddedSignature(t *testing.T) {
	msg := []byte("Thu Dec 19 18:06:16 EST 2013\n")
	// This base64 value was generated with:
	// % echo Thu Dec 19 18:06:16 EST 2013 > /tmp/msg
	// % openssl rsautl -sign -inkey key -out /tmp/sig -in /tmp/msg
	//
	// Where "key" contains the RSA private key given at the bottom of this
	// file.
	expectedSig := decodeBase64("pX4DR8azytjdQ1rtUiC040FjkepuQut5q2ZFX1pTjBrOVKNjgsCDyiJDGZTCNoh9qpXYbhl7iEym30BWWwuiZg==")

	sig, err := SignPKCS1v15(nil, rsaPrivateKey, crypto.Hash(0), msg)
	if err != nil {
		t.Fatalf("SignPKCS1v15 failed: %s", err)
	}
	if !bytes.Equal(sig, expectedSig) {
		t.Fatalf("signature is not expected value: got %x, want %x", sig, expectedSig)
	}
	if err := VerifyPKCS1v15(&rsaPrivateKey.PublicKey, crypto.Hash(0), msg, sig); err != nil {
		t.Fatalf("signature failed to verify: %s", err)
	}
}

func TestShortSessionKey(t *testing.T) {
	// This tests that attempting to decrypt a session key where the
	// ciphertext is too small doesn't run outside the array bounds.
	ciphertext, err := EncryptPKCS1v15(rand.Reader, &rsaPrivateKey.PublicKey, []byte{1})
	if err != nil {
		t.Fatalf("Failed to encrypt short message: %s", err)
	}

	var key [32]byte
	if err := DecryptPKCS1v15SessionKey(nil, rsaPrivateKey, ciphertext, key[:]); err != nil {
		t.Fatalf("Failed to decrypt short message: %s", err)
	}

	for _, v := range key {
		if v != 0 {
			t.Fatal("key was modified when ciphertext was invalid")
		}
	}
}

var rsaPrivateKey = parseKey(testingKey(`-----BEGIN RSA TESTING KEY-----
MIIBOgIBAAJBALKZD0nEffqM1ACuak0bijtqE2QrI/KLADv7l3kK3ppMyCuLKoF0
fd7Ai2KW5ToIwzFofvJcS/STa6HA5gQenRUCAwEAAQJBAIq9amn00aS0h/CrjXqu
/ThglAXJmZhOMPVn4eiu7/ROixi9sex436MaVeMqSNf7Ex9a8fRNfWss7Sqd9eWu
RTUCIQDasvGASLqmjeffBNLTXV2A5g4t+kLVCpsEIZAycV5GswIhANEPLmax0ME/
EO+ZJ79TJKN5yiGBRsv5yvx5UiHxajEXAiAhAol5N4EUyq6I9w1rYdhPMGpLfk7A
IU2snfRJ6Nq2CQIgFrPsWRCkV+gOYcajD17rEqmuLrdIRexpg8N1DOSXoJ8CIGlS
tAboUGBxTDq3ZroNism3DaMIbKPyYrAqhKov1h5V
-----END RSA TESTING KEY-----`))

func parsePublicKey(s string) *PublicKey {
	p, _ := pem.Decode([]byte(s))
	k, err := x509.ParsePKCS1PublicKey(p.Bytes)
	if err != nil {
		panic(err)
	}
	return k
}

func TestShortPKCS1v15Signature(t *testing.T) {
	pub := parsePublicKey(`-----BEGIN RSA PUBLIC KEY-----
MEgCQQCd9BVzo775lkohasxjnefF1nCMcNoibqIWEVDe/K7M2GSoO4zlSQB+gkix
O3AnTcdHB51iaZpWfxPSnew8yfulAgMBAAE=
-----END RSA PUBLIC KEY-----`)
	sig, err := hex.DecodeString("193a310d0dcf64094c6e3a00c8219b80ded70535473acff72c08e1222974bb24a93a535b1dc4c59fc0e65775df7ba2007dd20e9193f4c4025a18a7070aee93")
	if err != nil {
		t.Fatalf("failed to decode signature: %s", err)
	}

	h := sha256.Sum256([]byte("hello"))
	err = VerifyPKCS1v15(pub, crypto.SHA256, h[:], sig)
	if err == nil {
		t.Fatal("VerifyPKCS1v15 accepted a truncated signature")
	}
}
