// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fipstest collects external tests that would ordinarily live in
// crypto/internal/fips140/... packages. That tree gets snapshot at each
// validation, while we want tests to evolve and still apply to all versions of
// the module. Also, we can't fix failing tests in a module snapshot, so we need
// to either minimize, skip, or remove them. Finally, the module needs to avoid
// importing internal packages like testenv and cryptotest to avoid locking in
// their APIs.
//
// Also, this package includes the ACVP and functional testing harnesses.
package fipstest

import (
	"bytes"
	"crypto/internal/boring"
	"crypto/internal/fips140"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/aes/gcm"
	"crypto/internal/fips140/check"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/ecdh"
	"crypto/internal/fips140/ecdsa"
	"crypto/internal/fips140/ed25519"
	"crypto/internal/fips140/hkdf"
	"crypto/internal/fips140/hmac"
	"crypto/internal/fips140/mlkem"
	"crypto/internal/fips140/pbkdf2"
	"crypto/internal/fips140/rsa"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"crypto/internal/fips140/tls12"
	"crypto/internal/fips140/tls13"
	"crypto/rand"
	"encoding/hex"
	"runtime/debug"
	"strings"
	"testing"
)

func moduleStatus(t *testing.T) {
	if fips140.Enabled {
		t.Log("FIPS 140-3 mode enabled")
	} else {
		t.Log("FIPS 140-3 mode not enabled")
	}

	t.Logf("Module name: %s", fips140.Name())
	t.Logf("Module version: %s", fips140.Version())

	if noPAAPAI {
		t.Log("PAA/PAI disabled")
	} else {
		t.Log("PAA/PAI enabled")
	}

	if check.Verified {
		t.Log("FIPS 140-3 integrity self-check succeeded")
	} else {
		t.Log("FIPS 140-3 integrity self-check not succeeded")
	}
}

func TestVersion(t *testing.T) {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		t.Skip("no build info")
	}
	for _, setting := range bi.Settings {
		if setting.Key != "GOFIPS140" {
			continue
		}
		exp := setting.Value
		// Remove the -hash suffix, if any.
		// The version from fips140.Version omits it.
		exp, _, _ = strings.Cut(exp, "-")
		if v := fips140.Version(); v != exp {
			t.Errorf("Version is %q, expected %q", v, exp)
		}
		return
	}
	// Without GOFIPS140, the Version should be "latest".
	if v := fips140.Version(); v != "latest" {
		t.Errorf("Version is %q, expected latest", v)
	}
}

func TestFIPS140(t *testing.T) {
	moduleStatus(t)
	if boring.Enabled {
		t.Skip("Go+BoringCrypto shims prevent the service indicator from being set")
	}

	aesKey := make([]byte, 128/8)
	aesIV := make([]byte, aes.BlockSize)
	plaintext := []byte("Go Cryptographic Module TestFIPS140 plaintext...")
	plaintextSHA256 := decodeHex(t, "06b2614e2ef315832b23f5d0ff70294d8ddd3889527dfbe75707fe41da929325")
	aesBlock, err := aes.New(aesKey)
	fatalIfErr(t, err)

	t.Run("AES-CTR", func(t *testing.T) {
		ensureServiceIndicator(t)
		ctr := aes.NewCTR(aesBlock, aesIV)
		ciphertext := make([]byte, len(plaintext))
		ctr.XORKeyStream(ciphertext, plaintext)
		t.Logf("AES-CTR ciphertext: %x", ciphertext)
		out := make([]byte, len(plaintext))
		ctr = aes.NewCTR(aesBlock, aesIV)
		ctr.XORKeyStream(out, ciphertext)
		t.Logf("AES-CTR decrypted plaintext: %s", out)
		if !bytes.Equal(plaintext, out) {
			t.Errorf("AES-CTR round trip failed")
		}
	})

	t.Run("AES-CBC", func(t *testing.T) {
		ensureServiceIndicator(t)
		cbcEnc := aes.NewCBCEncrypter(aesBlock, [16]byte(aesIV))
		ciphertext := make([]byte, len(plaintext))
		cbcEnc.CryptBlocks(ciphertext, plaintext)
		t.Logf("AES-CBC ciphertext: %x", ciphertext)
		cbcDec := aes.NewCBCDecrypter(aesBlock, [16]byte(aesIV))
		out := make([]byte, len(plaintext))
		cbcDec.CryptBlocks(out, ciphertext)
		t.Logf("AES-CBC decrypted plaintext: %s", out)
		if !bytes.Equal(plaintext, out) {
			t.Errorf("AES-CBC round trip failed")
		}
	})

	t.Run("AES-GCM", func(t *testing.T) {
		ensureServiceIndicator(t)
		g, err := gcm.New(aesBlock, 12, 16)
		fatalIfErr(t, err)
		nonce := make([]byte, 12)
		ciphertext := make([]byte, len(plaintext)+g.Overhead())
		gcm.SealWithRandomNonce(g, nonce, ciphertext, plaintext, nil)
		t.Logf("AES-GCM ciphertext: %x", ciphertext)
		out, err := g.Open(nil, nonce, ciphertext, nil)
		fatalIfErr(t, err)
		t.Logf("AES-GCM decrypted plaintext: %s", out)
		if !bytes.Equal(plaintext, out) {
			t.Errorf("AES-GCM round trip failed")
		}
	})

	t.Run("Counter KDF", func(t *testing.T) {
		ensureServiceIndicator(t)
		k := gcm.NewCounterKDF(aesBlock)
		context := [12]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		key := k.DeriveKey(0x01, context)
		t.Logf("Counter KDF key: %x", key)
	})

	t.Run("KAS-ECC-SSC ephemeralUnified", func(t *testing.T) {
		ensureServiceIndicator(t)
		k, err := ecdh.GenerateKey(ecdh.P256(), rand.Reader)
		fatalIfErr(t, err)
		pk := k.PublicKey()
		shared, err := ecdh.ECDH(ecdh.P256(), k, pk)
		fatalIfErr(t, err)
		t.Logf("KAS-ECC-SSC shared secret: %x", shared)
	})

	t.Run("ECDSA KeyGen, SigGen, SigVer", func(t *testing.T) {
		ensureServiceIndicator(t)
		k, err := ecdsa.GenerateKey(ecdsa.P256(), rand.Reader)
		fatalIfErr(t, err)

		sig, err := ecdsa.Sign(ecdsa.P256(), sha256.New, k, rand.Reader, plaintextSHA256)
		fatalIfErr(t, err)
		t.Logf("ECDSA signature: %x", sig)
		err = ecdsa.Verify(ecdsa.P256(), k.PublicKey(), plaintextSHA256, sig)
		if err != nil {
			t.Errorf("ECDSA signature verification failed")
		}

		sig, err = ecdsa.SignDeterministic(ecdsa.P256(), sha256.New, k, plaintextSHA256)
		fatalIfErr(t, err)
		t.Logf("ECDSA deterministic signature: %x", sig)
		err = ecdsa.Verify(ecdsa.P256(), k.PublicKey(), plaintextSHA256, sig)
		if err != nil {
			t.Errorf("ECDSA deterministic signature verification failed")
		}
	})

	t.Run("EDDSA KeyGen, SigGen, SigVer", func(t *testing.T) {
		ensureServiceIndicator(t)
		k, err := ed25519.GenerateKey()
		fatalIfErr(t, err)

		sig := ed25519.Sign(k, plaintext)
		t.Logf("EDDSA signature: %x", sig)

		pk, err := ed25519.NewPublicKey(k.PublicKey())
		fatalIfErr(t, err)
		err = ed25519.Verify(pk, plaintext, sig)
		if err != nil {
			t.Errorf("EDDSA signature verification failed")
		}
	})

	t.Run("ctrDRBG", func(t *testing.T) {
		ensureServiceIndicator(t)
		r := drbg.NewCounter((*[48]byte)(plaintext))
		r.Reseed((*[48]byte)(plaintext), (*[48]byte)(plaintext))
		out := make([]byte, 16)
		r.Generate(out, (*[48]byte)(plaintext))
		t.Logf("ctrDRBG output: %x", out)
	})

	t.Run("HMAC", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := hmac.New(sha256.New, plaintext)
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("HMAC output: %x", out)
	})

	t.Run("ML-KEM KeyGen, Encap, Decap", func(t *testing.T) {
		ensureServiceIndicator(t)
		k, err := mlkem.GenerateKey768()
		fatalIfErr(t, err)

		ss, c := k.EncapsulationKey().Encapsulate()
		t.Logf("ML-KEM encapsulation: %x", c)

		ss2, err := k.Decapsulate(c)
		fatalIfErr(t, err)
		t.Logf("ML-KEM shared secret: %x", ss)
		if !bytes.Equal(ss, ss2) {
			t.Errorf("ML-KEM round trip failed")
		}
	})

	var rsaKey *rsa.PrivateKey
	t.Run("RSA KeyGen", func(t *testing.T) {
		ensureServiceIndicator(t)
		var err error
		rsaKey, err = rsa.GenerateKey(rand.Reader, 2048)
		fatalIfErr(t, err)
		t.Log("RSA key generated")
	})

	t.Run("RSA SigGen, SigVer PKCS 1.5", func(t *testing.T) {
		ensureServiceIndicator(t)
		sig, err := rsa.SignPKCS1v15(rsaKey, "SHA-256", plaintextSHA256)
		fatalIfErr(t, err)
		t.Logf("RSA PKCS1v15 signature: %x", sig)

		err = rsa.VerifyPKCS1v15(rsaKey.PublicKey(), "SHA-256", plaintextSHA256, sig)
		fatalIfErr(t, err)
	})

	t.Run("RSA SigGen, SigVer PSS", func(t *testing.T) {
		ensureServiceIndicator(t)
		sig, err := rsa.SignPSS(rand.Reader, rsaKey, sha256.New(), plaintextSHA256, 16)
		fatalIfErr(t, err)
		t.Logf("RSA PSS signature: %x", sig)

		err = rsa.VerifyPSS(rsaKey.PublicKey(), sha256.New(), plaintextSHA256, sig)
		fatalIfErr(t, err)
	})

	t.Run("RSA KeyGen w/ small key [NOT APPROVED]", func(t *testing.T) {
		ensureServiceIndicatorFalse(t)
		_, err := rsa.GenerateKey(rand.Reader, 512)
		fatalIfErr(t, err)
		t.Log("RSA key generated")
	})

	t.Run("KTS IFC OAEP", func(t *testing.T) {
		ensureServiceIndicator(t)
		c, err := rsa.EncryptOAEP(sha256.New(), sha256.New(), rand.Reader, rsaKey.PublicKey(), plaintextSHA256, nil)
		fatalIfErr(t, err)
		t.Logf("RSA OAEP ciphertext: %x", c)

		out, err := rsa.DecryptOAEP(sha256.New(), sha256.New(), rsaKey, c, nil)
		fatalIfErr(t, err)
		t.Logf("RSA OAEP decrypted plaintext: %x", out)
		if !bytes.Equal(plaintextSHA256, out) {
			t.Errorf("RSA OAEP round trip failed")
		}
	})

	t.Run("SHA2-224", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha256.New224()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA2-224 output: %x", out)
	})

	t.Run("SHA2-256", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha256.New()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA2-256 output: %x", out)
	})

	t.Run("SHA2-384", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha512.New384()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA2-384 output: %x", out)
	})

	t.Run("SHA2-512", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha512.New()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA2-512 output: %x", out)
	})

	t.Run("SHA2-512/224", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha512.New512_224()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA2-512/224 output: %x", out)
	})

	t.Run("SHA2-512/256", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha512.New512_256()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA2-512/256 output: %x", out)
	})

	t.Run("SHA3-224", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.New224()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA3-224 output: %x", out)
	})

	t.Run("SHA3-256", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.New256()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA3-256 output: %x", out)
	})

	t.Run("SHA3-384", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.New384()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA3-384 output: %x", out)
	})

	t.Run("SHA3-512", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.New512()
		h.Write(plaintext)
		out := h.Sum(nil)
		t.Logf("SHA3-512 output: %x", out)
	})

	t.Run("SHAKE-128", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.NewShake128()
		h.Write(plaintext)
		out := make([]byte, 16)
		h.Read(out)
		t.Logf("SHAKE-128 output: %x", out)
	})

	t.Run("SHAKE-256", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.NewShake256()
		h.Write(plaintext)
		out := make([]byte, 16)
		h.Read(out)
		t.Logf("SHAKE-256 output: %x", out)
	})

	t.Run("cSHAKE-128", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.NewCShake128(nil, []byte("test"))
		h.Write(plaintext)
		out := make([]byte, 16)
		h.Read(out)
		t.Logf("cSHAKE-128 output: %x", out)
	})

	t.Run("cSHAKE-256", func(t *testing.T) {
		ensureServiceIndicator(t)
		h := sha3.NewCShake256(nil, []byte("test"))
		h.Write(plaintext)
		out := make([]byte, 16)
		h.Read(out)
		t.Logf("cSHAKE-256 output: %x", out)
	})

	t.Run("KDA HKDF", func(t *testing.T) {
		ensureServiceIndicator(t)
		key := hkdf.Key(sha256.New, plaintextSHA256, []byte("salt"), "info", 16)
		t.Logf("HKDF key: %x", key)
	})

	t.Run("KDA OneStepNoCounter", func(t *testing.T) {
		ensureServiceIndicator(t)
		key := hkdf.Extract(sha256.New, plaintextSHA256, []byte("salt"))
		t.Logf("KDA OneStepNoCounter key: %x", key)
	})

	t.Run("Feedback KDF", func(t *testing.T) {
		ensureServiceIndicator(t)
		key := hkdf.Expand(sha256.New, plaintextSHA256, "info", 16)
		t.Logf("Feedback KDF key: %x", key)
	})

	t.Run("PBKDF", func(t *testing.T) {
		ensureServiceIndicator(t)
		key, err := pbkdf2.Key(sha256.New, "password", plaintextSHA256, 2, 16)
		fatalIfErr(t, err)
		t.Logf("PBKDF key: %x", key)
	})

	t.Run("KDF TLS v1.2 CVL", func(t *testing.T) {
		ensureServiceIndicator(t)
		key := tls12.MasterSecret(sha256.New, plaintextSHA256, []byte("test"))
		t.Logf("TLS v1.2 CVL Master Secret: %x", key)
	})

	t.Run("KDF TLS v1.3 CVL", func(t *testing.T) {
		ensureServiceIndicator(t)
		es := tls13.NewEarlySecret(sha256.New, plaintextSHA256)
		hs := es.HandshakeSecret(plaintextSHA256)
		ms := hs.MasterSecret()
		client := ms.ClientApplicationTrafficSecret(sha256.New())
		server := ms.ServerApplicationTrafficSecret(sha256.New())
		t.Logf("TLS v1.3 CVL Application Traffic Secrets: client %x, server %x", client, server)
	})
}

func ensureServiceIndicator(t *testing.T) {
	fips140.ResetServiceIndicator()
	t.Cleanup(func() {
		if fips140.ServiceIndicator() {
			t.Logf("Service indicator is set")
		} else {
			t.Errorf("Service indicator is not set")
		}
	})
}

func ensureServiceIndicatorFalse(t *testing.T) {
	fips140.ResetServiceIndicator()
	t.Cleanup(func() {
		if !fips140.ServiceIndicator() {
			t.Logf("Service indicator is not set")
		} else {
			t.Errorf("Service indicator is set")
		}
	})
}

func fatalIfErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func decodeHex(t *testing.T, s string) []byte {
	t.Helper()
	s = strings.ReplaceAll(s, " ", "")
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
