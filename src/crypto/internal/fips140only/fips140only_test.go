// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140only_test

import (
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/dsa"
	"crypto/ecdh"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/hkdf"
	"crypto/hmac"
	"crypto/hpke"
	"crypto/internal/cryptotest"
	"crypto/internal/fips140"
	"crypto/internal/fips140only"
	"crypto/md5"
	"crypto/mlkem"
	"crypto/mlkem/mlkemtest"
	"crypto/pbkdf2"
	"crypto/rand"
	"crypto/rc4"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/sha256"
	_ "crypto/sha3"
	_ "crypto/sha512"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"internal/godebug"
	"internal/testenv"
	"io"
	"math/big"
	"os"
	"strings"
	"testing"

	"golang.org/x/crypto/chacha20poly1305"
)

func TestFIPS140Only(t *testing.T) {
	cryptotest.MustSupportFIPS140(t)
	if !fips140only.Enforced() {
		cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestFIPS140Only$", "-test.v")
		cmd.Env = append(cmd.Environ(), "GODEBUG=fips140=only")
		out, err := cmd.CombinedOutput()
		t.Logf("running with GODEBUG=fips140=only:\n%s", out)
		if err != nil {
			t.Errorf("fips140=only subprocess failed: %v", err)
		}
		return
	}
	t.Run("cryptocustomrand=0", func(t *testing.T) {
		t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",cryptocustomrand=0")
		testFIPS140Only(t)
	})
	t.Run("cryptocustomrand=1", func(t *testing.T) {
		t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",cryptocustomrand=1")
		testFIPS140Only(t)
	})
}

func testFIPS140Only(t *testing.T) {
	if !fips140only.Enforced() {
		t.Fatal("FIPS 140-only mode not enforced")
	}
	t.Logf("GODEBUG=fips140=only enabled")
	fips140.ResetServiceIndicator()

	aesBlock, err := aes.NewCipher(make([]byte, 16))
	if err != nil {
		t.Fatal(err)
	}
	notAESBlock := blockWrap{aesBlock}
	iv := make([]byte, aes.BlockSize)

	cipher.NewCBCEncrypter(aesBlock, iv)
	expectPanic(t, func() { cipher.NewCBCEncrypter(notAESBlock, iv) })
	cipher.NewCBCDecrypter(aesBlock, iv)
	expectPanic(t, func() { cipher.NewCBCDecrypter(notAESBlock, iv) })

	expectPanic(t, func() { cipher.NewCFBEncrypter(aesBlock, iv) })
	expectPanic(t, func() { cipher.NewCFBDecrypter(aesBlock, iv) })

	cipher.NewCTR(aesBlock, iv)
	expectPanic(t, func() { cipher.NewCTR(notAESBlock, iv) })

	expectPanic(t, func() { cipher.NewOFB(aesBlock, iv) })

	expectErr(t, errRet2(cipher.NewGCM(aesBlock)))
	expectErr(t, errRet2(cipher.NewGCMWithNonceSize(aesBlock, 12)))
	expectErr(t, errRet2(cipher.NewGCMWithTagSize(aesBlock, 12)))
	expectNoErr(t, errRet2(cipher.NewGCMWithRandomNonce(aesBlock)))

	expectErr(t, errRet2(des.NewCipher(make([]byte, 8))))
	expectErr(t, errRet2(des.NewTripleDESCipher(make([]byte, 24))))

	expectErr(t, errRet2(rc4.NewCipher(make([]byte, 16))))

	expectErr(t, errRet2(chacha20poly1305.New(make([]byte, chacha20poly1305.KeySize))))
	expectErr(t, errRet2(chacha20poly1305.NewX(make([]byte, chacha20poly1305.KeySize))))

	expectPanic(t, func() { md5.New().Sum(nil) })
	expectErr(t, errRet2(md5.New().Write(make([]byte, 16))))
	expectPanic(t, func() { md5.Sum([]byte("foo")) })

	expectPanic(t, func() { sha1.New().Sum(nil) })
	expectErr(t, errRet2(sha1.New().Write(make([]byte, 16))))
	expectPanic(t, func() { sha1.Sum([]byte("foo")) })

	withApprovedHash(func(h crypto.Hash) { h.New().Sum(nil) })
	withNonApprovedHash(func(h crypto.Hash) { expectPanic(t, func() { h.New().Sum(nil) }) })

	expectErr(t, errRet2(pbkdf2.Key(sha256.New, "password", make([]byte, 16), 1, 10)))
	expectErr(t, errRet2(pbkdf2.Key(sha256.New, "password", make([]byte, 10), 1, 14)))
	withNonApprovedHash(func(h crypto.Hash) {
		expectErr(t, errRet2(pbkdf2.Key(h.New, "password", make([]byte, 16), 1, 14)))
	})
	withApprovedHash(func(h crypto.Hash) {
		expectNoErr(t, errRet2(pbkdf2.Key(h.New, "password", make([]byte, 16), 1, 14)))
	})

	expectPanic(t, func() { hmac.New(sha256.New, make([]byte, 10)) })
	withNonApprovedHash(func(h crypto.Hash) {
		expectPanic(t, func() { hmac.New(h.New, make([]byte, 16)) })
	})
	withApprovedHash(func(h crypto.Hash) { hmac.New(h.New, make([]byte, 16)) })

	expectErr(t, errRet2(hkdf.Key(sha256.New, make([]byte, 10), nil, "", 16)))
	withNonApprovedHash(func(h crypto.Hash) {
		expectErr(t, errRet2(hkdf.Key(h.New, make([]byte, 16), nil, "", 16)))
	})
	withApprovedHash(func(h crypto.Hash) {
		expectNoErr(t, errRet2(hkdf.Key(h.New, make([]byte, 16), nil, "", 16)))
	})

	expectErr(t, errRet2(hkdf.Extract(sha256.New, make([]byte, 10), nil)))
	withNonApprovedHash(func(h crypto.Hash) {
		expectErr(t, errRet2(hkdf.Extract(h.New, make([]byte, 16), nil)))
	})
	withApprovedHash(func(h crypto.Hash) {
		expectNoErr(t, errRet2(hkdf.Extract(h.New, make([]byte, 16), nil)))
	})

	expectErr(t, errRet2(hkdf.Expand(sha256.New, make([]byte, 10), "", 16)))
	withNonApprovedHash(func(h crypto.Hash) {
		expectErr(t, errRet2(hkdf.Expand(h.New, make([]byte, 16), "", 16)))
	})
	withApprovedHash(func(h crypto.Hash) {
		expectNoErr(t, errRet2(hkdf.Expand(h.New, make([]byte, 16), "", 16)))
	})

	expectErr(t, errRet2(rand.Prime(rand.Reader, 10)))

	expectErr(t, dsa.GenerateParameters(&dsa.Parameters{}, rand.Reader, dsa.L1024N160))
	expectErr(t, dsa.GenerateKey(&dsa.PrivateKey{}, rand.Reader))
	expectErr(t, errRet3(dsa.Sign(rand.Reader, &dsa.PrivateKey{}, make([]byte, 16))))
	expectPanic(t, func() {
		dsa.Verify(&dsa.PublicKey{}, make([]byte, 16), big.NewInt(1), big.NewInt(1))
	})

	expectErr(t, errRet2(ecdh.X25519().GenerateKey(rand.Reader)))
	expectErr(t, errRet2(ecdh.X25519().NewPrivateKey(make([]byte, 32))))
	expectErr(t, errRet2(ecdh.X25519().NewPublicKey(make([]byte, 32))))
	for _, curve := range []ecdh.Curve{ecdh.P256(), ecdh.P384(), ecdh.P521()} {
		expectErrIfCustomRand(t, errRet2(curve.GenerateKey(readerWrap{rand.Reader})))
		k, err := curve.GenerateKey(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		expectNoErr(t, errRet2(curve.NewPrivateKey(k.Bytes())))
		expectNoErr(t, errRet2(curve.NewPublicKey(k.PublicKey().Bytes())))
	}

	for _, curve := range []elliptic.Curve{elliptic.P256(), elliptic.P384(), elliptic.P521()} {
		expectErrIfCustomRand(t, errRet2(ecdsa.GenerateKey(curve, readerWrap{rand.Reader})))
		k, err := ecdsa.GenerateKey(curve, rand.Reader)
		if err != nil {
			t.Fatal(err)
		}

		expectErrIfCustomRand(t, errRet2(k.Sign(readerWrap{rand.Reader}, make([]byte, 32), nil)))
		expectErrIfCustomRand(t, errRet2(ecdsa.SignASN1(readerWrap{rand.Reader}, k, make([]byte, 32))))
		expectErrIfCustomRand(t, errRet3(ecdsa.Sign(readerWrap{rand.Reader}, k, make([]byte, 32))))
		expectNoErr(t, errRet2(k.Sign(rand.Reader, make([]byte, 32), nil)))
		expectNoErr(t, errRet2(ecdsa.SignASN1(rand.Reader, k, make([]byte, 32))))
		expectNoErr(t, errRet3(ecdsa.Sign(rand.Reader, k, make([]byte, 32))))

		withNonApprovedHash(func(h crypto.Hash) {
			expectErr(t, errRet2(k.Sign(nil, make([]byte, h.Size()), h)))
		})
		withApprovedHash(func(h crypto.Hash) {
			expectNoErr(t, errRet2(k.Sign(nil, make([]byte, h.Size()), h)))
		})
	}
	customCurve := &elliptic.CurveParams{Name: "custom", P: big.NewInt(1)}
	expectErr(t, errRet2(ecdsa.GenerateKey(customCurve, rand.Reader)))

	_, ed25519Key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	expectNoErr(t, errRet2(ed25519Key.Sign(nil, make([]byte, 32), crypto.Hash(0))))
	expectNoErr(t, errRet2(ed25519Key.Sign(nil, make([]byte, 64), crypto.SHA512)))
	// ed25519ctx is not allowed (but ed25519ph with context is).
	expectErr(t, errRet2(ed25519Key.Sign(nil, make([]byte, 32), &ed25519.Options{
		Context: "test",
	})))
	expectNoErr(t, errRet2(ed25519Key.Sign(nil, make([]byte, 64), &ed25519.Options{
		Hash: crypto.SHA512, Context: "test",
	})))
	expectNoErr(t, errRet2(ed25519Key.Sign(nil, make([]byte, 64), &ed25519.Options{
		Hash: crypto.SHA512,
	})))

	expectErr(t, errRet2(rsa.GenerateMultiPrimeKey(rand.Reader, 3, 2048)))
	expectErr(t, errRet2(rsa.GenerateKey(rand.Reader, 1024)))
	expectErr(t, errRet2(rsa.GenerateKey(rand.Reader, 2049)))
	expectErrIfCustomRand(t, errRet2(rsa.GenerateKey(readerWrap{rand.Reader}, 2048)))
	rsaKey, err := rsa.GenerateKey(rand.Reader, 2048)
	expectNoErr(t, err)

	smallKey := parseKey(testingKey(`-----BEGIN RSA TESTING KEY-----
MIICXQIBAAKBgQDMrln6XoAa3Rjts+kRi5obbP86qSf/562RcuDO+yMXeTLHfi4M
8ubyhoFY+UKBCGBLmmTO7ikbvQgdipkT3xVkU8nM3XTW4sxrnw0X5QXsl4PGlMo0
5UufxYyQxe7bbjuwFz2XnN6Jz4orpOfO0s36/KVHj9lZRl+REpr/Jy+nJQIDAQAB
AoGAJ9WEwGO01cWSzOwXH2mGX/EKCQ4TsUuS7XwogU/B6BcXyVhmuPFq/ecsdDbq
ePc62mvdU6JpELNsyWcIXKQtYsRgJHxNS+KJkCQIq6YeiAWRG0XL6q+qVj+HtT8a
1Qrmul9ZBd23Y9wLF8pg/xWDQYvb8DPAb/xJ0e/KEBZcWU8CQQDXFCFCGpCfwyxY
Cq8G/3B94D9UYwk5mK6jRIH5m8LbaX9bKKetf8+If8TWVgeuiRjjN4WEQ78lPoSg
3Fsz2qs3AkEA85/JCudNUf2FnY+T6h1c/2SWekZiZ1NS4lCh/C7iYuAN3oa8zGkf
gjjR5e0+Z8rUAcZkTukxyLLaNqy6rs9GgwJAVR6pXvEGhcQHe7yWso1LpvWl+q7L
StkrXIBTdEb54j4pYhl/6wFnUB1I+I7JsYCeseYaWFM7hfDtKoCrM6V6FwJBANxh
KmfmnJcSkw/YlaEuNrYAs+6gRNvbEBsRfba2Yqu2qlUl5Ruz7IDMDXPEjLMvU2DX
ql2HrTU0NRlIXwdLESkCQQDGJ54H6WK1eE1YvtxCaLm28zmogcFlvc21pym+PpM1
bXVL8iKLrG91IYQByUHZIn3WVAd2bfi4MfKagRt0ggd4
-----END RSA TESTING KEY-----`))

	expectNoErr(t, errRet2(rsaKey.Sign(rand.Reader, make([]byte, 32), crypto.SHA256)))
	expectErr(t, errRet2(smallKey.Sign(rand.Reader, make([]byte, 32), crypto.SHA256)))
	expectErr(t, errRet2(rsaKey.Sign(rand.Reader, make([]byte, 20), crypto.SHA1)))
	// rand is always ignored for PKCS1v15 signing
	expectNoErr(t, errRet2(rsaKey.Sign(readerWrap{rand.Reader}, make([]byte, 32), crypto.SHA256)))

	sigPKCS1v15, err := rsa.SignPKCS1v15(rand.Reader, rsaKey, crypto.SHA256, make([]byte, 32))
	expectNoErr(t, err)
	expectErr(t, errRet2(rsa.SignPKCS1v15(rand.Reader, smallKey, crypto.SHA256, make([]byte, 32))))
	expectErr(t, errRet2(rsa.SignPKCS1v15(rand.Reader, rsaKey, crypto.SHA1, make([]byte, 20))))
	// rand is always ignored for PKCS1v15 signing
	expectNoErr(t, errRet2(rsa.SignPKCS1v15(readerWrap{rand.Reader}, rsaKey, crypto.SHA256, make([]byte, 32))))

	expectNoErr(t, rsa.VerifyPKCS1v15(&rsaKey.PublicKey, crypto.SHA256, make([]byte, 32), sigPKCS1v15))
	expectErr(t, rsa.VerifyPKCS1v15(&smallKey.PublicKey, crypto.SHA256, make([]byte, 32), sigPKCS1v15))
	expectErr(t, rsa.VerifyPKCS1v15(&rsaKey.PublicKey, crypto.SHA1, make([]byte, 20), sigPKCS1v15))

	sigPSS, err := rsa.SignPSS(rand.Reader, rsaKey, crypto.SHA256, make([]byte, 32), nil)
	expectNoErr(t, err)
	expectErr(t, errRet2(rsa.SignPSS(rand.Reader, smallKey, crypto.SHA256, make([]byte, 32), nil)))
	expectErr(t, errRet2(rsa.SignPSS(rand.Reader, rsaKey, crypto.SHA1, make([]byte, 20), nil)))
	expectErr(t, errRet2(rsa.SignPSS(readerWrap{rand.Reader}, rsaKey, crypto.SHA256, make([]byte, 32), nil)))

	expectNoErr(t, rsa.VerifyPSS(&rsaKey.PublicKey, crypto.SHA256, make([]byte, 32), sigPSS, nil))
	expectErr(t, rsa.VerifyPSS(&smallKey.PublicKey, crypto.SHA256, make([]byte, 32), sigPSS, nil))
	expectErr(t, rsa.VerifyPSS(&rsaKey.PublicKey, crypto.SHA1, make([]byte, 20), sigPSS, nil))

	k, err := mlkem.GenerateKey768()
	expectNoErr(t, err)
	expectErr(t, errRet3(mlkemtest.Encapsulate768(k.EncapsulationKey(), make([]byte, 32))))
	k1024, err := mlkem.GenerateKey1024()
	expectNoErr(t, err)
	expectErr(t, errRet3(mlkemtest.Encapsulate1024(k1024.EncapsulationKey(), make([]byte, 32))))

	for _, kem := range []hpke.KEM{
		hpke.DHKEM(ecdh.P256()),
		hpke.DHKEM(ecdh.P384()),
		hpke.DHKEM(ecdh.P521()),
		hpke.MLKEM768(),
		hpke.MLKEM1024(),
		hpke.MLKEM768P256(),
		hpke.MLKEM1024P384(),
		hpke.MLKEM768X25519(), // allowed as hybrid
	} {
		t.Run(fmt.Sprintf("HKPE KEM %04x", kem.ID()), func(t *testing.T) {
			k, err := kem.GenerateKey()
			expectNoErr(t, err)
			expectNoErr(t, errRet2(kem.DeriveKeyPair(make([]byte, 64))))
			kb, err := k.Bytes()
			expectNoErr(t, err)
			expectNoErr(t, errRet2(kem.NewPrivateKey(kb)))
			expectNoErr(t, errRet2(kem.NewPublicKey(k.PublicKey().Bytes())))
			if fips140.Version() == "v1.0.0" {
				t.Skip("FIPS 140-3 Module v1.0.0 does not provide HPKE GCM modes")
			}
			c, err := hpke.Seal(k.PublicKey(), hpke.HKDFSHA256(), hpke.AES128GCM(), nil, nil)
			expectNoErr(t, err)
			_, err = hpke.Open(k, hpke.HKDFSHA256(), hpke.AES128GCM(), nil, c)
			expectNoErr(t, err)
		})
	}
	expectErr(t, errRet2(hpke.DHKEM(ecdh.X25519()).GenerateKey()))
	expectErr(t, errRet2(hpke.DHKEM(ecdh.X25519()).DeriveKeyPair(make([]byte, 64))))
	expectErr(t, errRet2(hpke.DHKEM(ecdh.X25519()).NewPrivateKey(make([]byte, 32))))
	expectErr(t, errRet2(hpke.DHKEM(ecdh.X25519()).NewPublicKey(make([]byte, 32))))
	hpkeK, err := hpke.MLKEM768().GenerateKey()
	expectNoErr(t, err)
	expectErr(t, errRet2(hpke.Seal(hpkeK.PublicKey(), hpke.HKDFSHA256(), hpke.ChaCha20Poly1305(), nil, nil)))
	expectErr(t, errRet2(hpke.Open(hpkeK, hpke.HKDFSHA256(), hpke.ChaCha20Poly1305(), nil, make([]byte, 2000))))

	// fips140=only mode should prevent any operation that would make the FIPS
	// 140-3 module set its service indicator to false.
	if !fips140.ServiceIndicator() {
		t.Errorf("service indicator not set")
	}
}

type blockWrap struct {
	cipher.Block
}

type readerWrap struct {
	io.Reader
}

func withApprovedHash(f func(crypto.Hash)) {
	f(crypto.SHA224)
	f(crypto.SHA256)
	f(crypto.SHA384)
	f(crypto.SHA512)
	f(crypto.SHA3_224)
	f(crypto.SHA3_256)
	f(crypto.SHA3_384)
	f(crypto.SHA3_512)
	f(crypto.SHA512_224)
	f(crypto.SHA512_256)
}

func withNonApprovedHash(f func(crypto.Hash)) {
	f(crypto.MD5)
	f(crypto.SHA1)
}

func expectPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		if err := recover(); err == nil {
			t.Errorf("expected panic")
		} else {
			if s, ok := err.(string); !ok || !strings.Contains(s, "FIPS 140-only") {
				t.Errorf("unexpected panic: %v", err)
			}
		}
	}()
	f()
}

var cryptocustomrand = godebug.New("cryptocustomrand")

func expectErr(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Errorf("expected error")
	} else if !strings.Contains(err.Error(), "FIPS 140-only") {
		t.Errorf("unexpected error: %v", err)
	}
}

func expectNoErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func expectErrIfCustomRand(t *testing.T, err error) {
	t.Helper()
	if cryptocustomrand.Value() == "1" {
		expectErr(t, err)
	} else {
		expectNoErr(t, err)
	}
}

func errRet2[T any](_ T, err error) error {
	return err
}

func errRet3[T any](_, _ T, err error) error {
	return err
}

func testingKey(s string) string { return strings.ReplaceAll(s, "TESTING KEY", "PRIVATE KEY") }

func parseKey(s string) *rsa.PrivateKey {
	p, _ := pem.Decode([]byte(s))
	k, err := x509.ParsePKCS1PrivateKey(p.Bytes)
	if err != nil {
		panic(err)
	}
	return k
}
