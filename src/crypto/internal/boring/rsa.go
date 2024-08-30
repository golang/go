// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && (amd64 || arm64) && !android && !msan

package boring

// #include "goboringcrypto.h"
import "C"
import (
	"crypto"
	"crypto/subtle"
	"errors"
	"hash"
	"runtime"
	"strconv"
	"unsafe"
)

func GenerateKeyRSA(bits int) (N, E, D, P, Q, Dp, Dq, Qinv BigInt, err error) {
	bad := func(e error) (N, E, D, P, Q, Dp, Dq, Qinv BigInt, err error) {
		return nil, nil, nil, nil, nil, nil, nil, nil, e
	}

	key := C._goboringcrypto_RSA_new()
	if key == nil {
		return bad(fail("RSA_new"))
	}
	defer C._goboringcrypto_RSA_free(key)

	if C._goboringcrypto_RSA_generate_key_fips(key, C.int(bits), nil) == 0 {
		return bad(fail("RSA_generate_key_fips"))
	}

	var n, e, d, p, q, dp, dq, qinv *C.GO_BIGNUM
	C._goboringcrypto_RSA_get0_key(key, &n, &e, &d)
	C._goboringcrypto_RSA_get0_factors(key, &p, &q)
	C._goboringcrypto_RSA_get0_crt_params(key, &dp, &dq, &qinv)
	return bnToBig(n), bnToBig(e), bnToBig(d), bnToBig(p), bnToBig(q), bnToBig(dp), bnToBig(dq), bnToBig(qinv), nil
}

type PublicKeyRSA struct {
	// _key MUST NOT be accessed directly. Instead, use the withKey method.
	_key *C.GO_RSA
}

func NewPublicKeyRSA(N, E BigInt) (*PublicKeyRSA, error) {
	key := C._goboringcrypto_RSA_new()
	if key == nil {
		return nil, fail("RSA_new")
	}
	if !bigToBn(&key.n, N) ||
		!bigToBn(&key.e, E) {
		return nil, fail("BN_bin2bn")
	}
	k := &PublicKeyRSA{_key: key}
	runtime.SetFinalizer(k, (*PublicKeyRSA).finalize)
	return k, nil
}

func (k *PublicKeyRSA) finalize() {
	C._goboringcrypto_RSA_free(k._key)
}

func (k *PublicKeyRSA) withKey(f func(*C.GO_RSA) C.int) C.int {
	// Because of the finalizer, any time _key is passed to cgo, that call must
	// be followed by a call to runtime.KeepAlive, to make sure k is not
	// collected (and finalized) before the cgo call returns.
	defer runtime.KeepAlive(k)
	return f(k._key)
}

type PrivateKeyRSA struct {
	// _key MUST NOT be accessed directly. Instead, use the withKey method.
	_key *C.GO_RSA
}

func NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv BigInt) (*PrivateKeyRSA, error) {
	key := C._goboringcrypto_RSA_new()
	if key == nil {
		return nil, fail("RSA_new")
	}
	if !bigToBn(&key.n, N) ||
		!bigToBn(&key.e, E) ||
		!bigToBn(&key.d, D) ||
		!bigToBn(&key.p, P) ||
		!bigToBn(&key.q, Q) ||
		!bigToBn(&key.dmp1, Dp) ||
		!bigToBn(&key.dmq1, Dq) ||
		!bigToBn(&key.iqmp, Qinv) {
		return nil, fail("BN_bin2bn")
	}
	k := &PrivateKeyRSA{_key: key}
	runtime.SetFinalizer(k, (*PrivateKeyRSA).finalize)
	return k, nil
}

func (k *PrivateKeyRSA) finalize() {
	C._goboringcrypto_RSA_free(k._key)
}

func (k *PrivateKeyRSA) withKey(f func(*C.GO_RSA) C.int) C.int {
	// Because of the finalizer, any time _key is passed to cgo, that call must
	// be followed by a call to runtime.KeepAlive, to make sure k is not
	// collected (and finalized) before the cgo call returns.
	defer runtime.KeepAlive(k)
	return f(k._key)
}

func setupRSA(withKey func(func(*C.GO_RSA) C.int) C.int,
	padding C.int, h, mgfHash hash.Hash, label []byte, saltLen int, ch crypto.Hash,
	init func(*C.GO_EVP_PKEY_CTX) C.int) (pkey *C.GO_EVP_PKEY, ctx *C.GO_EVP_PKEY_CTX, err error) {
	defer func() {
		if err != nil {
			if pkey != nil {
				C._goboringcrypto_EVP_PKEY_free(pkey)
				pkey = nil
			}
			if ctx != nil {
				C._goboringcrypto_EVP_PKEY_CTX_free(ctx)
				ctx = nil
			}
		}
	}()

	pkey = C._goboringcrypto_EVP_PKEY_new()
	if pkey == nil {
		return pkey, ctx, fail("EVP_PKEY_new")
	}
	if withKey(func(key *C.GO_RSA) C.int {
		return C._goboringcrypto_EVP_PKEY_set1_RSA(pkey, key)
	}) == 0 {
		return pkey, ctx, fail("EVP_PKEY_set1_RSA")
	}
	ctx = C._goboringcrypto_EVP_PKEY_CTX_new(pkey, nil)
	if ctx == nil {
		return pkey, ctx, fail("EVP_PKEY_CTX_new")
	}
	if init(ctx) == 0 {
		return pkey, ctx, fail("EVP_PKEY_operation_init")
	}
	if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_padding(ctx, padding) == 0 {
		return pkey, ctx, fail("EVP_PKEY_CTX_set_rsa_padding")
	}
	if padding == C.GO_RSA_PKCS1_OAEP_PADDING {
		md := hashToMD(h)
		if md == nil {
			return pkey, ctx, errors.New("crypto/rsa: unsupported hash function")
		}
		mgfMD := hashToMD(mgfHash)
		if mgfMD == nil {
			return pkey, ctx, errors.New("crypto/rsa: unsupported hash function")
		}
		if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_oaep_md(ctx, md) == 0 {
			return pkey, ctx, fail("EVP_PKEY_set_rsa_oaep_md")
		}
		if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_mgf1_md(ctx, mgfMD) == 0 {
			return pkey, ctx, fail("EVP_PKEY_set_rsa_mgf1_md")
		}
		// ctx takes ownership of label, so malloc a copy for BoringCrypto to free.
		clabel := (*C.uint8_t)(C._goboringcrypto_OPENSSL_malloc(C.size_t(len(label))))
		if clabel == nil {
			return pkey, ctx, fail("OPENSSL_malloc")
		}
		copy((*[1 << 30]byte)(unsafe.Pointer(clabel))[:len(label)], label)
		if C._goboringcrypto_EVP_PKEY_CTX_set0_rsa_oaep_label(ctx, clabel, C.size_t(len(label))) == 0 {
			return pkey, ctx, fail("EVP_PKEY_CTX_set0_rsa_oaep_label")
		}
	}
	if padding == C.GO_RSA_PKCS1_PSS_PADDING {
		if saltLen != 0 {
			if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_pss_saltlen(ctx, C.int(saltLen)) == 0 {
				return pkey, ctx, fail("EVP_PKEY_set_rsa_pss_saltlen")
			}
		}
		md := cryptoHashToMD(ch)
		if md == nil {
			return pkey, ctx, errors.New("crypto/rsa: unsupported hash function")
		}
		if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_mgf1_md(ctx, md) == 0 {
			return pkey, ctx, fail("EVP_PKEY_set_rsa_mgf1_md")
		}
	}

	return pkey, ctx, nil
}

func cryptRSA(withKey func(func(*C.GO_RSA) C.int) C.int,
	padding C.int, h, mgfHash hash.Hash, label []byte, saltLen int, ch crypto.Hash,
	init func(*C.GO_EVP_PKEY_CTX) C.int,
	crypt func(*C.GO_EVP_PKEY_CTX, *C.uint8_t, *C.size_t, *C.uint8_t, C.size_t) C.int,
	in []byte) ([]byte, error) {

	pkey, ctx, err := setupRSA(withKey, padding, h, mgfHash, label, saltLen, ch, init)
	if err != nil {
		return nil, err
	}
	defer C._goboringcrypto_EVP_PKEY_free(pkey)
	defer C._goboringcrypto_EVP_PKEY_CTX_free(ctx)

	var outLen C.size_t
	if crypt(ctx, nil, &outLen, base(in), C.size_t(len(in))) == 0 {
		return nil, fail("EVP_PKEY_decrypt/encrypt")
	}
	out := make([]byte, outLen)
	if crypt(ctx, base(out), &outLen, base(in), C.size_t(len(in))) == 0 {
		return nil, fail("EVP_PKEY_decrypt/encrypt")
	}
	return out[:outLen], nil
}

func DecryptRSAOAEP(h, mgfHash hash.Hash, priv *PrivateKeyRSA, ciphertext, label []byte) ([]byte, error) {
	return cryptRSA(priv.withKey, C.GO_RSA_PKCS1_OAEP_PADDING, h, mgfHash, label, 0, 0, decryptInit, decrypt, ciphertext)
}

func EncryptRSAOAEP(h, mgfHash hash.Hash, pub *PublicKeyRSA, msg, label []byte) ([]byte, error) {
	return cryptRSA(pub.withKey, C.GO_RSA_PKCS1_OAEP_PADDING, h, mgfHash, label, 0, 0, encryptInit, encrypt, msg)
}

func DecryptRSAPKCS1(priv *PrivateKeyRSA, ciphertext []byte) ([]byte, error) {
	return cryptRSA(priv.withKey, C.GO_RSA_PKCS1_PADDING, nil, nil, nil, 0, 0, decryptInit, decrypt, ciphertext)
}

func EncryptRSAPKCS1(pub *PublicKeyRSA, msg []byte) ([]byte, error) {
	return cryptRSA(pub.withKey, C.GO_RSA_PKCS1_PADDING, nil, nil, nil, 0, 0, encryptInit, encrypt, msg)
}

func DecryptRSANoPadding(priv *PrivateKeyRSA, ciphertext []byte) ([]byte, error) {
	return cryptRSA(priv.withKey, C.GO_RSA_NO_PADDING, nil, nil, nil, 0, 0, decryptInit, decrypt, ciphertext)
}

func EncryptRSANoPadding(pub *PublicKeyRSA, msg []byte) ([]byte, error) {
	return cryptRSA(pub.withKey, C.GO_RSA_NO_PADDING, nil, nil, nil, 0, 0, encryptInit, encrypt, msg)
}

// These dumb wrappers work around the fact that cgo functions cannot be used as values directly.

func decryptInit(ctx *C.GO_EVP_PKEY_CTX) C.int {
	return C._goboringcrypto_EVP_PKEY_decrypt_init(ctx)
}

func decrypt(ctx *C.GO_EVP_PKEY_CTX, out *C.uint8_t, outLen *C.size_t, in *C.uint8_t, inLen C.size_t) C.int {
	return C._goboringcrypto_EVP_PKEY_decrypt(ctx, out, outLen, in, inLen)
}

func encryptInit(ctx *C.GO_EVP_PKEY_CTX) C.int {
	return C._goboringcrypto_EVP_PKEY_encrypt_init(ctx)
}

func encrypt(ctx *C.GO_EVP_PKEY_CTX, out *C.uint8_t, outLen *C.size_t, in *C.uint8_t, inLen C.size_t) C.int {
	return C._goboringcrypto_EVP_PKEY_encrypt(ctx, out, outLen, in, inLen)
}

var invalidSaltLenErr = errors.New("crypto/rsa: PSSOptions.SaltLength cannot be negative")

func SignRSAPSS(priv *PrivateKeyRSA, h crypto.Hash, hashed []byte, saltLen int) ([]byte, error) {
	md := cryptoHashToMD(h)
	if md == nil {
		return nil, errors.New("crypto/rsa: unsupported hash function")
	}

	// A salt length of -2 is valid in BoringSSL, but not in crypto/rsa, so reject
	// it, and lengths < -2, before we convert to the BoringSSL sentinel values.
	if saltLen <= -2 {
		return nil, invalidSaltLenErr
	}

	// BoringSSL uses sentinel salt length values like we do, but the values don't
	// fully match what we use. We both use -1 for salt length equal to hash length,
	// but BoringSSL uses -2 to mean maximal size where we use 0. In the latter
	// case convert to the BoringSSL version.
	if saltLen == 0 {
		saltLen = -2
	}

	var out []byte
	var outLen C.size_t
	if priv.withKey(func(key *C.GO_RSA) C.int {
		out = make([]byte, C._goboringcrypto_RSA_size(key))
		return C._goboringcrypto_RSA_sign_pss_mgf1(key, &outLen, base(out), C.size_t(len(out)),
			base(hashed), C.size_t(len(hashed)), md, nil, C.int(saltLen))
	}) == 0 {
		return nil, fail("RSA_sign_pss_mgf1")
	}

	return out[:outLen], nil
}

func VerifyRSAPSS(pub *PublicKeyRSA, h crypto.Hash, hashed, sig []byte, saltLen int) error {
	md := cryptoHashToMD(h)
	if md == nil {
		return errors.New("crypto/rsa: unsupported hash function")
	}

	// A salt length of -2 is valid in BoringSSL, but not in crypto/rsa, so reject
	// it, and lengths < -2, before we convert to the BoringSSL sentinel values.
	if saltLen <= -2 {
		return invalidSaltLenErr
	}

	// BoringSSL uses sentinel salt length values like we do, but the values don't
	// fully match what we use. We both use -1 for salt length equal to hash length,
	// but BoringSSL uses -2 to mean maximal size where we use 0. In the latter
	// case convert to the BoringSSL version.
	if saltLen == 0 {
		saltLen = -2
	}

	if pub.withKey(func(key *C.GO_RSA) C.int {
		return C._goboringcrypto_RSA_verify_pss_mgf1(key, base(hashed), C.size_t(len(hashed)),
			md, nil, C.int(saltLen), base(sig), C.size_t(len(sig)))
	}) == 0 {
		return fail("RSA_verify_pss_mgf1")
	}
	return nil
}

func SignRSAPKCS1v15(priv *PrivateKeyRSA, h crypto.Hash, hashed []byte) ([]byte, error) {
	if h == 0 {
		// No hashing.
		var out []byte
		var outLen C.size_t
		if priv.withKey(func(key *C.GO_RSA) C.int {
			out = make([]byte, C._goboringcrypto_RSA_size(key))
			return C._goboringcrypto_RSA_sign_raw(key, &outLen, base(out), C.size_t(len(out)),
				base(hashed), C.size_t(len(hashed)), C.GO_RSA_PKCS1_PADDING)
		}) == 0 {
			return nil, fail("RSA_sign_raw")
		}
		return out[:outLen], nil
	}

	md := cryptoHashToMD(h)
	if md == nil {
		return nil, errors.New("crypto/rsa: unsupported hash function: " + strconv.Itoa(int(h)))
	}
	nid := C._goboringcrypto_EVP_MD_type(md)
	var out []byte
	var outLen C.uint
	if priv.withKey(func(key *C.GO_RSA) C.int {
		out = make([]byte, C._goboringcrypto_RSA_size(key))
		return C._goboringcrypto_RSA_sign(nid, base(hashed), C.uint(len(hashed)),
			base(out), &outLen, key)
	}) == 0 {
		return nil, fail("RSA_sign")
	}
	return out[:outLen], nil
}

func VerifyRSAPKCS1v15(pub *PublicKeyRSA, h crypto.Hash, hashed, sig []byte) error {
	if h == 0 {
		var out []byte
		var outLen C.size_t
		if pub.withKey(func(key *C.GO_RSA) C.int {
			out = make([]byte, C._goboringcrypto_RSA_size(key))
			return C._goboringcrypto_RSA_verify_raw(key, &outLen, base(out),
				C.size_t(len(out)), base(sig), C.size_t(len(sig)), C.GO_RSA_PKCS1_PADDING)
		}) == 0 {
			return fail("RSA_verify")
		}
		if subtle.ConstantTimeCompare(hashed, out[:outLen]) != 1 {
			return fail("RSA_verify")
		}
		return nil
	}
	md := cryptoHashToMD(h)
	if md == nil {
		return errors.New("crypto/rsa: unsupported hash function")
	}
	nid := C._goboringcrypto_EVP_MD_type(md)
	if pub.withKey(func(key *C.GO_RSA) C.int {
		return C._goboringcrypto_RSA_verify(nid, base(hashed), C.size_t(len(hashed)),
			base(sig), C.size_t(len(sig)), key)
	}) == 0 {
		return fail("RSA_verify")
	}
	return nil
}
