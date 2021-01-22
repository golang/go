// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build openssl
// +build !android
// +build !no_openssl
// +build !cmd_go_bootstrap
// +build !msan

package openssl

// #include "goopenssl.h"
import "C"
import (
	"crypto"
	"crypto/subtle"
	"errors"
	"hash"
	"math/big"
	"strconv"
	"unsafe"
)

type GoRSA = C.GO_RSA

type rsa struct{}

func (_ rsa) GenerateKeyRSA(bits int) (N, E, D, P, Q, Dp, Dq, Qinv *big.Int, err error) {
	bad := func(e error) (N, E, D, P, Q, Dp, Dq, Qinv *big.Int, err error) {
		return nil, nil, nil, nil, nil, nil, nil, nil, e
	}

	key := C._goboringcrypto_RSA_new()
	if key == nil {
		return bad(newOpenSSLError("RSA_new failed"))
	}
	defer C._goboringcrypto_RSA_free(key)

	if C._goboringcrypto_RSA_generate_key_fips(key, C.int(bits), nil) == 0 {
		return bad(newOpenSSLError("RSA_generate_key_fips failed"))
	}

	var n, e, d, p, q, dp, dq, qinv *C.GO_BIGNUM
	C._goboringcrypto_RSA_get0_key(key, &n, &e, &d)
	C._goboringcrypto_RSA_get0_factors(key, &p, &q)
	C._goboringcrypto_RSA_get0_crt_params(key, &dp, &dq, &qinv)
	return bnToBig(n), bnToBig(e), bnToBig(d), bnToBig(p), bnToBig(q), bnToBig(dp), bnToBig(dq), bnToBig(qinv), nil
}

func (_ rsa) NewPublicKeyRSA(N, E *big.Int) (*GoRSA, error) {
	key := C._goboringcrypto_RSA_new()
	if key == nil {
		return nil, newOpenSSLError("RSA_new failed")
	}
	var n, e *C.GO_BIGNUM
	if !bigToBn(&n, N) ||
		!bigToBn(&e, E) {
		return nil, fail("BN_bin2bn")
	}
	C._goboringcrypto_RSA_set0_key(key, n, e, nil)
	return key, nil
}

func (_ rsa) NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv *big.Int) (*GoRSA, error) {
	key := C._goboringcrypto_RSA_new()
	if key == nil {
		return nil, newOpenSSLError("RSA_new failed")
	}
	var n, e, d, p, q, dp, dq, qinv *C.GO_BIGNUM
	n = bigToBN(N)
	e = bigToBN(E)
	d = bigToBN(D)
	if C._goboringcrypto_RSA_set0_key(key, n, e, d) == 0 {
		return nil, newOpenSSLError("RSA_set0_key")
	}
	if P != nil && Q != nil {
		p = bigToBN(P)
		q = bigToBN(Q)
		if C._goboringcrypto_RSA_set0_factors(key, p, q) == 0 {
			return nil, newOpenSSLError("RSA_set0_factors")
		}
	}
	if Dp != nil && Dq != nil && Qinv != nil {
		dp = bigToBN(Dp)
		dq = bigToBN(Dq)
		qinv = bigToBN(Qinv)
		if C._goboringcrypto_RSA_set0_crt_params(key, dp, dq, qinv) == 0 {
			return nil, newOpenSSLError("RSA_set0_crt_params")
		}
	}
	return key, nil
}

func (_ rsa) RSAFree(key *GoRSA) {
	C._goboringcrypto_RSA_free(key)
}

func setupRSA(key *GoRSA, padding C.int, h hash.Hash, label []byte, saltLen int, ch crypto.Hash,
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
		return nil, nil, newOpenSSLError("EVP_PKEY_new failed")
	}
	if C._goboringcrypto_EVP_PKEY_set1_RSA(pkey, key) == 0 {
		return nil, nil, fail("EVP_PKEY_set1_RSA")
	}
	ctx = C._goboringcrypto_EVP_PKEY_CTX_new(pkey, nil)
	if ctx == nil {
		return nil, nil, newOpenSSLError("EVP_PKEY_CTX_new failed")
	}
	if init(ctx) == 0 {
		return nil, nil, newOpenSSLError("EVP_PKEY_operation_init failed")
	}
	if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_padding(ctx, padding) == 0 {
		return nil, nil, newOpenSSLError("EVP_PKEY_CTX_set_rsa_padding failed")
	}
	if padding == C.GO_RSA_PKCS1_OAEP_PADDING {
		md := hashToMD(h)
		if md == nil {
			return nil, nil, errors.New("crypto/rsa: unsupported hash function")
		}
		if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_oaep_md(ctx, md) == 0 {
			return nil, nil, newOpenSSLError("EVP_PKEY_set_rsa_oaep_md failed")
		}
		// ctx takes ownership of label, so malloc a copy for BoringCrypto to free.
		clabel := (*C.uint8_t)(C.malloc(C.size_t(len(label))))
		if clabel == nil {
			return nil, nil, fail("OPENSSL_malloc")
		}
		copy((*[1 << 30]byte)(unsafe.Pointer(clabel))[:len(label)], label)
		if C._goboringcrypto_EVP_PKEY_CTX_set0_rsa_oaep_label(ctx, clabel, C.int(len(label))) == 0 {
			return nil, nil, newOpenSSLError("EVP_PKEY_CTX_set0_rsa_oaep_label failed")
		}
	}
	if padding == C.GO_RSA_PKCS1_PSS_PADDING {
		if saltLen != 0 {
			if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_pss_saltlen(ctx, C.int(saltLen)) == 0 {
				return nil, nil, newOpenSSLError("EVP_PKEY_set_rsa_pss_saltlen failed")
			}
		}
		md := cryptoHashToMD(ch)
		if md == nil {
			return nil, nil, errors.New("crypto/rsa: unsupported hash function")
		}
		if C._goboringcrypto_EVP_PKEY_CTX_set_rsa_mgf1_md(ctx, md) == 0 {
			return nil, nil, newOpenSSLError("EVP_PKEY_set_rsa_mgf1_md failed")
		}
	}

	return pkey, ctx, nil
}

func cryptRSA(key *GoRSA,
	padding C.int, h hash.Hash, label []byte, saltLen int, ch crypto.Hash,
	init func(*C.GO_EVP_PKEY_CTX) C.int,
	crypt func(*C.GO_EVP_PKEY_CTX, *C.uint8_t, *C.uint, *C.uint8_t, C.uint) C.int,
	in []byte) ([]byte, error) {

	pkey, ctx, err := setupRSA(key, padding, h, label, saltLen, ch, init)
	if err != nil {
		return nil, err
	}
	defer C._goboringcrypto_EVP_PKEY_free(pkey)
	defer C._goboringcrypto_EVP_PKEY_CTX_free(ctx)

	var outLen C.uint
	if crypt(ctx, nil, &outLen, base(in), C.uint(len(in))) == 0 {
		return nil, newOpenSSLError("EVP_PKEY_decrypt/encrypt failed")
	}
	out := make([]byte, outLen)
	if crypt(ctx, base(out), &outLen, base(in), C.uint(len(in))) == 0 {
		return nil, newOpenSSLError("EVP_PKEY_decrypt/encrypt failed")
	}
	return out[:outLen], nil
}

func (_ rsa) DecryptRSAOAEP(h hash.Hash, priv *GoRSA, ciphertext, label []byte) ([]byte, error) {
	return cryptRSA(priv, C.GO_RSA_PKCS1_OAEP_PADDING, h, label, 0, 0, decryptInit, decrypt, ciphertext)
}

func (_ rsa) EncryptRSAOAEP(h hash.Hash, pub *GoRSA, msg, label []byte) ([]byte, error) {
	return cryptRSA(pub, C.GO_RSA_PKCS1_OAEP_PADDING, h, label, 0, 0, encryptInit, encrypt, msg)
}

func (_ rsa) DecryptRSAPKCS1(priv *GoRSA, ciphertext []byte) ([]byte, error) {
	return cryptRSA(priv, C.GO_RSA_PKCS1_PADDING, nil, nil, 0, 0, decryptInit, decrypt, ciphertext)
}

func (_ rsa) EncryptRSAPKCS1(pub *GoRSA, msg []byte) ([]byte, error) {
	return cryptRSA(pub, C.GO_RSA_PKCS1_PADDING, nil, nil, 0, 0, encryptInit, encrypt, msg)
}

func (_ rsa) DecryptRSANoPadding(priv *GoRSA, ciphertext []byte) ([]byte, error) {
	return cryptRSA(priv, C.GO_RSA_NO_PADDING, nil, nil, 0, 0, decryptInit, decrypt, ciphertext)
}

func (_ rsa) EncryptRSANoPadding(pub *GoRSA, msg []byte) ([]byte, error) {
	return cryptRSA(pub, C.GO_RSA_NO_PADDING, nil, nil, 0, 0, encryptInit, encrypt, msg)
}

// These dumb wrappers work around the fact that cgo functions cannot be used as values directly.

func decryptInit(ctx *C.GO_EVP_PKEY_CTX) C.int {
	return C._goboringcrypto_EVP_PKEY_decrypt_init(ctx)
}

func decrypt(ctx *C.GO_EVP_PKEY_CTX, out *C.uint8_t, outLen *C.uint, in *C.uint8_t, inLen C.uint) C.int {
	return C._goboringcrypto_EVP_PKEY_decrypt(ctx, out, outLen, in, inLen)
}

func encryptInit(ctx *C.GO_EVP_PKEY_CTX) C.int {
	return C._goboringcrypto_EVP_PKEY_encrypt_init(ctx)
}

func encrypt(ctx *C.GO_EVP_PKEY_CTX, out *C.uint8_t, outLen *C.uint, in *C.uint8_t, inLen C.uint) C.int {
	return C._goboringcrypto_EVP_PKEY_encrypt(ctx, out, outLen, in, inLen)
}

func (_ rsa) SignRSAPSS(key *GoRSA, hashed []byte, h crypto.Hash, saltLen int) ([]byte, error) {
	md := cryptoHashToMD(h)
	if md == nil {
		return nil, errors.New("crypto/rsa: unsupported hash function")
	}
	if saltLen == 0 {
		saltLen = -1
	}
	var out []byte
	var outLen C.uint
	out = make([]byte, C._goboringcrypto_RSA_size(key))
	if C._goboringcrypto_RSA_sign_pss_mgf1(key, &outLen, base(out), C.uint(len(out)),
		base(hashed), C.uint(len(hashed)), md, nil, C.int(saltLen)) == 0 {
		return nil, newOpenSSLError("RSA_sign_pss_mgf1")
	}

	return out[:outLen], nil
}

func (_ rsa) VerifyRSAPSS(key *GoRSA, h crypto.Hash, hashed, sig []byte, saltLen int) error {
	md := cryptoHashToMD(h)
	if md == nil {
		return errors.New("crypto/rsa: unsupported hash function")
	}
	if saltLen == 0 {
		saltLen = -2 // auto-recover
	}
	if C._goboringcrypto_RSA_verify_pss_mgf1(key, base(hashed), C.uint(len(hashed)),
		md, nil, C.int(saltLen), base(sig), C.uint(len(sig))) == 0 {
		return newOpenSSLError("RSA_verify_pss_mgf1")
	}
	return nil
}

func (_ rsa) SignRSAPKCS1v15(key *GoRSA, h crypto.Hash, msg []byte) ([]byte, error) {
	if h == 0 {
		// No hashing.
		var out []byte
		out = make([]byte, C._goboringcrypto_RSA_size(key))
		outLen := C._goboringcrypto_RSA_private_encrypt(C.int(len(msg)), base(msg), base(out), key, C.GO_RSA_PKCS1_PADDING)
		if outLen == -1 {
			return nil, newOpenSSLError("EVP_RSA_sign")
		}
		return out[:outLen], nil
	}

	md := cryptoHashToMD(h)
	if md == nil {
		return nil, errors.New("crypto/rsa: unsupported hash function: " + strconv.Itoa(int(h)))
	}
	var out []byte
	var outLen C.uint
	nid := C._goboringcrypto_EVP_MD_type(md)
	out = make([]byte, C._goboringcrypto_RSA_size(key))
	if C._goboringcrypto_RSA_sign(nid, base(msg), C.uint(len(msg)), base(out), &outLen, key) == 0 {
		return nil, newOpenSSLError("RSA_sign")
	}
	return out[:outLen], nil
}

func (_ rsa) VerifyRSAPKCS1v15(key *GoRSA, h crypto.Hash, msg, sig []byte) error {
	if h == 0 {
		var out []byte
		keySize := C._goboringcrypto_RSA_size(key)
		out = make([]byte, keySize)
		outLen := C._goboringcrypto_RSA_public_decrypt(C.int(len(sig)), base(sig), base(out), key, C.GO_RSA_PKCS1_PADDING)
		if outLen == -1 {
			return newOpenSSLError("RSA_verify")
		}
		if subtle.ConstantTimeCompare(msg, out[:outLen]) != 1 {
			return newOpenSSLError("RSA_verify")
		}
		// Per RFC 8017, reject signatures which are not the same length as the RSA modulus.
		var n, e *C.BIGNUM
		C._goboringcrypto_RSA_get0_key(key, &n, &e, nil)
		mod := int(C._goboringcrypto_BN_num_bits(n))
		if len(sig) != (mod+7)/8 {
			return fail("RSA_verify")
		}
		return nil
	}
	md := cryptoHashToMD(h)
	if md == nil {
		return errors.New("crypto/rsa: unsupported hash function")
	}

	size := int(C._goboringcrypto_RSA_size(key))
	if len(sig) < size {
		return fail("signature length is less than expected")
	}

	nid := C._goboringcrypto_EVP_MD_type(md)
	if C._goboringcrypto_RSA_verify(nid, base(msg), C.uint(len(msg)), base(sig), C.uint(len(sig)), key) == 0 {
		return newOpenSSLError("RSA_verify")
	}
	return nil
}
