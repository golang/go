// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && (amd64 || arm64) && !android && !cmd_go_bootstrap && !msan

package boring

// #include "goboringcrypto.h"
import "C"
import (
	"errors"
	"runtime"
)

type ecdsaSignature struct {
	R, S BigInt
}

type PrivateKeyECDSA struct {
	key *C.GO_EC_KEY
}

func (k *PrivateKeyECDSA) finalize() {
	C._goboringcrypto_EC_KEY_free(k.key)
}

type PublicKeyECDSA struct {
	key *C.GO_EC_KEY
}

func (k *PublicKeyECDSA) finalize() {
	C._goboringcrypto_EC_KEY_free(k.key)
}

var errUnknownCurve = errors.New("boringcrypto: unknown elliptic curve")

func curveNID(curve string) (C.int, error) {
	switch curve {
	case "P-224":
		return C.GO_NID_secp224r1, nil
	case "P-256":
		return C.GO_NID_X9_62_prime256v1, nil
	case "P-384":
		return C.GO_NID_secp384r1, nil
	case "P-521":
		return C.GO_NID_secp521r1, nil
	}
	return 0, errUnknownCurve
}

func NewPublicKeyECDSA(curve string, X, Y BigInt) (*PublicKeyECDSA, error) {
	key, err := newECKey(curve, X, Y)
	if err != nil {
		return nil, err
	}
	k := &PublicKeyECDSA{key}
	// Note: Because of the finalizer, any time k.key is passed to cgo,
	// that call must be followed by a call to runtime.KeepAlive(k),
	// to make sure k is not collected (and finalized) before the cgo
	// call returns.
	runtime.SetFinalizer(k, (*PublicKeyECDSA).finalize)
	return k, nil
}

func newECKey(curve string, X, Y BigInt) (*C.GO_EC_KEY, error) {
	nid, err := curveNID(curve)
	if err != nil {
		return nil, err
	}
	key := C._goboringcrypto_EC_KEY_new_by_curve_name(nid)
	if key == nil {
		return nil, fail("EC_KEY_new_by_curve_name")
	}
	group := C._goboringcrypto_EC_KEY_get0_group(key)
	pt := C._goboringcrypto_EC_POINT_new(group)
	if pt == nil {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, fail("EC_POINT_new")
	}
	bx := bigToBN(X)
	by := bigToBN(Y)
	ok := bx != nil && by != nil && C._goboringcrypto_EC_POINT_set_affine_coordinates_GFp(group, pt, bx, by, nil) != 0 &&
		C._goboringcrypto_EC_KEY_set_public_key(key, pt) != 0
	if bx != nil {
		C._goboringcrypto_BN_free(bx)
	}
	if by != nil {
		C._goboringcrypto_BN_free(by)
	}
	C._goboringcrypto_EC_POINT_free(pt)
	if !ok {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, fail("EC_POINT_set_affine_coordinates_GFp")
	}
	return key, nil
}

func NewPrivateKeyECDSA(curve string, X, Y BigInt, D BigInt) (*PrivateKeyECDSA, error) {
	key, err := newECKey(curve, X, Y)
	if err != nil {
		return nil, err
	}
	bd := bigToBN(D)
	ok := bd != nil && C._goboringcrypto_EC_KEY_set_private_key(key, bd) != 0
	if bd != nil {
		C._goboringcrypto_BN_free(bd)
	}
	if !ok {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, fail("EC_KEY_set_private_key")
	}
	k := &PrivateKeyECDSA{key}
	// Note: Because of the finalizer, any time k.key is passed to cgo,
	// that call must be followed by a call to runtime.KeepAlive(k),
	// to make sure k is not collected (and finalized) before the cgo
	// call returns.
	runtime.SetFinalizer(k, (*PrivateKeyECDSA).finalize)
	return k, nil
}

func SignMarshalECDSA(priv *PrivateKeyECDSA, hash []byte) ([]byte, error) {
	size := C._goboringcrypto_ECDSA_size(priv.key)
	sig := make([]byte, size)
	var sigLen C.uint
	if C._goboringcrypto_ECDSA_sign(0, base(hash), C.size_t(len(hash)), base(sig), &sigLen, priv.key) == 0 {
		return nil, fail("ECDSA_sign")
	}
	runtime.KeepAlive(priv)
	return sig[:sigLen], nil
}

func VerifyECDSA(pub *PublicKeyECDSA, hash []byte, sig []byte) bool {
	ok := C._goboringcrypto_ECDSA_verify(0, base(hash), C.size_t(len(hash)), base(sig), C.size_t(len(sig)), pub.key) != 0
	runtime.KeepAlive(pub)
	return ok
}

func GenerateKeyECDSA(curve string) (X, Y, D BigInt, err error) {
	nid, err := curveNID(curve)
	if err != nil {
		return nil, nil, nil, err
	}
	key := C._goboringcrypto_EC_KEY_new_by_curve_name(nid)
	if key == nil {
		return nil, nil, nil, fail("EC_KEY_new_by_curve_name")
	}
	defer C._goboringcrypto_EC_KEY_free(key)
	if C._goboringcrypto_EC_KEY_generate_key_fips(key) == 0 {
		return nil, nil, nil, fail("EC_KEY_generate_key_fips")
	}
	group := C._goboringcrypto_EC_KEY_get0_group(key)
	pt := C._goboringcrypto_EC_KEY_get0_public_key(key)
	bd := C._goboringcrypto_EC_KEY_get0_private_key(key)
	if pt == nil || bd == nil {
		return nil, nil, nil, fail("EC_KEY_get0_private_key")
	}
	bx := C._goboringcrypto_BN_new()
	if bx == nil {
		return nil, nil, nil, fail("BN_new")
	}
	defer C._goboringcrypto_BN_free(bx)
	by := C._goboringcrypto_BN_new()
	if by == nil {
		return nil, nil, nil, fail("BN_new")
	}
	defer C._goboringcrypto_BN_free(by)
	if C._goboringcrypto_EC_POINT_get_affine_coordinates_GFp(group, pt, bx, by, nil) == 0 {
		return nil, nil, nil, fail("EC_POINT_get_affine_coordinates_GFp")
	}
	return bnToBig(bx), bnToBig(by), bnToBig(bd), nil
}
