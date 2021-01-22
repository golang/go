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
	"encoding/asn1"
	"math/big"
	"runtime"
	"unsafe"
)

type GoECKey = C.GO_EC_KEY

type ecdsa struct{}

type ecdsaSignature struct {
	R, S *big.Int
}

var errUnknownCurve = fail("unknown elliptic curve")
var errUnsupportedCurve = "unsupported elliptic curve"

func curveNID(curve string) (C.int, error) {
	switch curve {
	case "P-224":
		panicIfStrictFIPS(errUnsupportedCurve)
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

func (_ ecdsa) NewPublicKeyECDSA(curve string, X, Y *big.Int) (*GoECKey, error) {
	return newECKey(curve, X, Y)
}

func (_ ecdsa) ECKeyFree(key *GoECKey) {
	C._goboringcrypto_EC_KEY_free(key)
}

func newECKey(curve string, X, Y *big.Int) (*GoECKey, error) {
	nid, err := curveNID(curve)
	if err != nil {
		return nil, err
	}
	key := C._goboringcrypto_EC_KEY_new_by_curve_name(nid)
	if key == nil {
		return nil, newOpenSSLError("EC_KEY_new_by_curve_name failed")
	}
	group := C._goboringcrypto_EC_KEY_get0_group(key)
	pt := C._goboringcrypto_EC_POINT_new(group)
	if pt == nil {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, newOpenSSLError("EC_POINT_new failed")
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
		return nil, newOpenSSLError("EC_POINT_free failed")
	}
	return key, nil
}

func (_ ecdsa) NewPrivateKeyECDSA(curve string, X, Y *big.Int, D *big.Int) (*GoECKey, error) {
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
		return nil, newOpenSSLError("EC_KEY_set_private_key failed")
	}
	return key, nil
}

func (e ecdsa) SignECDSA(priv *GoECKey, hash []byte) (r, s *big.Int, err error) {
	// We could use ECDSA_do_sign instead but would need to convert
	// the resulting BIGNUMs to *big.Int form. If we're going to do a
	// conversion, converting the ASN.1 form is more convenient and
	// likely not much more expensive.
	sig, err := e.SignMarshalECDSA(priv, hash)
	if err != nil {
		return nil, nil, err
	}
	var esig ecdsaSignature
	if _, err := asn1.Unmarshal(sig, &esig); err != nil {
		return nil, nil, err
	}
	return esig.R, esig.S, nil
}

func (_ ecdsa) SignMarshalECDSA(key *GoECKey, hash []byte) ([]byte, error) {
	size := C._goboringcrypto_ECDSA_size(key)
	sig := make([]byte, size)
	var sigLen C.uint
	ok := C._goboringcrypto_internal_ECDSA_sign(0, base(hash), C.size_t(len(hash)), (*C.uint8_t)(unsafe.Pointer(&sig[0])), &sigLen, key) > 0
	if !ok {
		return nil, newOpenSSLError(("ECDSA_sign failed"))
	}
	runtime.KeepAlive(key)
	return sig[:sigLen], nil
}

func (_ ecdsa) VerifyECDSA(key *GoECKey, msg []byte, r, s *big.Int) bool {
	// We could use ECDSA_do_verify instead but would need to convert
	// r and s to BIGNUM form. If we're going to do a conversion, marshaling
	// to ASN.1 is more convenient and likely not much more expensive.
	sig, err := asn1.Marshal(ecdsaSignature{r, s})
	if err != nil {
		return false
	}
	ret := C._goboringcrypto_internal_ECDSA_verify(0, base(msg), C.size_t(len(msg)),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])), C.uint(len(sig)), key)

	runtime.KeepAlive(key)
	return ret > 0
}

func (_ ecdsa) GenerateKeyECDSA(curve string) (X, Y, D *big.Int, err error) {
	nid, err := curveNID(curve)
	if err != nil {
		return nil, nil, nil, err
	}
	key := C._goboringcrypto_EC_KEY_new_by_curve_name(nid)
	if key == nil {
		return nil, nil, nil, newOpenSSLError("EC_KEY_new_by_curve_name failed")
	}
	defer C._goboringcrypto_EC_KEY_free(key)
	if C._goboringcrypto_EC_KEY_generate_key(key) == 0 {
		return nil, nil, nil, newOpenSSLError("EC_KEY_generate_key failed")
	}
	group := C._goboringcrypto_EC_KEY_get0_group(key)
	pt := C._goboringcrypto_EC_KEY_get0_public_key(key)
	bd := C._goboringcrypto_EC_KEY_get0_private_key(key)
	if pt == nil || bd == nil {
		return nil, nil, nil, newOpenSSLError("EC_KEY_get0_private_key failed")
	}
	bx := C._goboringcrypto_BN_new()
	if bx == nil {
		return nil, nil, nil, newOpenSSLError("BN_new failed")
	}
	defer C._goboringcrypto_BN_free(bx)
	by := C._goboringcrypto_BN_new()
	if by == nil {
		return nil, nil, nil, newOpenSSLError("BN_new failed")
	}
	defer C._goboringcrypto_BN_free(by)
	if C._goboringcrypto_EC_POINT_get_affine_coordinates_GFp(group, pt, bx, by, nil) == 0 {
		return nil, nil, nil, newOpenSSLError("EC_POINT_get_affine_coordinates_GFp failed")
	}
	return bnToBig(bx), bnToBig(by), bnToBig(bd), nil
}
