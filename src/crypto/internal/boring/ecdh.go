// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && (amd64 || arm64) && !android && !msan

package boring

// #include "goboringcrypto.h"
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

type PublicKeyECDH struct {
	curve string
	key   *C.GO_EC_POINT
	group *C.GO_EC_GROUP
	bytes []byte
}

func (k *PublicKeyECDH) finalize() {
	C._goboringcrypto_EC_POINT_free(k.key)
}

type PrivateKeyECDH struct {
	curve string
	key   *C.GO_EC_KEY
}

func (k *PrivateKeyECDH) finalize() {
	C._goboringcrypto_EC_KEY_free(k.key)
}

func NewPublicKeyECDH(curve string, bytes []byte) (*PublicKeyECDH, error) {
	if len(bytes) != 1+2*curveSize(curve) {
		return nil, errors.New("NewPublicKeyECDH: wrong key length")
	}

	nid, err := curveNID(curve)
	if err != nil {
		return nil, err
	}

	group := C._goboringcrypto_EC_GROUP_new_by_curve_name(nid)
	if group == nil {
		return nil, fail("EC_GROUP_new_by_curve_name")
	}
	defer C._goboringcrypto_EC_GROUP_free(group)
	key := C._goboringcrypto_EC_POINT_new(group)
	if key == nil {
		return nil, fail("EC_POINT_new")
	}
	ok := C._goboringcrypto_EC_POINT_oct2point(group, key, (*C.uint8_t)(unsafe.Pointer(&bytes[0])), C.size_t(len(bytes)), nil) != 0
	if !ok {
		C._goboringcrypto_EC_POINT_free(key)
		return nil, errors.New("point not on curve")
	}

	k := &PublicKeyECDH{curve, key, group, append([]byte(nil), bytes...)}
	// Note: Because of the finalizer, any time k.key is passed to cgo,
	// that call must be followed by a call to runtime.KeepAlive(k),
	// to make sure k is not collected (and finalized) before the cgo
	// call returns.
	runtime.SetFinalizer(k, (*PublicKeyECDH).finalize)
	return k, nil
}

func (k *PublicKeyECDH) Bytes() []byte { return k.bytes }

func NewPrivateKeyECDH(curve string, bytes []byte) (*PrivateKeyECDH, error) {
	if len(bytes) != curveSize(curve) {
		return nil, errors.New("NewPrivateKeyECDH: wrong key length")
	}

	nid, err := curveNID(curve)
	if err != nil {
		return nil, err
	}
	key := C._goboringcrypto_EC_KEY_new_by_curve_name(nid)
	if key == nil {
		return nil, fail("EC_KEY_new_by_curve_name")
	}
	b := bytesToBN(bytes)
	ok := b != nil && C._goboringcrypto_EC_KEY_set_private_key(key, b) != 0
	if b != nil {
		C._goboringcrypto_BN_free(b)
	}
	if !ok {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, fail("EC_KEY_set_private_key")
	}
	k := &PrivateKeyECDH{curve, key}
	// Note: Same as in NewPublicKeyECDH regarding finalizer and KeepAlive.
	runtime.SetFinalizer(k, (*PrivateKeyECDH).finalize)
	return k, nil
}

func (k *PrivateKeyECDH) PublicKey() (*PublicKeyECDH, error) {
	defer runtime.KeepAlive(k)

	group := C._goboringcrypto_EC_KEY_get0_group(k.key)
	if group == nil {
		return nil, fail("EC_KEY_get0_group")
	}
	kbig := C._goboringcrypto_EC_KEY_get0_private_key(k.key)
	if kbig == nil {
		return nil, fail("EC_KEY_get0_private_key")
	}
	pt := C._goboringcrypto_EC_POINT_new(group)
	if pt == nil {
		return nil, fail("EC_POINT_new")
	}
	if C._goboringcrypto_EC_POINT_mul(group, pt, kbig, nil, nil, nil) == 0 {
		C._goboringcrypto_EC_POINT_free(pt)
		return nil, fail("EC_POINT_mul")
	}
	bytes, err := pointBytesECDH(k.curve, group, pt)
	if err != nil {
		C._goboringcrypto_EC_POINT_free(pt)
		return nil, err
	}
	pub := &PublicKeyECDH{k.curve, pt, group, bytes}
	// Note: Same as in NewPublicKeyECDH regarding finalizer and KeepAlive.
	runtime.SetFinalizer(pub, (*PublicKeyECDH).finalize)
	return pub, nil
}

func pointBytesECDH(curve string, group *C.GO_EC_GROUP, pt *C.GO_EC_POINT) ([]byte, error) {
	out := make([]byte, 1+2*curveSize(curve))
	n := C._goboringcrypto_EC_POINT_point2oct(group, pt, C.GO_POINT_CONVERSION_UNCOMPRESSED, (*C.uint8_t)(unsafe.Pointer(&out[0])), C.size_t(len(out)), nil)
	if int(n) != len(out) {
		return nil, fail("EC_POINT_point2oct")
	}
	return out, nil
}

func ECDH(priv *PrivateKeyECDH, pub *PublicKeyECDH) ([]byte, error) {
	// Make sure priv and pub are not garbage collected while we are in a cgo
	// call.
	//
	// The call to xCoordBytesECDH should prevent priv from being collected, but
	// include this in case the code is reordered and there is a subsequent call
	// cgo call after that point.
	defer runtime.KeepAlive(priv)
	defer runtime.KeepAlive(pub)

	group := C._goboringcrypto_EC_KEY_get0_group(priv.key)
	if group == nil {
		return nil, fail("EC_KEY_get0_group")
	}
	privBig := C._goboringcrypto_EC_KEY_get0_private_key(priv.key)
	if privBig == nil {
		return nil, fail("EC_KEY_get0_private_key")
	}
	pt := C._goboringcrypto_EC_POINT_new(group)
	if pt == nil {
		return nil, fail("EC_POINT_new")
	}
	defer C._goboringcrypto_EC_POINT_free(pt)
	if C._goboringcrypto_EC_POINT_mul(group, pt, nil, pub.key, privBig, nil) == 0 {
		return nil, fail("EC_POINT_mul")
	}
	out, err := xCoordBytesECDH(priv.curve, group, pt)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func xCoordBytesECDH(curve string, group *C.GO_EC_GROUP, pt *C.GO_EC_POINT) ([]byte, error) {
	big := C._goboringcrypto_BN_new()
	defer C._goboringcrypto_BN_free(big)
	if C._goboringcrypto_EC_POINT_get_affine_coordinates_GFp(group, pt, big, nil, nil) == 0 {
		return nil, fail("EC_POINT_get_affine_coordinates_GFp")
	}
	return bigBytesECDH(curve, big)
}

func bigBytesECDH(curve string, big *C.GO_BIGNUM) ([]byte, error) {
	out := make([]byte, curveSize(curve))
	if C._goboringcrypto_BN_bn2bin_padded((*C.uint8_t)(&out[0]), C.size_t(len(out)), big) == 0 {
		return nil, fail("BN_bn2bin_padded")
	}
	return out, nil
}

func curveSize(curve string) int {
	switch curve {
	default:
		panic("crypto/internal/boring: unknown curve " + curve)
	case "P-256":
		return 256 / 8
	case "P-384":
		return 384 / 8
	case "P-521":
		return (521 + 7) / 8
	}
}

func GenerateKeyECDH(curve string) (*PrivateKeyECDH, []byte, error) {
	nid, err := curveNID(curve)
	if err != nil {
		return nil, nil, err
	}
	key := C._goboringcrypto_EC_KEY_new_by_curve_name(nid)
	if key == nil {
		return nil, nil, fail("EC_KEY_new_by_curve_name")
	}
	if C._goboringcrypto_EC_KEY_generate_key_fips(key) == 0 {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, nil, fail("EC_KEY_generate_key_fips")
	}

	group := C._goboringcrypto_EC_KEY_get0_group(key)
	if group == nil {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, nil, fail("EC_KEY_get0_group")
	}
	b := C._goboringcrypto_EC_KEY_get0_private_key(key)
	if b == nil {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, nil, fail("EC_KEY_get0_private_key")
	}
	bytes, err := bigBytesECDH(curve, b)
	if err != nil {
		C._goboringcrypto_EC_KEY_free(key)
		return nil, nil, err
	}

	k := &PrivateKeyECDH{curve, key}
	// Note: Same as in NewPublicKeyECDH regarding finalizer and KeepAlive.
	runtime.SetFinalizer(k, (*PrivateKeyECDH).finalize)
	return k, bytes, nil
}
