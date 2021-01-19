// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

package boring

import "C"
import (
	"crypto/internal/boring/boringcrypto"
	"math/big"
	"runtime"
)

type ecdsa interface {
	NewPublicKeyECDSA(curve string, X, Y *big.Int) (*boringcrypto.GoECKey, error)
	NewPrivateKeyECDSA(curve string, X, Y *big.Int, D *big.Int) (*boringcrypto.GoECKey, error)
	SignECDSA(priv *boringcrypto.GoECKey, hash []byte) (r, s *big.Int, err error)
	SignMarshalECDSA(priv *boringcrypto.GoECKey, hash []byte) ([]byte, error)
	VerifyECDSA(pub *boringcrypto.GoECKey, hash []byte, r, s *big.Int) bool
	GenerateKeyECDSA(curve string) (X, Y, D *big.Int, err error)
	ECKeyFree(*boringcrypto.GoECKey)
}

type PrivateKeyECDSA struct {
	key *boringcrypto.GoECKey
}

func (k *PrivateKeyECDSA) finalize() {
	external.ECKeyFree(k.key)
}

type PublicKeyECDSA struct {
	key *boringcrypto.GoECKey
}

func (k *PublicKeyECDSA) finalize() {
	external.ECKeyFree(k.key)
}

func NewPublicKeyECDSA(curve string, X, Y *big.Int) (*PublicKeyECDSA, error) {
	key, err := external.NewPublicKeyECDSA(curve, X, Y)
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

func NewPrivateKeyECDSA(curve string, X, Y *big.Int, D *big.Int) (*PrivateKeyECDSA, error) {
	key, err := external.NewPrivateKeyECDSA(curve, X, Y, D)
	if err != nil {
		return nil, err
	}
	k := &PrivateKeyECDSA{key}
	// Note: Because of the finalizer, any time k.key is passed to cgo,
	// that call must be followed by a call to runtime.KeepAlive(k),
	// to make sure k is not collected (and finalized) before the cgo
	// call returns.
	runtime.SetFinalizer(k, (*PrivateKeyECDSA).finalize)
	return k, nil
}

func SignECDSA(priv *PrivateKeyECDSA, hash []byte) (r, s *big.Int, err error) {
	return external.SignECDSA(priv.key, hash)
}

func SignMarshalECDSA(priv *PrivateKeyECDSA, hash []byte) ([]byte, error) {
	return external.SignMarshalECDSA(priv.key, hash)
}

func VerifyECDSA(pub *PublicKeyECDSA, hash []byte, r, s *big.Int) bool {
	return external.VerifyECDSA(pub.key, hash, r, s)
}

func GenerateKeyECDSA(curve string) (X, Y, D *big.Int, err error) {
	return external.GenerateKeyECDSA(curve)
}
