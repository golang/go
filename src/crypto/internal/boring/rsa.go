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
	"crypto"
	"crypto/internal/boring/boringcrypto"
	"hash"
	"math/big"
	"runtime"
)

type rsa interface {
	GenerateKeyRSA(bits int) (N, E, D, P, Q, Dp, Dq, Qinv *big.Int, err error)
	NewPublicKeyRSA(N, E *big.Int) (*boringcrypto.GoRSA, error)
	NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv *big.Int) (*boringcrypto.GoRSA, error)
	DecryptRSAOAEP(h hash.Hash, priv *boringcrypto.GoRSA, ciphertext, label []byte) ([]byte, error)
	EncryptRSAOAEP(h hash.Hash, pub *boringcrypto.GoRSA, msg, label []byte) ([]byte, error)
	DecryptRSAPKCS1(priv *boringcrypto.GoRSA, ciphertext []byte) ([]byte, error)
	EncryptRSAPKCS1(pub *boringcrypto.GoRSA, msg []byte) ([]byte, error)
	DecryptRSANoPadding(priv *boringcrypto.GoRSA, ciphertext []byte) ([]byte, error)
	EncryptRSANoPadding(pub *boringcrypto.GoRSA, msg []byte) ([]byte, error)
	SignRSAPSS(priv *boringcrypto.GoRSA, hashed []byte, h crypto.Hash, saltLen int) ([]byte, error)
	VerifyRSAPSS(pub *boringcrypto.GoRSA, h crypto.Hash, hashed, sig []byte, saltLen int) error
	SignRSAPKCS1v15(priv *boringcrypto.GoRSA, h crypto.Hash, hashed []byte) ([]byte, error)
	VerifyRSAPKCS1v15(pub *boringcrypto.GoRSA, h crypto.Hash, hashed, sig []byte) error
	RSAFree(key *boringcrypto.GoRSA)
}

func GenerateKeyRSA(bits int) (N, E, D, P, Q, Dp, Dq, Qinv *big.Int, err error) {
	return external.GenerateKeyRSA(bits)
}

type PublicKeyRSA struct {
	// _key MUST NOT be accessed directly. Instead, use the withKey method.
	_key *boringcrypto.GoRSA
}

func NewPublicKeyRSA(N, E *big.Int) (*PublicKeyRSA, error) {
	key, err := external.NewPublicKeyRSA(N, E)
	if err != nil {
		return nil, err
	}
	k := &PublicKeyRSA{_key: key}
	runtime.SetFinalizer(k, (*PublicKeyRSA).finalize)
	return k, nil
}

func (k *PublicKeyRSA) finalize() {
	external.RSAFree(k._key)
}

func (k *PublicKeyRSA) withKey(f func(*boringcrypto.GoRSA)) {
	// Because of the finalizer, any time _key is passed to cgo, that call must
	// be followed by a call to runtime.KeepAlive, to make sure k is not
	// collected (and finalized) before the cgo call returns.
	defer runtime.KeepAlive(k)
	f(k._key)
}

type PrivateKeyRSA struct {
	// _key MUST NOT be accessed directly. Instead, use the withKey method.
	_key *boringcrypto.GoRSA
}

func (k *PrivateKeyRSA) finalize() {
	external.RSAFree(k._key)
}

func (k *PrivateKeyRSA) withKey(f func(*boringcrypto.GoRSA)) {
	// Because of the finalizer, any time _key is passed to cgo, that call must
	// be followed by a call to runtime.KeepAlive, to make sure k is not
	// collected (and finalized) before the cgo call returns.
	defer runtime.KeepAlive(k)
	f(k._key)
}

func NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv *big.Int) (*PrivateKeyRSA, error) {
	key, err := external.NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv)
	if err != nil {
		return nil, err
	}
	k := &PrivateKeyRSA{_key: key}
	runtime.SetFinalizer(k, (*PrivateKeyRSA).finalize)
	return k, nil
}

func DecryptRSAOAEP(h hash.Hash, priv *PrivateKeyRSA, ciphertext, label []byte) (out []byte, err error) {
	priv.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.DecryptRSAOAEP(h, key, ciphertext, label)
	})
	return out, err
}

func EncryptRSAOAEP(h hash.Hash, pub *PublicKeyRSA, msg, label []byte) (out []byte, err error) {
	pub.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.EncryptRSAOAEP(h, key, msg, label)
	})
	return out, err
}

func DecryptRSAPKCS1(priv *PrivateKeyRSA, ciphertext []byte) (out []byte, err error) {
	priv.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.DecryptRSAPKCS1(key, ciphertext)
	})
	return out, err
}

func EncryptRSAPKCS1(pub *PublicKeyRSA, msg []byte) (out []byte, err error) {
	pub.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.EncryptRSAPKCS1(key, msg)
	})
	return out, err
}

func DecryptRSANoPadding(priv *PrivateKeyRSA, ciphertext []byte) (out []byte, err error) {
	priv.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.DecryptRSANoPadding(key, ciphertext)
	})
	return out, err
}

func EncryptRSANoPadding(pub *PublicKeyRSA, msg []byte) (out []byte, err error) {
	pub.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.EncryptRSANoPadding(key, msg)
	})
	return out, err
}

func SignRSAPSS(priv *PrivateKeyRSA, h crypto.Hash, hashed []byte, saltLen int) (out []byte, err error) {
	if saltLen == 0 {
		saltLen = -1
	}
	priv.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.SignRSAPSS(key, hashed, h, saltLen)
	})
	return out, err
}

func VerifyRSAPSS(pub *PublicKeyRSA, h crypto.Hash, hashed, sig []byte, saltLen int) (err error) {
	if saltLen == 0 {
		saltLen = -2 // auto-recover
	}
	pub.withKey(func(key *boringcrypto.GoRSA) {
		err = external.VerifyRSAPSS(key, h, hashed, sig, saltLen)
	})
	return err
}

func SignRSAPKCS1v15(priv *PrivateKeyRSA, h crypto.Hash, hashed []byte) (out []byte, err error) {
	priv.withKey(func(key *boringcrypto.GoRSA) {
		out, err = external.SignRSAPKCS1v15(key, h, hashed)
	})
	return out, err
}

func VerifyRSAPKCS1v15(pub *PublicKeyRSA, h crypto.Hash, hashed, sig []byte) (err error) {
	pub.withKey(func(key *boringcrypto.GoRSA) {
		err = external.VerifyRSAPKCS1v15(key, h, hashed, sig)
	})
	return err
}
