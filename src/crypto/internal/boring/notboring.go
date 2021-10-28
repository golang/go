// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux || !amd64 || !cgo || android || cmd_go_bootstrap || msan
// +build !linux !amd64 !cgo android cmd_go_bootstrap msan

package boring

import (
	"crypto"
	"crypto/cipher"
	"crypto/internal/boring/sig"
	"hash"
	"math/big"
)

const available = false

// Unreachable marks code that should be unreachable
// when BoringCrypto is in use. It is a no-op without BoringCrypto.
func Unreachable() {
	// Code that's unreachable when using BoringCrypto
	// is exactly the code we want to detect for reporting
	// standard Go crypto.
	sig.StandardCrypto()
}

// UnreachableExceptTests marks code that should be unreachable
// when BoringCrypto is in use. It is a no-op without BoringCrypto.
func UnreachableExceptTests() {}

type randReader int

func (randReader) Read(b []byte) (int, error) { panic("boringcrypto: not available") }

const RandReader = randReader(0)

func NewSHA1() hash.Hash   { panic("boringcrypto: not available") }
func NewSHA224() hash.Hash { panic("boringcrypto: not available") }
func NewSHA256() hash.Hash { panic("boringcrypto: not available") }
func NewSHA384() hash.Hash { panic("boringcrypto: not available") }
func NewSHA512() hash.Hash { panic("boringcrypto: not available") }

func NewHMAC(h func() hash.Hash, key []byte) hash.Hash { panic("boringcrypto: not available") }

func NewAESCipher(key []byte) (cipher.Block, error) { panic("boringcrypto: not available") }

type PublicKeyECDSA struct{ _ int }
type PrivateKeyECDSA struct{ _ int }

func GenerateKeyECDSA(curve string) (X, Y, D *big.Int, err error) {
	panic("boringcrypto: not available")
}
func NewPrivateKeyECDSA(curve string, X, Y, D *big.Int) (*PrivateKeyECDSA, error) {
	panic("boringcrypto: not available")
}
func NewPublicKeyECDSA(curve string, X, Y *big.Int) (*PublicKeyECDSA, error) {
	panic("boringcrypto: not available")
}
func SignECDSA(priv *PrivateKeyECDSA, hash []byte) (r, s *big.Int, err error) {
	panic("boringcrypto: not available")
}
func SignMarshalECDSA(priv *PrivateKeyECDSA, hash []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func VerifyECDSA(pub *PublicKeyECDSA, hash []byte, r, s *big.Int) bool {
	panic("boringcrypto: not available")
}

type PublicKeyRSA struct{ _ int }
type PrivateKeyRSA struct{ _ int }

func DecryptRSAOAEP(h hash.Hash, priv *PrivateKeyRSA, ciphertext, label []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func DecryptRSAPKCS1(priv *PrivateKeyRSA, ciphertext []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func DecryptRSANoPadding(priv *PrivateKeyRSA, ciphertext []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func EncryptRSAOAEP(h hash.Hash, pub *PublicKeyRSA, msg, label []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func EncryptRSAPKCS1(pub *PublicKeyRSA, msg []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func EncryptRSANoPadding(pub *PublicKeyRSA, msg []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func GenerateKeyRSA(bits int) (N, E, D, P, Q, Dp, Dq, Qinv *big.Int, err error) {
	panic("boringcrypto: not available")
}
func NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv *big.Int) (*PrivateKeyRSA, error) {
	panic("boringcrypto: not available")
}
func NewPublicKeyRSA(N, E *big.Int) (*PublicKeyRSA, error) { panic("boringcrypto: not available") }
func SignRSAPKCS1v15(priv *PrivateKeyRSA, h crypto.Hash, hashed []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func SignRSAPSS(priv *PrivateKeyRSA, h crypto.Hash, hashed []byte, saltLen int) ([]byte, error) {
	panic("boringcrypto: not available")
}
func VerifyRSAPKCS1v15(pub *PublicKeyRSA, h crypto.Hash, hashed, sig []byte) error {
	panic("boringcrypto: not available")
}
func VerifyRSAPSS(pub *PublicKeyRSA, h crypto.Hash, hashed, sig []byte, saltLen int) error {
	panic("boringcrypto: not available")
}
