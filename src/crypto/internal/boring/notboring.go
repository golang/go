// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(boringcrypto && linux && (amd64 || arm64) && !android && !cmd_go_bootstrap && !msan && cgo)

package boring

import (
	"crypto"
	"crypto/cipher"
	"crypto/internal/boring/sig"
	"hash"
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

func SHA1([]byte) [20]byte   { panic("boringcrypto: not available") }
func SHA224([]byte) [28]byte { panic("boringcrypto: not available") }
func SHA256([]byte) [32]byte { panic("boringcrypto: not available") }
func SHA384([]byte) [48]byte { panic("boringcrypto: not available") }
func SHA512([]byte) [64]byte { panic("boringcrypto: not available") }

func NewHMAC(h func() hash.Hash, key []byte) hash.Hash { panic("boringcrypto: not available") }

func NewAESCipher(key []byte) (cipher.Block, error) { panic("boringcrypto: not available") }
func NewGCMTLS(cipher.Block) (cipher.AEAD, error)   { panic("boringcrypto: not available") }
func NewGCMTLS13(cipher.Block) (cipher.AEAD, error) { panic("boringcrypto: not available") }

type PublicKeyECDSA struct{ _ int }
type PrivateKeyECDSA struct{ _ int }

func GenerateKeyECDSA(curve string) (X, Y, D BigInt, err error) {
	panic("boringcrypto: not available")
}
func NewPrivateKeyECDSA(curve string, X, Y, D BigInt) (*PrivateKeyECDSA, error) {
	panic("boringcrypto: not available")
}
func NewPublicKeyECDSA(curve string, X, Y BigInt) (*PublicKeyECDSA, error) {
	panic("boringcrypto: not available")
}
func SignMarshalECDSA(priv *PrivateKeyECDSA, hash []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func VerifyECDSA(pub *PublicKeyECDSA, hash []byte, sig []byte) bool {
	panic("boringcrypto: not available")
}

type PublicKeyRSA struct{ _ int }
type PrivateKeyRSA struct{ _ int }

func DecryptRSAOAEP(h, mgfHash hash.Hash, priv *PrivateKeyRSA, ciphertext, label []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func DecryptRSAPKCS1(priv *PrivateKeyRSA, ciphertext []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func DecryptRSANoPadding(priv *PrivateKeyRSA, ciphertext []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func EncryptRSAOAEP(h, mgfHash hash.Hash, pub *PublicKeyRSA, msg, label []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func EncryptRSAPKCS1(pub *PublicKeyRSA, msg []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func EncryptRSANoPadding(pub *PublicKeyRSA, msg []byte) ([]byte, error) {
	panic("boringcrypto: not available")
}
func GenerateKeyRSA(bits int) (N, E, D, P, Q, Dp, Dq, Qinv BigInt, err error) {
	panic("boringcrypto: not available")
}
func NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv BigInt) (*PrivateKeyRSA, error) {
	panic("boringcrypto: not available")
}
func NewPublicKeyRSA(N, E BigInt) (*PublicKeyRSA, error) { panic("boringcrypto: not available") }
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

type PublicKeyECDH struct{}
type PrivateKeyECDH struct{}

func ECDH(*PrivateKeyECDH, *PublicKeyECDH) ([]byte, error)      { panic("boringcrypto: not available") }
func GenerateKeyECDH(string) (*PrivateKeyECDH, []byte, error)   { panic("boringcrypto: not available") }
func NewPrivateKeyECDH(string, []byte) (*PrivateKeyECDH, error) { panic("boringcrypto: not available") }
func NewPublicKeyECDH(string, []byte) (*PublicKeyECDH, error)   { panic("boringcrypto: not available") }
func (*PublicKeyECDH) Bytes() []byte                            { panic("boringcrypto: not available") }
func (*PrivateKeyECDH) PublicKey() (*PublicKeyECDH, error)      { panic("boringcrypto: not available") }
