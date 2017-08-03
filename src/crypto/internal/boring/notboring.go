// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux !amd64 cmd_go_bootstrap

package boring

import (
	"crypto/cipher"
	"hash"
	"math/big"
)

const available = false

// Unreachable marks code that should be unreachable
// when BoringCrypto is in use. It is a no-op without BoringCrypto.
func Unreachable() {}

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
