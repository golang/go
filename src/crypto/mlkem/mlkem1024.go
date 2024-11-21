// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlkem

import "crypto/internal/fips140/mlkem"

const (
	// CiphertextSize1024 is the size of a ciphertext produced by the 1024-bit
	// variant of ML-KEM.
	CiphertextSize1024 = 1568

	// EncapsulationKeySize1024 is the size of an encapsulation key for the
	// 1024-bit variant of ML-KEM.
	EncapsulationKeySize1024 = 1568
)

// DecapsulationKey1024 is the secret key used to decapsulate a shared key
// from a ciphertext. It includes various precomputed values.
type DecapsulationKey1024 struct {
	key *mlkem.DecapsulationKey1024
}

// GenerateKey1024 generates a new decapsulation key, drawing random bytes from
// crypto/rand. The decapsulation key must be kept secret.
func GenerateKey1024() (*DecapsulationKey1024, error) {
	key, err := mlkem.GenerateKey1024()
	if err != nil {
		return nil, err
	}

	return &DecapsulationKey1024{key}, nil
}

// NewDecapsulationKey1024 parses a decapsulation key from a 64-byte seed in the
// "d || z" form. The seed must be uniformly random.
func NewDecapsulationKey1024(seed []byte) (*DecapsulationKey1024, error) {
	key, err := mlkem.NewDecapsulationKey1024(seed)
	if err != nil {
		return nil, err
	}

	return &DecapsulationKey1024{key}, nil
}

// Bytes returns the decapsulation key as a 64-byte seed in the "d || z" form.
//
// The decapsulation key must be kept secret.
func (dk *DecapsulationKey1024) Bytes() []byte {
	return dk.key.Bytes()
}

// Decapsulate generates a shared key from a ciphertext and a decapsulation
// key. If the ciphertext is not valid, Decapsulate returns an error.
//
// The shared key must be kept secret.
func (dk *DecapsulationKey1024) Decapsulate(ciphertext []byte) (sharedKey []byte, err error) {
	return dk.key.Decapsulate(ciphertext)
}

// EncapsulationKey returns the public encapsulation key necessary to produce
// ciphertexts.
func (dk *DecapsulationKey1024) EncapsulationKey() *EncapsulationKey1024 {
	return &EncapsulationKey1024{dk.key.EncapsulationKey()}
}

// An EncapsulationKey1024 is the public key used to produce ciphertexts to be
// decapsulated by the corresponding DecapsulationKey1024.
type EncapsulationKey1024 struct {
	key *mlkem.EncapsulationKey1024
}

// NewEncapsulationKey1024 parses an encapsulation key from its encoded form. If
// the encapsulation key is not valid, NewEncapsulationKey1024 returns an error.
func NewEncapsulationKey1024(encapsulationKey []byte) (*EncapsulationKey1024, error) {
	key, err := mlkem.NewEncapsulationKey1024(encapsulationKey)
	if err != nil {
		return nil, err
	}

	return &EncapsulationKey1024{key}, nil
}

// Bytes returns the encapsulation key as a byte slice.
func (ek *EncapsulationKey1024) Bytes() []byte {
	return ek.key.Bytes()
}

// Encapsulate generates a shared key and an associated ciphertext from an
// encapsulation key, drawing random bytes from crypto/rand.
//
// The shared key must be kept secret.
func (ek *EncapsulationKey1024) Encapsulate() (ciphertext, sharedKey []byte) {
	return ek.key.Encapsulate()
}
