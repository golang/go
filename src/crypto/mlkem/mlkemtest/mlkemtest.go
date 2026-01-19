// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mlkemtest provides testing functions for the ML-KEM algorithm.
package mlkemtest

import (
	fips140mlkem "crypto/internal/fips140/mlkem"
	"crypto/internal/fips140only"
	"crypto/mlkem"
	"errors"
)

// Encapsulate768 implements derandomized ML-KEM-768 encapsulation
// (ML-KEM.Encaps_internal from FIPS 203) using the provided encapsulation key
// ek and 32 bytes of randomness.
//
// It must only be used for known-answer tests.
func Encapsulate768(ek *mlkem.EncapsulationKey768, random []byte) (sharedKey, ciphertext []byte, err error) {
	if len(random) != 32 {
		return nil, nil, errors.New("mlkemtest: Encapsulate768: random must be 32 bytes")
	}
	if fips140only.Enforced() {
		return nil, nil, errors.New("crypto/mlkem/mlkemtest: use of derandomized encapsulation is not allowed in FIPS 140-only mode")
	}
	k, err := fips140mlkem.NewEncapsulationKey768(ek.Bytes())
	if err != nil {
		return nil, nil, errors.New("mlkemtest: Encapsulate768: failed to reconstruct key: " + err.Error())
	}
	sharedKey, ciphertext = k.EncapsulateInternal((*[32]byte)(random))
	return sharedKey, ciphertext, nil
}

// Encapsulate1024 implements derandomized ML-KEM-1024 encapsulation
// (ML-KEM.Encaps_internal from FIPS 203) using the provided encapsulation key
// ek and 32 bytes of randomness.
//
// It must only be used for known-answer tests.
func Encapsulate1024(ek *mlkem.EncapsulationKey1024, random []byte) (sharedKey, ciphertext []byte, err error) {
	if len(random) != 32 {
		return nil, nil, errors.New("mlkemtest: Encapsulate1024: random must be 32 bytes")
	}
	if fips140only.Enforced() {
		return nil, nil, errors.New("crypto/mlkem/mlkemtest: use of derandomized encapsulation is not allowed in FIPS 140-only mode")
	}
	k, err := fips140mlkem.NewEncapsulationKey1024(ek.Bytes())
	if err != nil {
		return nil, nil, errors.New("mlkemtest: Encapsulate1024: failed to reconstruct key: " + err.Error())
	}
	sharedKey, ciphertext = k.EncapsulateInternal((*[32]byte)(random))
	return sharedKey, ciphertext, nil
}
