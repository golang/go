// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/aes"
)

// CounterKDF implements a KDF in Counter Mode instantiated with CMAC-AES,
// according to NIST SP 800-108 Revision 1 Update 1, Section 4.1.
//
// It produces a 256-bit output, and accepts a 8-bit Label and a 96-bit Context.
// It uses a counter of 16 bits placed before the fixed data. The fixed data is
// the sequence Label || 0x00 || Context. The L field is omitted, since the
// output key length is fixed.
//
// It's optimized for use in XAES-256-GCM (https://c2sp.org/XAES-256-GCM),
// rather than for exposing it to applications as a stand-alone KDF.
type CounterKDF struct {
	mac CMAC
}

// NewCounterKDF creates a new CounterKDF with the given key.
func NewCounterKDF(b *aes.Block) *CounterKDF {
	return &CounterKDF{mac: *NewCMAC(b)}
}

// DeriveKey derives a key from the given label and context.
func (kdf *CounterKDF) DeriveKey(label byte, context [12]byte) [32]byte {
	fips140.RecordApproved()
	var output [32]byte

	var input [aes.BlockSize]byte
	input[2] = label
	copy(input[4:], context[:])

	input[1] = 0x01 // i = 1
	K1 := kdf.mac.MAC(input[:])

	input[1] = 0x02 // i = 2
	K2 := kdf.mac.MAC(input[:])

	copy(output[:], K1[:])
	copy(output[aes.BlockSize:], K2[:])
	return output
}
