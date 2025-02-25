// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package drbg provides cryptographically secure random bytes
// usable by FIPS code. In FIPS mode it uses an SP 800-90A Rev. 1
// Deterministic Random Bit Generator (DRBG). Otherwise,
// it uses the operating system's random number generator.
package drbg

import (
	"crypto/internal/entropy"
	"crypto/internal/fips140"
	"crypto/internal/randutil"
	"crypto/internal/sysrand"
	"io"
	"sync"
)

var drbgs = sync.Pool{
	New: func() any {
		var c *Counter
		entropy.Depleted(func(seed *[48]byte) {
			c = NewCounter(seed)
		})
		return c
	},
}

// Read fills b with cryptographically secure random bytes. In FIPS mode, it
// uses an SP 800-90A Rev. 1 Deterministic Random Bit Generator (DRBG).
// Otherwise, it uses the operating system's random number generator.
func Read(b []byte) {
	if !fips140.Enabled {
		sysrand.Read(b)
		return
	}

	// At every read, 128 random bits from the operating system are mixed as
	// additional input, to make the output as strong as non-FIPS randomness.
	// This is not credited as entropy for FIPS purposes, as allowed by Section
	// 8.7.2: "Note that a DRBG does not rely on additional input to provide
	// entropy, even though entropy could be provided in the additional input".
	additionalInput := new([SeedSize]byte)
	sysrand.Read(additionalInput[:16])

	drbg := drbgs.Get().(*Counter)
	defer drbgs.Put(drbg)

	for len(b) > 0 {
		size := min(len(b), maxRequestSize)
		if reseedRequired := drbg.Generate(b[:size], additionalInput); reseedRequired {
			// See SP 800-90A Rev. 1, Section 9.3.1, Steps 6-8, as explained in
			// Section 9.3.2: if Generate reports a reseed is required, the
			// additional input is passed to Reseed along with the entropy and
			// then nulled before the next Generate call.
			entropy.Depleted(func(seed *[48]byte) {
				drbg.Reseed(seed, additionalInput)
			})
			additionalInput = nil
			continue
		}
		b = b[size:]
	}
}

// DefaultReader is a sentinel type, embedded in the default
// [crypto/rand.Reader], used to recognize it when passed to
// APIs that accept a rand io.Reader.
type DefaultReader interface{ defaultReader() }

// ReadWithReader uses Reader to fill b with cryptographically secure random
// bytes. It is intended for use in APIs that expose a rand io.Reader.
//
// If Reader is not the default Reader from crypto/rand,
// [randutil.MaybeReadByte] and [fips140.RecordNonApproved] are called.
func ReadWithReader(r io.Reader, b []byte) error {
	if _, ok := r.(DefaultReader); ok {
		Read(b)
		return nil
	}

	fips140.RecordNonApproved()
	randutil.MaybeReadByte(r)
	_, err := io.ReadFull(r, b)
	return err
}

// ReadWithReaderDeterministic is like ReadWithReader, but it doesn't call
// [randutil.MaybeReadByte] on non-default Readers.
func ReadWithReaderDeterministic(r io.Reader, b []byte) error {
	if _, ok := r.(DefaultReader); ok {
		Read(b)
		return nil
	}

	fips140.RecordNonApproved()
	_, err := io.ReadFull(r, b)
	return err
}
