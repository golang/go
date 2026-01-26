// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package drbg provides cryptographically secure random bytes
// usable by FIPS code. In FIPS mode it uses an SP 800-90A Rev. 1
// Deterministic Random Bit Generator (DRBG). Otherwise,
// it uses the operating system's random number generator.
package drbg

import (
	entropy "crypto/internal/entropy/v1.0.0"
	"crypto/internal/fips140"
	"crypto/internal/sysrand"
	"io"
	"sync"
	"sync/atomic"
)

// memory is a scratch buffer that is accessed between samples by the entropy
// source to expose it to memory access timings.
//
// We reuse it and share it between Seed calls to avoid the significant (~500µs)
// cost of zeroing a new allocation every time. The entropy source accesses it
// using atomics (and doesn't care about its contents).
//
// It should end up in the .noptrbss section, and become backed by physical pages
// at first use. This ensures that programs that do not use the FIPS 140-3 module
// do not incur any memory use or initialization penalties.
var memory entropy.ScratchBuffer

func getEntropy() *[SeedSize]byte {
	var retries int
	seed, err := entropy.Seed(&memory)
	for err != nil {
		// The CPU jitter-based SP 800-90B entropy source has a non-negligible
		// chance of failing the startup health tests.
		//
		// Each time it does, it enters a permanent failure state, and we
		// restart it anew. This is not expected to happen more than a few times
		// in a row.
		if retries++; retries > 100 {
			panic("fips140/drbg: failed to obtain initial entropy")
		}
		seed, err = entropy.Seed(&memory)
	}
	return &seed
}

// getEntropy is very slow (~500µs), so we don't want it on the hot path.
// We keep both a persistent DRBG instance and a pool of additional instances.
// Occasional uses will use drbgInstance, even if the pool was emptied since the
// last use. Frequent concurrent uses will fill the pool and use it.
var drbgInstance atomic.Pointer[Counter]
var drbgPool = sync.Pool{
	New: func() any {
		return NewCounter(getEntropy())
	},
}

// Read fills b with cryptographically secure random bytes. In FIPS mode, it
// uses an SP 800-90A Rev. 1 Deterministic Random Bit Generator (DRBG).
// Otherwise, it uses the operating system's random number generator.
func Read(b []byte) {
	if testingReader != nil {
		fips140.RecordNonApproved()
		// Avoid letting b escape in the non-testing case.
		bb := make([]byte, len(b))
		testingReader.Read(bb)
		copy(b, bb)
		return
	}

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

	drbg := drbgInstance.Swap(nil)
	if drbg == nil {
		drbg = drbgPool.Get().(*Counter)
	}
	defer func() {
		if !drbgInstance.CompareAndSwap(nil, drbg) {
			drbgPool.Put(drbg)
		}
	}()

	for len(b) > 0 {
		size := min(len(b), maxRequestSize)
		if reseedRequired := drbg.Generate(b[:size], additionalInput); reseedRequired {
			// See SP 800-90A Rev. 1, Section 9.3.1, Steps 6-8, as explained in
			// Section 9.3.2: if Generate reports a reseed is required, the
			// additional input is passed to Reseed along with the entropy and
			// then nulled before the next Generate call.
			drbg.Reseed(getEntropy(), additionalInput)
			additionalInput = nil
			continue
		}
		b = b[size:]
	}
}

var testingReader io.Reader

// SetTestingReader sets a global, deterministic cryptographic randomness source
// for testing purposes. Its Read method must never return an error, it must
// never return short, and it must be safe for concurrent use.
//
// This is only intended to be used by the testing/cryptotest package.
func SetTestingReader(r io.Reader) {
	testingReader = r
}

// DefaultReader is a sentinel type, embedded in the default
// [crypto/rand.Reader], used to recognize it when passed to
// APIs that accept a rand io.Reader.
//
// Any Reader that implements this interface is assumed to
// call [Read] as its Read method.
type DefaultReader interface{ defaultReader() }

// ReadWithReader uses Reader to fill b with cryptographically secure random
// bytes. It is intended for use in APIs that expose a rand io.Reader.
func ReadWithReader(r io.Reader, b []byte) error {
	if _, ok := r.(DefaultReader); ok {
		Read(b)
		return nil
	}

	fips140.RecordNonApproved()
	_, err := io.ReadFull(r, b)
	return err
}
