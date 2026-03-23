// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !wasm

// This file contains reading from from entropy sources in FIPS-140
// mode. It uses a scratch buffer in the BSS section (see below),
// which usually doesn't cost much, except on Wasm, due to the way
// the linear memory works. FIPS-140 mode is not supported on Wasm,
// so we just use a build tag to exclude it. (Could also exclude other
// platforms that does not support FIPS-140 mode, but as the BSS
// variable doesn't cost much, don't bother.)

package drbg

import (
	entropy "crypto/internal/entropy/v1.0.0"
	"crypto/internal/sysrand"
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

func readFromEntropy(b []byte) {
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
