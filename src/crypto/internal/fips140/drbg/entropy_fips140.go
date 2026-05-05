// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Entropy generation in FIPS 140-3 mode uses a scratch buffer in the BSS
// section (see below), which usually doesn't cost much, except on Wasm, due to
// the way the linear memory works. FIPS 140-3 mode is not supported on Wasm, so
// we just use a build tag to exclude it. (Could also exclude other platforms
// that does not support FIPS 140-3 mode, but as the BSS variable doesn't cost
// much, don't bother.)
//
//go:build !wasm

package drbg

import entropy "crypto/internal/entropy/v1.0.0"

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
