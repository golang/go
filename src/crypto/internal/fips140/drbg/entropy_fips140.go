// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package drbg

import (
	"crypto/internal/entropy/scratch"
	entropy "crypto/internal/entropy/v1.0.0"
)

func getEntropy() *[SeedSize]byte {
	b := scratch.Acquire(len(entropy.ScratchBuffer{}))
	defer scratch.Release(b)
	buf := (*entropy.ScratchBuffer)(b)

	var retries int
	seed, err := entropy.Seed(buf)
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
		seed, err = entropy.Seed(buf)
	}
	return &seed
}
