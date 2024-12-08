// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package drbg

import (
	"crypto/internal/entropy"
	"crypto/internal/fips140"
	"crypto/internal/sysrand"
	"sync"
)

var mu sync.Mutex
var drbg *Counter

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

	mu.Lock()
	defer mu.Unlock()

	if drbg == nil {
		entropy.Depleted(func(seed *[48]byte) {
			drbg = NewCounter(seed)
		})
	}

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
