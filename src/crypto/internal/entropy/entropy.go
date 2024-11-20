// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package entropy provides the passive entropy source for the FIPS 140-3
// module. It is only used in FIPS mode by [crypto/internal/fips/drbg.Read].
//
// This complies with IG 9.3.A, Additional Comment 12, which until January 1,
// 2026 allows new modules to meet an [earlier version] of Resolution 2(b):
// "A software module that contains an approved DRBG that receives a LOAD
// command (or its logical equivalent) with entropy obtained from [...] inside
// the physical perimeter of the operational environment of the module [...]."
//
// Distributions that have their own SP 800-90B entropy source should replace
// this package with their own implementation.
//
// [earlier version]: https://csrc.nist.gov/CSRC/media/Projects/cryptographic-module-validation-program/documents/IG%209.3.A%20Resolution%202b%5BMarch%2026%202024%5D.pdf
package entropy

import "crypto/internal/sysrand"

// Depleted notifies the entropy source that the entropy in the module is
// "depleted" and provides the callback for the LOAD command.
func Depleted(LOAD func(*[48]byte)) {
	var entropy [48]byte
	sysrand.Read(entropy[:])
	LOAD(&entropy)
}
