// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140only

import "crypto/fips140"

// Enforced reports whether FIPS 140-only mode is enabled and enforced, in which non-approved
// cryptography returns an error or panics.
func Enforced() bool {
	return fips140.Enforced()
}
