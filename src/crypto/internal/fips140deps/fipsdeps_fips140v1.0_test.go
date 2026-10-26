// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0 || fips140v1.26

package fipsdeps

func init() {
	// In these module versions, hmac imports sha512 and sha3.
	checkDeps["crypto/internal/fips140/sha512"] = true
	checkDeps["crypto/internal/fips140/sha3"] = true
}
