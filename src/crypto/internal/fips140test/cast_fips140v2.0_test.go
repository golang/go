// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package fipstest

import "crypto/internal/fips140/mldsa"

func fips140v2Conditionals() {
	// ML-DSA sign and verify PCT
	kMLDSA := mldsa.GenerateKey44()
	// ML-DSA-44
	mldsa.SignDeterministic(kMLDSA, make([]byte, 32), "")
}
