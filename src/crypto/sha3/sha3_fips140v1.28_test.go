// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(fips140v1.0 || fips140v1.26)

package sha3_test

import "crypto/internal/fips140/sha3"

func ReadMulti(s []*sha3.SHAKE, out [][]byte) {
	sha3.ReadMulti(s, out)
}
