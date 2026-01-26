// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0

package rand

import "io"

func fips140SetTestingReader(r io.Reader) {
	panic("cryptotest.SetGlobalRandom is not supported when building against Go Cryptographic Module v1.0.0")
}
