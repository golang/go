// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm

package drbg

func getEntropy() *[SeedSize]byte {
	panic("FIPS 140-3 entropy generation is not supported on Wasm")
}
