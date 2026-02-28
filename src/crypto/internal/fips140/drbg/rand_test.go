// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package drbg

import (
	"crypto/internal/fips140"
	"testing"
)

func BenchmarkDBRG(b *testing.B) {
	old := fips140.Enabled
	defer func() {
		fips140.Enabled = old
	}()
	fips140.Enabled = true

	const N = 64
	b.SetBytes(N)
	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, N)
		for pb.Next() {
			Read(buf)
		}
	})
}
