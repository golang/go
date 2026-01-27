// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha256

import "testing"

// benchmarkBlock measures the throughput of a specific block
// implementation over a buffer of the given size (multiple of 64).
func benchmarkBlock(b *testing.B, size int, fn func(*Digest, []byte)) {
	if size%chunk != 0 {
		b.Fatalf("size must be multiple of %d", chunk)
	}
	buf := make([]byte, size)
	var d Digest

	b.SetBytes(int64(size))
	b.ResetTimer() // exclude setup cost (buffer/digest init)
	for i := 0; i < b.N; i++ {
		d.Reset()
		fn(&d, buf)
	}
}

func BenchmarkBlock_1K(b *testing.B) {
	benchmarkBlock(b, 1024, block)
}

func BenchmarkBlock_8K(b *testing.B) {
	benchmarkBlock(b, 8*1024, block)
}

func BenchmarkBlock_64K(b *testing.B) {
	benchmarkBlock(b, 64*1024, block)
}
