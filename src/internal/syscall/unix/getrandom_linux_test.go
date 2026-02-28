// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix_test

import (
	"internal/syscall/unix"
	"testing"
)

func BenchmarkParallelGetRandom(b *testing.B) {
	b.SetBytes(4)
	b.RunParallel(func(pb *testing.PB) {
		var buf [4]byte
		for pb.Next() {
			if _, err := unix.GetRandom(buf[:], 0); err != nil {
				b.Fatal(err)
			}
		}
	})
}
