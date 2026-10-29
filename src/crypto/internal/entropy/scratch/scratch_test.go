// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scratch

import "testing"

func TestAcquire(t *testing.T) {
	const n = 1 << 25

	for range 2 {
		b := Acquire(n)
		if len(b) != n {
			t.Fatalf("Acquire(%d) returned %d bytes", n, len(b))
		}
		for i := 0; i < n; i += 512 {
			b[i]++
		}
		b[n-1]++

		if b2 := Acquire(n); &b2[0] != &b[0] {
			t.Error("overlapping Acquire calls returned different buffers")
		}
		Release(b)
		b[0]++ // still held by b2
		Release(b)

		if shared.buf != nil || shared.refs != 0 {
			t.Errorf("buffer not released: len %d, refs %d", len(shared.buf), shared.refs)
		}
	}
}
