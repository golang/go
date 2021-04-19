// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context_test

import (
	. "context"
	"fmt"
	"testing"
)

func BenchmarkContextCancelTree(b *testing.B) {
	depths := []int{1, 10, 100, 1000}
	for _, d := range depths {
		b.Run(fmt.Sprintf("depth=%d", d), func(b *testing.B) {
			b.Run("Root=Background", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					buildContextTree(Background(), d)
				}
			})
			b.Run("Root=OpenCanceler", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					ctx, cancel := WithCancel(Background())
					buildContextTree(ctx, d)
					cancel()
				}
			})
			b.Run("Root=ClosedCanceler", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					ctx, cancel := WithCancel(Background())
					cancel()
					buildContextTree(ctx, d)
				}
			})
		})
	}
}

func buildContextTree(root Context, depth int) {
	for d := 0; d < depth; d++ {
		root, _ = WithCancel(root)
	}
}
