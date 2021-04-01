// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"strconv"
	"testing"
)

func BenchmarkMutatorBytes(b *testing.B) {
	for _, size := range []int{
		1,
		10,
		100,
		1000,
		10000,
		100000,
	} {
		size := size
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			vals := []interface{}{make([]byte, size)}
			m := newMutator()
			for i := 0; i < b.N; i++ {
				m.mutate(vals, workerSharedMemSize)
			}
		})
	}
}
