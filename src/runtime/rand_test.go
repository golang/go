// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
)

func BenchmarkFastrand(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Fastrand()
		}
	})
}

func BenchmarkFastrandHashiter(b *testing.B) {
	var m = make(map[int]int, 10)
	for i := 0; i < 10; i++ {
		m[i] = i
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			for _ = range m {
				break
			}
		}
	})
}
