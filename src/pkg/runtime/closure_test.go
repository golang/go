// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package runtime_test

import "testing"

var s int

func BenchmarkCallClosure(b *testing.B) {
	for i := 0; i < b.N; i++ {
		s += func(ii int) int { return 2 * ii }(i)
	}
}

func BenchmarkCallClosure1(b *testing.B) {
	for i := 0; i < b.N; i++ {
		j := i
		s += func(ii int) int { return 2*ii + j }(i)
	}
}
