// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import (
	"testing"
)

func BenchmarkSearchInts(b *testing.B) {
	data := make([]int, 10000)
	for i := 0; i < 10000; i++ {
		data[i] = i
	}
	const x = 8
	if r := searchInts(data, x); r != x {
		b.Errorf("got index = %d; want %d", r, x)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		searchInts(data, x)
	}
}
