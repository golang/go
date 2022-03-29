// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ssa

import (
	"math/rand"
	"testing"
)

var d int

//go:noinline
func fn(a, b int) bool {
	c := false
	if a > 0 {
		if b < 0 {
			d = d + 1
		}
		c = true
	}
	return c
}

func BenchmarkPhioptPass(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := rand.Perm(i/10 + 10)
		for i := 1; i < len(a)/2; i++ {
			fn(a[i]-a[i-1], a[i+len(a)/2-2]-a[i+len(a)/2-1])
		}
	}
}
