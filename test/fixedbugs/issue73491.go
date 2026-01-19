// build

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T int

const K T = 5

type P struct {
	a [K]*byte
}

//go:noinline
func f(p *P) {
	for i := range K {
		p.a[i] = nil
	}
}
func main() {
	f(nil)
}
