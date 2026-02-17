// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type P struct {
	q struct{}
	p *int
}

func f(x any) {
	h(x.(P))
}

//go:noinline
func h(P) {
}
