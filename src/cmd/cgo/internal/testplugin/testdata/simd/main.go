// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"plugin"
	"simd"
)

func main() {
	p, err := plugin.Open("simd.so")
	if err != nil {
		panic(err)
	}

	sym, err := p.Lookup("Lanes")
	if err != nil {
		panic(err)
	}
	got := sym.(func() int)()
	if want := simd.VectorBits() / 32; got != want {
		panic(fmt.Sprintf("Lanes() = %d, want %d", got, want))
	}
}
