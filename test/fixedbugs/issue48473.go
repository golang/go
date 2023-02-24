// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func f(x uint64) uint64 {
	s := "\x04"
	c := s[0]
	return x - x<<c<<4
}

func g(x uint32) uint32 {
	s := "\x04"
	c := s[0]
	return x - x<<c<<4
}

func main() {
	if want, got := uint64(0xffffffffffffff01), f(1); want != got {
		panic(fmt.Sprintf("want %x got %x", want, got))
	}
	if want, got := uint32(0xffffff01), g(1); want != got {
		panic(fmt.Sprintf("want %x got %x", want, got))
	}
}
