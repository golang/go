// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

var glob []byte

//go:noinline
func g(n int) []byte {
	return make([]byte, n)
}

//go:noinline
func f(n int) []byte {
	return g(n)
}
func main() {
	tmp := f(512)
	for i := range tmp {
		tmp[i] = byte(i + 1)
	}
	tmpOK := true
	for i := range tmp {
		if tmp[i] != byte(i+1) {
			tmpOK = false
			break
		}
	}

	glob = f(2048)
	for i := range glob {
		glob[i] = byte(i*31 + 7)
	}
	for range 8 {
		j := make([]byte, 2048)
		for i := range j {
			j[i] = 0xAA
		}
		_ = j[0]
	}

	bad := !tmpOK
	for i := range glob {
		if glob[i] != byte(i*31+7) {
			bad = true
			break
		}
	}
	if bad {
		fmt.Println("LAW arena result reclaimed by a non-consenting frame")
		os.Exit(1)
	}
	fmt.Println("ok")
}
