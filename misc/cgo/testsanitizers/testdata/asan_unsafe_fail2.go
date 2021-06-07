// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"unsafe"
)

func main() {
	a := 1
	b := 2
	c := add(a, b)
	d := a + b
	fmt.Println(c, d)
}

//go:noinline
func add(a1, b1 int) (ret int) {
	// The return value
	// When -asan is enabled, the unsafe.Pointer(&ret) conversion is escaping.
	var p *int = (*int)(unsafe.Add(unsafe.Pointer(&ret), 1*unsafe.Sizeof(int(1))))
	*p = 123 // BOOM
	ret = a1 + b1
	return
}
