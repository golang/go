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
	// The local variables.
	// When -asan is enabled, the unsafe.Pointer(&a) conversion is escaping.
	var p *int = (*int)(unsafe.Add(unsafe.Pointer(&a), 1*unsafe.Sizeof(int(1))))
	*p = 20 // BOOM
	d := a + b
	fmt.Println(d)
}
