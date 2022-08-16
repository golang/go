// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"unsafe"
)

func main() {
	hello := [5]byte{'m', 'o', 's', 'h', 'i'}
	if unsafe.String(&hello[0], uint64(len(hello))) != "moshi" {
		panic("unsafe.String convert error")
	}
}
