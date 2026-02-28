// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"reflect"
	"unsafe"
)

func main() {
	var s = "abc"
	sh1 := (*reflect.StringHeader)(unsafe.Pointer(&s))
	ptr2 := unsafe.Pointer(unsafe.StringData(s))
	if ptr2 != unsafe.Pointer(sh1.Data) {
		panic(fmt.Errorf("unsafe.StringData ret %p != %p", ptr2, unsafe.Pointer(sh1.Data)))
	}
}
