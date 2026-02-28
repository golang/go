// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"reflect"
	"unsafe"
)

var slice = []byte{'H', 'e', 'l', 'l', 'o', ','}

func main() {
	ptr := uintptr(unsafe.Pointer(&slice)) + 100
	header := (*reflect.SliceHeader)(unsafe.Pointer(ptr))
	header.Data += 1
	fmt.Printf("%d %d\n", cap(slice), header.Cap)
}
