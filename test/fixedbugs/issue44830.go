// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"unsafe"
)

func main() {
	t := reflect.TypeOf(unsafe.Pointer(nil))
	if pkgPath := t.PkgPath(); pkgPath != "unsafe" {
		panic("unexpected t.PkgPath(): " + pkgPath)
	}
}
