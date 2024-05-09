// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname generic functions in internal/weak is not
// allowed; legitimate instantiation is ok.

package main

import (
	"unique"
	"unsafe"
)

//go:linkname weakMake internal/weak.Make[string]
func weakMake(string) unsafe.Pointer

func main() {
	h := unique.Make("xxx")
	println(h.Value())
	weakMake("xxx")
}
