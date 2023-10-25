// -fakeImportC

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importC

import "C"
import _ /* ERROR `cannot rename import "C"` */ "C"
import foo /* ERROR `cannot rename import "C"` */ "C"
import . /* ERROR `cannot rename import "C"` */ "C"

// Test cases extracted from issue #22090.

import "unsafe"

const _ C.int = 0xff // no error due to invalid constant type

type T struct {
	Name    string
	Ordinal int
}

func _(args []T) {
	var s string
	for i, v := range args {
		cname := C.CString(v.Name)
		args[i].Ordinal = int(C.sqlite3_bind_parameter_index(s, cname)) // no error due to i not being "used"
		C.free(unsafe.Pointer(cname))
	}
}

type CType C.Type

const _ CType = C.X // no error due to invalid constant type
const _ = C.X

// Test cases extracted from issue #23712.

func _() {
	var a [C.ArrayLength]byte
	_ = a[0] // no index out of bounds error here
}

// Additional tests to verify fix for #23712.

func _() {
	var a [C.ArrayLength1]byte
	_ = 1 / len(a) // no division by zero error here and below
	_ = 1 / cap(a)
	_ = uint(unsafe.Sizeof(a)) // must not be negative

	var b [C.ArrayLength2]byte
	a = b // should be valid
}
