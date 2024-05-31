// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "C"

func F() *C.char {
	s, err := C.CString("hi") // ERROR HERE: no two-result form
	if err != nil {
		println(err)
	}
	return s
}
