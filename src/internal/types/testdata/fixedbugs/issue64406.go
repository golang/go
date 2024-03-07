// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue64406

import (
	"unsafe"
)

func sliceData[E any, S ~[]E](s S) *E {
	return unsafe.SliceData(s)
}

func slice[E any, S ~*E](s S) []E {
	return unsafe.Slice(s, 0)
}

func f() {
	s := []uint32{0}
	_ = sliceData(s)
	_ = slice(&s)
}
