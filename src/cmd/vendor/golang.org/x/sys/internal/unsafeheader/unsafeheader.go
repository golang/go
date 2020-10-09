// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unsafeheader contains header declarations for the Go runtime's
// slice and string implementations.
//
// This package allows x/sys to use types equivalent to
// reflect.SliceHeader and reflect.StringHeader without introducing
// a dependency on the (relatively heavy) "reflect" package.
package unsafeheader

import (
	"unsafe"
)

// Slice is the runtime representation of a slice.
// It cannot be used safely or portably and its representation may change in a later release.
type Slice struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

// String is the runtime representation of a string.
// It cannot be used safely or portably and its representation may change in a later release.
type String struct {
	Data unsafe.Pointer
	Len  int
}
