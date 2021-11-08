// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unsafeheader contains header declarations for the Go runtime's slice
// and string implementations.
//
// This package allows packages that cannot import "reflect" to use types that
// are tested to be equivalent to reflect.SliceHeader and reflect.StringHeader.
package unsafeheader

import (
	"unsafe"
)

// Slice is the runtime representation of a slice.
// It cannot be used safely or portably and its representation may
// change in a later release.
//
// Unlike reflect.SliceHeader, its Data field is sufficient to guarantee the
// data it references will not be garbage collected.
type Slice struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

// String is the runtime representation of a string.
// It cannot be used safely or portably and its representation may
// change in a later release.
//
// Unlike reflect.StringHeader, its Data field is sufficient to guarantee the
// data it references will not be garbage collected.
type String struct {
	Data unsafe.Pointer
	Len  int
}
