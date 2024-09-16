// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

package race

import (
	"unsafe"
)

const Enabled = true

// Functions below pushed from runtime.

//go:linkname Acquire
func Acquire(addr unsafe.Pointer)

//go:linkname Release
func Release(addr unsafe.Pointer)

//go:linkname ReleaseMerge
func ReleaseMerge(addr unsafe.Pointer)

//go:linkname Disable
func Disable()

//go:linkname Enable
func Enable()

//go:linkname Read
func Read(addr unsafe.Pointer)

//go:linkname Write
func Write(addr unsafe.Pointer)

//go:linkname ReadRange
func ReadRange(addr unsafe.Pointer, len int)

//go:linkname WriteRange
func WriteRange(addr unsafe.Pointer, len int)

//go:linkname Errors
func Errors() int
