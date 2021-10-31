// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import "unsafe"

func _[IntPtr ~uintptr, RealPtr *T, AnyPtr uintptr | *T, T any]() {
	var (
		i IntPtr
		r RealPtr
		a AnyPtr
	)
	_ = unsafe.Pointer(i)          // incorrect, but not detected
	_ = unsafe.Pointer(i + i)      // incorrect, but not detected
	_ = unsafe.Pointer(1 + i)      // incorrect, but not detected
	_ = unsafe.Pointer(uintptr(i)) // want "possible misuse of unsafe.Pointer"
	_ = unsafe.Pointer(r)
	_ = unsafe.Pointer(a) // possibly incorrect, but not detected
}
