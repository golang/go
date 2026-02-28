// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unsafeptr defines an Analyzer that checks for invalid
// conversions of uintptr to unsafe.Pointer.
//
// # Analyzer unsafeptr
//
// unsafeptr: check for invalid conversions of uintptr to unsafe.Pointer
//
// The unsafeptr analyzer reports likely incorrect uses of unsafe.Pointer
// to convert integers to pointers. A conversion from uintptr to
// unsafe.Pointer is invalid if it implies that there is a uintptr-typed
// word in memory that holds a pointer value, because that word will be
// invisible to stack copying and to the garbage collector.
package unsafeptr
