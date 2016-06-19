// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE: If you change this file you must run "go generate"
// to update builtin.go. This is not done automatically
// to avoid depending on having a working compiler binary.

// +build ignore

package unsafe

type Pointer uintptr // not really; filled in by compiler

// return types here are ignored; see unsafe.go
func Offsetof(any) uintptr
func Sizeof(any) uintptr
func Alignof(any) uintptr
