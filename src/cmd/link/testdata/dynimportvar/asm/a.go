// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a separate package because we cannot have Go
// assembly code and cgo code in the same package.

//go:build darwin

package asm

//go:cgo_import_dynamic libc_mach_task_self_ mach_task_self_ "/usr/lib/libSystem.B.dylib"

// load mach_task_self_ from assembly code
func Mach_task_self() uint32
