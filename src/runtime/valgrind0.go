// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Valgrind instrumentation is only available on linux amd64 and arm64.

//go:build !valgrind || !linux || (!amd64 && !arm64)

package runtime

import "unsafe"

const valgrindenabled = false

func valgrindRegisterStack(start, end unsafe.Pointer) uintptr       { return 0 }
func valgrindDeregisterStack(id uintptr)                            {}
func valgrindChangeStack(id uintptr, start, end unsafe.Pointer)     {}
func valgrindMalloc(addr unsafe.Pointer, size uintptr)              {}
func valgrindFree(addr unsafe.Pointer)                              {}
func valgrindCreateMempool(addr unsafe.Pointer)                     {}
func valgrindMempoolMalloc(pool, addr unsafe.Pointer, size uintptr) {}
func valgrindMempoolFree(pool, addr unsafe.Pointer)                 {}
func valgrindMakeMemUndefined(addr unsafe.Pointer, size uintptr)    {}
func valgrindMakeMemDefined(addr unsafe.Pointer, size uintptr)      {}
func valgrindMakeMemNoAccess(addr unsafe.Pointer, size uintptr)     {}
