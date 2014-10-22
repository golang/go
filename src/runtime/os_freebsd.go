// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func thr_new(param unsafe.Pointer, size int32)
func sigaltstack(new, old unsafe.Pointer)
func sigaction(sig int32, new, old unsafe.Pointer)
func sigprocmask(new, old unsafe.Pointer)
func setitimer(mode int32, new, old unsafe.Pointer)
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32
func getrlimit(kind int32, limit unsafe.Pointer) int32
func raise(sig int32)
func sys_umtx_op(addr unsafe.Pointer, mode int32, val uint32, ptr2, ts unsafe.Pointer) int32
