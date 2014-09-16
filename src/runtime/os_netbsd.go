// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func setitimer(mode int32, new, old unsafe.Pointer)
func sigaction(sig int32, new, old unsafe.Pointer)
func sigaltstack(new, old unsafe.Pointer)
func sigprocmask(mode int32, new, old unsafe.Pointer)
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32
func lwp_tramp()
func raise(sig int32)
func getcontext(ctxt unsafe.Pointer)
func lwp_create(ctxt unsafe.Pointer, flags uintptr, lwpid unsafe.Pointer) int32
func lwp_park(abstime unsafe.Pointer, unpark int32, hint, unparkhint unsafe.Pointer) int32
func lwp_unpark(lwp int32, hint unsafe.Pointer) int32
func lwp_self() int32
