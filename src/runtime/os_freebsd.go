// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type mOS struct{}

//go:noescape
func thr_new(param *thrparam, size int32)

//go:noescape
func sigaltstack(new, old *stackt)

//go:noescape
func sigaction(sig int32, new, old *sigactiont)

//go:noescape
func sigprocmask(how int32, new, old *sigset)

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

//go:noescape
func getrlimit(kind int32, limit unsafe.Pointer) int32
func raise(sig int32)
func raiseproc(sig int32)

//go:noescape
func sys_umtx_op(addr *uint32, mode int32, val uint32, ptr2, ts *timespec) int32

func osyield()
