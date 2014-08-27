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
func getrlimit(kind int32, limit unsafe.Pointer)
func asmsysvicall6(fn unsafe.Pointer)
func miniterrno(fn unsafe.Pointer)
func raise(sig int32)
func getcontext(ctxt unsafe.Pointer)
func tstart_sysvicall(mm unsafe.Pointer) uint32
func nanotime1() int64
func usleep1(usec uint32)
func osyield1()
