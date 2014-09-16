// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func setitimer(mode int32, new, old unsafe.Pointer)
func sigaction(sig int32, new, old unsafe.Pointer)
func sigaltstack(new, old unsafe.Pointer)
func sigprocmask(mode int32, new uint32) uint32
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32
func raise(sig int32)
func tfork(param unsafe.Pointer, psize uintptr, mm, gg, fn unsafe.Pointer) int32
func thrsleep(ident unsafe.Pointer, clock_id int32, tsp, lock, abort unsafe.Pointer) int32
func thrwakeup(ident unsafe.Pointer, n int32) int32
