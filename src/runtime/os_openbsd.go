// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

type mOS struct {
	waitsemacount uint32
}

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func sigaction(sig int32, new, old *sigactiont)

//go:noescape
func sigaltstack(new, old *stackt)

//go:noescape
func sigprocmask(mode int32, new sigset) sigset

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

func raise(sig int32)
func raiseproc(sig int32)

//go:noescape
func tfork(param *tforkt, psize uintptr, mm *m, gg *g, fn uintptr) int32

//go:noescape
func thrsleep(ident uintptr, clock_id int32, tsp *timespec, lock uintptr, abort *uint32) int32

//go:noescape
func thrwakeup(ident uintptr, n int32) int32

func osyield()
