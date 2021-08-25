// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && mips64

package runtime

import (
	"unsafe"
)

//go:noescape
func sigaction(sig uint32, new, old *sigactiont)

func kqueue() int32

//go:noescape
func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32

func raiseproc(sig uint32)

func getthrid() int32
func thrkill(tid int32, sig int)

// read calls the read system call.
// It returns a non-negative number of bytes written or a negative errno value.
func read(fd int32, p unsafe.Pointer, n int32) int32

func closefd(fd int32) int32

func exit(code int32)
func usleep(usec uint32)

//go:nosplit
func usleep_no_g(usec uint32) {
	usleep(usec)
}

// write calls the write system call.
// It returns a non-negative number of bytes written or a negative errno value.
//go:noescape
func write1(fd uintptr, p unsafe.Pointer, n int32) int32

//go:noescape
func open(name *byte, mode, perm int32) int32

// return value is only set on linux to be used in osinit()
func madvise(addr unsafe.Pointer, n uintptr, flags int32) int32

// exitThread terminates the current thread, writing *wait = 0 when
// the stack is safe to reclaim.
//
//go:noescape
func exitThread(wait *uint32)

//go:noescape
func obsdsigprocmask(how int32, new sigset) sigset

//go:nosplit
//go:nowritebarrierrec
func sigprocmask(how int32, new, old *sigset) {
	n := sigset(0)
	if new != nil {
		n = *new
	}
	r := obsdsigprocmask(how, n)
	if old != nil {
		*old = r
	}
}

func pipe() (r, w int32, errno int32)
func pipe2(flags int32) (r, w int32, errno int32)

//go:noescape
func setitimer(mode int32, new, old *itimerval)

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

// mmap calls the mmap system call. It is implemented in assembly.
// We only pass the lower 32 bits of file offset to the
// assembly routine; the higher bits (if required), should be provided
// by the assembly routine as 0.
// The err result is an OS error code such as ENOMEM.
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (p unsafe.Pointer, err int)

// munmap calls the munmap system call. It is implemented in assembly.
func munmap(addr unsafe.Pointer, n uintptr)

func nanotime1() int64

//go:noescape
func sigaltstack(new, old *stackt)

func closeonexec(fd int32)
func setNonblock(fd int32)

func walltime() (sec int64, nsec int32)
