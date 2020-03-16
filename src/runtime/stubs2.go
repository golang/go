// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9
// +build !solaris
// +build !windows
// +build !js
// +build !darwin
// +build !aix

package runtime

import "unsafe"

// read calls the read system call.
// It returns a non-negative number of bytes written or a negative errno value.
func read(fd int32, p unsafe.Pointer, n int32) int32

func closefd(fd int32) int32

func exit(code int32)
func usleep(usec uint32)

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
