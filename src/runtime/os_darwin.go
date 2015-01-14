// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func bsdthread_create(stk unsafe.Pointer, mm *m, gg *g, fn uintptr) int32
func bsdthread_register() int32

//go:noescape
func mach_msg_trap(h unsafe.Pointer, op int32, send_size, rcv_size, rcv_name, timeout, notify uint32) int32

func mach_reply_port() uint32
func mach_task_self() uint32
func mach_thread_self() uint32

//go:noescape
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32

//go:noescape
func sigprocmask(sig uint32, new, old *uint32)

//go:noescape
func sigaction(mode uint32, new, old *sigactiont)

//go:noescape
func sigaltstack(new, old *stackt)

func sigtramp()

//go:noescape
func setitimer(mode int32, new, old *itimerval)

func raise(int32)
func raiseproc(int32)
