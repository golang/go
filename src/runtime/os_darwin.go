// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func bsdthread_create(stk, mm, gg, fn unsafe.Pointer) int32
func bsdthread_register() int32
func mach_msg_trap(h unsafe.Pointer, op int32, send_size, rcv_size, rcv_name, timeout, notify uint32) int32
func mach_reply_port() uint32
func mach_task_self() uint32
func mach_thread_self() uint32
func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32
func sigprocmask(sig int32, new, old unsafe.Pointer)
func sigaction(mode uint32, new, old unsafe.Pointer)
func sigaltstack(new, old unsafe.Pointer)
func sigtramp()
func setitimer(mode int32, new, old unsafe.Pointer)
func mach_semaphore_wait(sema uint32) int32
func mach_semaphore_timedwait(sema, sec, nsec uint32) int32
func mach_semaphore_signal(sema uint32) int32
func mach_semaphore_signal_all(sema uint32) int32
