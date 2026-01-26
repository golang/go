// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	_ "runtime/testdata/testsyscall/testsyscallc" // unfortunately, we can't put C and assembly in the package
	_ "unsafe"                                    // for go:linkname
)

//go:linkname syscall_syscall syscall.syscall
func syscall_syscall(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr)

//go:linkname syscall_syscall6 syscall.syscall6
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)

//go:linkname syscall_syscall9 syscall.syscall9
func syscall_syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr)

var (
	syscall_check0_trampoline_addr uintptr
	syscall_check1_trampoline_addr uintptr
	syscall_check2_trampoline_addr uintptr
	syscall_check3_trampoline_addr uintptr
	syscall_check4_trampoline_addr uintptr
	syscall_check5_trampoline_addr uintptr
	syscall_check6_trampoline_addr uintptr
	syscall_check7_trampoline_addr uintptr
	syscall_check8_trampoline_addr uintptr
	syscall_check9_trampoline_addr uintptr
)

func main() {
	if ret, _, _ := syscall_syscall(syscall_check0_trampoline_addr, 0, 0, 0); ret != 1 {
		panic("hello0")
	}
	if ret, _, _ := syscall_syscall(syscall_check1_trampoline_addr, 1, 0, 0); ret != 1 {
		panic("hello1")
	}
	if ret, _, _ := syscall_syscall(syscall_check2_trampoline_addr, 1, 2, 0); ret != 1 {
		panic("hello2")
	}
	if ret, _, _ := syscall_syscall(syscall_check3_trampoline_addr, 1, 2, 3); ret != 1 {
		panic("hello3")
	}
	if ret, _, _ := syscall_syscall6(syscall_check4_trampoline_addr, 1, 2, 3, 4, 0, 0); ret != 1 {
		panic("hello4")
	}
	if ret, _, _ := syscall_syscall6(syscall_check5_trampoline_addr, 1, 2, 3, 4, 5, 0); ret != 1 {
		panic("hello5")
	}
	if ret, _, _ := syscall_syscall6(syscall_check6_trampoline_addr, 1, 2, 3, 4, 5, 6); ret != 1 {
		panic("hello6")
	}
	if ret, _, _ := syscall_syscall9(syscall_check7_trampoline_addr, 1, 2, 3, 4, 5, 6, 7, 0, 0); ret != 1 {
		panic("hello7")
	}
	if ret, _, _ := syscall_syscall9(syscall_check8_trampoline_addr, 1, 2, 3, 4, 5, 6, 7, 8, 0); ret != 1 {
		panic("hello8")
	}
	if ret, _, _ := syscall_syscall9(syscall_check9_trampoline_addr, 1, 2, 3, 4, 5, 6, 7, 8, 9); ret != 1 {
		panic("hello9")
	}
}
