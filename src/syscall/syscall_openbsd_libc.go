// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

package syscall

import (
	"internal/abi"
)

var dupTrampoline = abi.FuncPCABI0(libc_dup3_trampoline)

func init() {
	execveOpenBSD = execve
}

func syscallInternal(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno) {
	// OpenBSD 7.5+ no longer supports indirect syscalls. A number of Go
	// packages make use of syscall.Syscall with SYS_IOCTL since it is
	// not well supported by golang.org/x/sys/unix. Reroute this system
	// call number to the respective libc stub so that it continues to
	// work for the time being. See #63900 for further details.
	if trap == SYS_IOCTL {
		return syscallX(abi.FuncPCABI0(libc_ioctl_trampoline), a1, a2, a3)
	}
	return 0, 0, ENOSYS
}

func syscall6Internal(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) {
	// OpenBSD 7.5+ no longer supports indirect syscalls. A number of Go
	// packages make use of syscall.Syscall with SYS___SYSCTL since it is
	// not well supported by golang.org/x/sys/unix. Reroute this system
	// call number to the respective libc stub so that it continues to
	// work for the time being. See #63900 for further details.
	if trap == SYS___SYSCTL {
		return syscall6X(abi.FuncPCABI0(libc_sysctl_trampoline), a1, a2, a3, a4, a5, a6)
	}
	return 0, 0, ENOSYS
}

func rawSyscallInternal(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

func rawSyscall6Internal(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

func syscall9Internal(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

// Implemented in the runtime package (runtime/sys_openbsd3.go)
func syscall(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func syscallX(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func syscall6X(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func syscall10(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 uintptr) (r1, r2 uintptr, err Errno)
func syscall10X(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 uintptr) (r1, r2 uintptr, err Errno)
func rawSyscall(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func rawSyscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func rawSyscall6X(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func rawSyscall10X(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 uintptr) (r1, r2 uintptr, err Errno)

func syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno) {
	return syscall10(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, 0)
}
func syscall9X(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno) {
	return syscall10X(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, 0)
}

//sys	readlen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_read
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, err error) = SYS_lseek
//sys	getcwd(buf []byte) (n int, err error)
//sys	sysctl(mib []_C_int, old *byte, oldlen *uintptr, new *byte, newlen uintptr) (err error)
//sysnb fork() (pid int, err error)
//sysnb execve(path *byte, argv **byte, envp **byte) (err error)
//sysnb exit(res int) (err error)
//sys   ptrace(request int, pid int, addr uintptr, data uintptr) (err error)
//sysnb getentropy(p []byte) (err error)
//sys   fstatat(fd int, path string, stat *Stat_t, flags int) (err error)
//sys   unlinkat(fd int, path string, flags int) (err error)
//sys   openat(fd int, path string, flags int, perm uint32) (fdret int, err error)
