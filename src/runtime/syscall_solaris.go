// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var (
	libc_chdir,
	libc_chroot,
	libc_close,
	libc_dlopen,
	libc_dlclose,
	libc_dlsym,
	libc_execve,
	libc_exit,
	libc_fcntl,
	libc_forkx,
	libc_gethostname,
	libc_ioctl,
	libc_pipe,
	libc_setgid,
	libc_setgroups,
	libc_setsid,
	libc_setuid,
	libc_setpgid,
	libc_syscall,
	libc_wait4,
	libc_write,
	pipe1 libcFunc
)

//go:nosplit
func syscall_sysvicall6(fn, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	call := libcall{
		fn:   fn,
		n:    nargs,
		args: uintptr(unsafe.Pointer(&a1)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	return call.r1, call.r2, call.err
}

//go:nosplit
func syscall_rawsysvicall6(fn, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	call := libcall{
		fn:   fn,
		n:    nargs,
		args: uintptr(unsafe.Pointer(&a1)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.r1, call.r2, call.err
}

// TODO(aram): Once we remove all instances of C calling sysvicallN, make
// sysvicallN return errors and replace the body of the following functions
// with calls to sysvicallN.

//go:nosplit
func syscall_chdir(path uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_chdir)),
		n:    1,
		args: uintptr(unsafe.Pointer(&path)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
func syscall_chroot(path uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_chroot)),
		n:    1,
		args: uintptr(unsafe.Pointer(&path)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

// like close, but must not split stack, for forkx.
//go:nosplit
func syscall_close(fd int32) int32 {
	return int32(sysvicall1(&libc_close, uintptr(fd)))
}

func syscall_dlopen(name *byte, mode uintptr) (handle uintptr, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_dlopen)),
		n:    2,
		args: uintptr(unsafe.Pointer(&name)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	if call.r1 == 0 {
		return call.r1, call.err
	}
	return call.r1, 0
}

func syscall_dlclose(handle uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_dlclose)),
		n:    1,
		args: uintptr(unsafe.Pointer(&handle)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	return call.r1
}

func syscall_dlsym(handle uintptr, name *byte) (proc uintptr, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_dlsym)),
		n:    2,
		args: uintptr(unsafe.Pointer(&handle)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	if call.r1 == 0 {
		return call.r1, call.err
	}
	return call.r1, 0
}

//go:nosplit
func syscall_execve(path, argv, envp uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_execve)),
		n:    3,
		args: uintptr(unsafe.Pointer(&path)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

// like exit, but must not split stack, for forkx.
//go:nosplit
func syscall_exit(code uintptr) {
	sysvicall1(&libc_exit, code)
}

//go:nosplit
func syscall_fcntl(fd, cmd, arg uintptr) (val, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_fcntl)),
		n:    3,
		args: uintptr(unsafe.Pointer(&fd)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.r1, call.err
}

//go:nosplit
func syscall_forkx(flags uintptr) (pid uintptr, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_forkx)),
		n:    1,
		args: uintptr(unsafe.Pointer(&flags)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.r1, call.err
}

func syscall_gethostname() (name string, err uintptr) {
	cname := new([_MAXHOSTNAMELEN]byte)
	var args = [2]uintptr{uintptr(unsafe.Pointer(&cname[0])), _MAXHOSTNAMELEN}
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_gethostname)),
		n:    2,
		args: uintptr(unsafe.Pointer(&args[0])),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	if call.r1 != 0 {
		return "", call.err
	}
	cname[_MAXHOSTNAMELEN-1] = 0
	return gostringnocopy(&cname[0]), 0
}

//go:nosplit
func syscall_ioctl(fd, req, arg uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_ioctl)),
		n:    3,
		args: uintptr(unsafe.Pointer(&fd)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

func syscall_pipe() (r, w, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&pipe1)),
		n:    0,
		args: uintptr(unsafe.Pointer(&pipe1)), // it's unused but must be non-nil, otherwise crashes
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	return call.r1, call.r2, call.err
}

// This is syscall.RawSyscall, it exists to satisfy some build dependency,
// but it doesn't work correctly.
//
// DO NOT USE!
//
// TODO(aram): make this panic once we stop calling fcntl(2) in net using it.
func syscall_rawsyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_syscall)),
		n:    4,
		args: uintptr(unsafe.Pointer(&trap)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.r1, call.r2, call.err
}

//go:nosplit
func syscall_setgid(gid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setgid)),
		n:    1,
		args: uintptr(unsafe.Pointer(&gid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
func syscall_setgroups(ngid, gid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setgroups)),
		n:    2,
		args: uintptr(unsafe.Pointer(&ngid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
func syscall_setsid() (pid, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setsid)),
		n:    0,
		args: uintptr(unsafe.Pointer(&libc_setsid)), // it's unused but must be non-nil, otherwise crashes
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.r1, call.err
}

//go:nosplit
func syscall_setuid(uid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setuid)),
		n:    1,
		args: uintptr(unsafe.Pointer(&uid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
func syscall_setpgid(pid, pgid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setpgid)),
		n:    2,
		args: uintptr(unsafe.Pointer(&pid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.err
}

// This is syscall.Syscall, it exists to satisfy some build dependency,
// but it doesn't work correctly.
//
// DO NOT USE!
//
// TODO(aram): make this panic once we stop calling fcntl(2) in net using it.
func syscall_syscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_syscall)),
		n:    4,
		args: uintptr(unsafe.Pointer(&trap)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	return call.r1, call.r2, call.err
}

func syscall_wait4(pid uintptr, wstatus *uint32, options uintptr, rusage unsafe.Pointer) (wpid int, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_wait4)),
		n:    4,
		args: uintptr(unsafe.Pointer(&pid)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	exitsyscall()
	return int(call.r1), call.err
}

//go:nosplit
func syscall_write(fd, buf, nbyte uintptr) (n, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_write)),
		n:    3,
		args: uintptr(unsafe.Pointer(&fd)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6), unsafe.Pointer(&call))
	return call.r1, call.err
}
