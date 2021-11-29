// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var (
	libc_chdir,
	libc_chroot,
	libc_close,
	libc_execve,
	libc_fcntl,
	libc_forkx,
	libc_gethostname,
	libc_getpid,
	libc_ioctl,
	libc_setgid,
	libc_setgroups,
	libc_setsid,
	libc_setuid,
	libc_setpgid,
	libc_syscall,
	libc_wait4 libcFunc
)

//go:linkname pipe1x runtime.pipe1
var pipe1x libcFunc // name to take addr of pipe1

func pipe1() // declared for vet; do NOT call

// Many of these are exported via linkname to assembly in the syscall
// package.

//go:nosplit
//go:linkname syscall_sysvicall6
//go:cgo_unsafe_args
func syscall_sysvicall6(fn, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	call := libcall{
		fn:   fn,
		n:    nargs,
		args: uintptr(unsafe.Pointer(&a1)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	exitsyscall()
	return call.r1, call.r2, call.err
}

//go:nosplit
//go:linkname syscall_rawsysvicall6
//go:cgo_unsafe_args
func syscall_rawsysvicall6(fn, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	call := libcall{
		fn:   fn,
		n:    nargs,
		args: uintptr(unsafe.Pointer(&a1)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.r1, call.r2, call.err
}

// TODO(aram): Once we remove all instances of C calling sysvicallN, make
// sysvicallN return errors and replace the body of the following functions
// with calls to sysvicallN.

//go:nosplit
//go:linkname syscall_chdir
func syscall_chdir(path uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_chdir)),
		n:    1,
		args: uintptr(unsafe.Pointer(&path)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
//go:linkname syscall_chroot
func syscall_chroot(path uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_chroot)),
		n:    1,
		args: uintptr(unsafe.Pointer(&path)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

// like close, but must not split stack, for forkx.
//go:nosplit
//go:linkname syscall_close
func syscall_close(fd int32) int32 {
	return int32(sysvicall1(&libc_close, uintptr(fd)))
}

const _F_DUP2FD = 0x9

//go:nosplit
//go:linkname syscall_dup2
func syscall_dup2(oldfd, newfd uintptr) (val, err uintptr) {
	return syscall_fcntl(oldfd, _F_DUP2FD, newfd)
}

//go:nosplit
//go:linkname syscall_execve
//go:cgo_unsafe_args
func syscall_execve(path, argv, envp uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_execve)),
		n:    3,
		args: uintptr(unsafe.Pointer(&path)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

// like exit, but must not split stack, for forkx.
//go:nosplit
//go:linkname syscall_exit
func syscall_exit(code uintptr) {
	sysvicall1(&libc_exit, code)
}

//go:nosplit
//go:linkname syscall_fcntl
//go:cgo_unsafe_args
func syscall_fcntl(fd, cmd, arg uintptr) (val, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_fcntl)),
		n:    3,
		args: uintptr(unsafe.Pointer(&fd)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.r1, call.err
}

//go:nosplit
//go:linkname syscall_forkx
func syscall_forkx(flags uintptr) (pid uintptr, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_forkx)),
		n:    1,
		args: uintptr(unsafe.Pointer(&flags)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	if int(call.r1) != -1 {
		call.err = 0
	}
	return call.r1, call.err
}

//go:linkname syscall_gethostname
func syscall_gethostname() (name string, err uintptr) {
	cname := new([_MAXHOSTNAMELEN]byte)
	var args = [2]uintptr{uintptr(unsafe.Pointer(&cname[0])), _MAXHOSTNAMELEN}
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_gethostname)),
		n:    2,
		args: uintptr(unsafe.Pointer(&args[0])),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	exitsyscall()
	if call.r1 != 0 {
		return "", call.err
	}
	cname[_MAXHOSTNAMELEN-1] = 0
	return gostringnocopy(&cname[0]), 0
}

//go:nosplit
//go:linkname syscall_getpid
func syscall_getpid() (pid, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_getpid)),
		n:    0,
		args: uintptr(unsafe.Pointer(&libc_getpid)), // it's unused but must be non-nil, otherwise crashes
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.r1, call.err
}

//go:nosplit
//go:linkname syscall_ioctl
//go:cgo_unsafe_args
func syscall_ioctl(fd, req, arg uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_ioctl)),
		n:    3,
		args: uintptr(unsafe.Pointer(&fd)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

//go:linkname syscall_pipe
func syscall_pipe() (r, w, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&pipe1x)),
		n:    0,
		args: uintptr(unsafe.Pointer(&pipe1x)), // it's unused but must be non-nil, otherwise crashes
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	exitsyscall()
	return call.r1, call.r2, call.err
}

// This is syscall.RawSyscall, it exists to satisfy some build dependency,
// but it doesn't work.
//
//go:linkname syscall_rawsyscall
func syscall_rawsyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	panic("RawSyscall not available on Solaris")
}

// This is syscall.RawSyscall6, it exists to avoid a linker error because
// syscall.RawSyscall6 is already declared. See golang.org/issue/24357
//
//go:linkname syscall_rawsyscall6
func syscall_rawsyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	panic("RawSyscall6 not available on Solaris")
}

//go:nosplit
//go:linkname syscall_setgid
func syscall_setgid(gid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setgid)),
		n:    1,
		args: uintptr(unsafe.Pointer(&gid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
//go:linkname syscall_setgroups
//go:cgo_unsafe_args
func syscall_setgroups(ngid, gid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setgroups)),
		n:    2,
		args: uintptr(unsafe.Pointer(&ngid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
//go:linkname syscall_setsid
func syscall_setsid() (pid, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setsid)),
		n:    0,
		args: uintptr(unsafe.Pointer(&libc_setsid)), // it's unused but must be non-nil, otherwise crashes
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.r1, call.err
}

//go:nosplit
//go:linkname syscall_setuid
func syscall_setuid(uid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setuid)),
		n:    1,
		args: uintptr(unsafe.Pointer(&uid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

//go:nosplit
//go:linkname syscall_setpgid
//go:cgo_unsafe_args
func syscall_setpgid(pid, pgid uintptr) (err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_setpgid)),
		n:    2,
		args: uintptr(unsafe.Pointer(&pid)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.err
}

//go:linkname syscall_syscall
//go:cgo_unsafe_args
func syscall_syscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_syscall)),
		n:    4,
		args: uintptr(unsafe.Pointer(&trap)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	exitsyscall()
	return call.r1, call.r2, call.err
}

//go:linkname syscall_wait4
//go:cgo_unsafe_args
func syscall_wait4(pid uintptr, wstatus *uint32, options uintptr, rusage unsafe.Pointer) (wpid int, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_wait4)),
		n:    4,
		args: uintptr(unsafe.Pointer(&pid)),
	}
	entersyscallblock()
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	exitsyscall()
	KeepAlive(wstatus)
	KeepAlive(rusage)
	return int(call.r1), call.err
}

//go:nosplit
//go:linkname syscall_write
//go:cgo_unsafe_args
func syscall_write(fd, buf, nbyte uintptr) (n, err uintptr) {
	call := libcall{
		fn:   uintptr(unsafe.Pointer(&libc_write)),
		n:    3,
		args: uintptr(unsafe.Pointer(&fd)),
	}
	asmcgocall(unsafe.Pointer(&asmsysvicall6x), unsafe.Pointer(&call))
	return call.r1, call.err
}
