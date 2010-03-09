// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows system calls.

package syscall

import "unsafe"

const OS = "mingw"

/*

small demo to detect version of windows you are running:

package main

import (
	"syscall"
)

func print_version(v uint32) {
	major := byte(v)
	minor := uint8(v >> 8)
	build := uint16(v >> 16)
	print("windows version ", major, ".", minor, " (Build ", build, ")\n")
}

func main() {
	h, err := syscall.LoadLibrary("kernel32.dll")
	if err != 0 {
		panic("failed to LoadLibrary #", err, "\n")
	}
	defer syscall.FreeLibrary(h)
	proc, err := syscall.GetProcAddress(h, "GetVersion")
	if err != 0 {
		panic("could not GetProcAddress #", err, "\n")
	}
	r, _, e := syscall.Syscall(uintptr(proc), 0, 0, 0)
	err = int(e)
	if err != 0 {
		panic("GetVersion failed #", err, "\n")
	}
	print_version(uint32(r))
}

*/

//sys	GetLastError() (lasterrno int)

// TODO(brainman): probably should use LoadLibraryW here instead
//sys	LoadLibraryA(libname string) (handle Module, errno int)

func LoadLibrary(libname string) (handle Module, errno int) {
	h, e := LoadLibraryA(libname)
	if int(h) != 0 {
		return h, 0
	}
	return h, e
}

// TODO(brainman): should handle errors like in LoadLibrary, otherwise will be returning 'old' errors
//sys	FreeLibrary(handle Module) (ok Bool, errno int)
//sys	GetProcAddress(module Module, procname string) (proc uint32, errno int)
//sys	GetVersion() (ver uint32, errno int)

// dll helpers

// implemented in ../pkg/runtime/mingw/syscall.cgo
func loadlibraryex(filename uintptr) (handle uint32)
func getprocaddress(handle uint32, procname uintptr) (proc uintptr)

func loadDll(fname string) Module {
	m := loadlibraryex(uintptr(unsafe.Pointer(StringBytePtr(fname))))
	if m == 0 {
		panic("syscall: could not LoadLibraryEx ", fname)
	}
	return Module(m)
}

func getSysProcAddr(m Module, pname string) uintptr {
	p := getprocaddress(uint32(m), uintptr(unsafe.Pointer(StringBytePtr(pname))))
	if p == 0 {
		panic("syscall: could not GetProcAddress for ", pname)
	}
	return p
}

// TODO(brainman): fix all this meaningless code, it is here to compile exec.go

func Pipe(p []int) (errno int) { return EMINGW }

//sys	Close(fd int) (errno int)
//sys	read(fd int, buf *byte, nbuf int) (n int, errno int)

func fcntl(fd, cmd, arg int) (val int, errno int) {
	return 0, EMINGW
}

const (
	F_SETFD = 1 + iota
	FD_CLOEXEC
	F_GETFL
	F_SETFL
	O_NONBLOCK
	SYS_FORK
	SYS_PTRACE
	SYS_CHDIR
	SYS_DUP2
	SYS_FCNTL
	SYS_EXECVE
	PTRACE_TRACEME
	SYS_CLOSE
	SYS_WRITE
	SYS_EXIT
	SYS_READ
	EPIPE
	EINTR
)

type Rusage struct {
	Utime    Timeval
	Stime    Timeval
	Maxrss   int32
	Ixrss    int32
	Idrss    int32
	Isrss    int32
	Minflt   int32
	Majflt   int32
	Nswap    int32
	Inblock  int32
	Oublock  int32
	Msgsnd   int32
	Msgrcv   int32
	Nsignals int32
	Nvcsw    int32
	Nivcsw   int32
}

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, errno int) {
	return 0, EMINGW
}

type WaitStatus uint32

func (WaitStatus) Exited() bool { return false }

func (WaitStatus) ExitStatus() int { return -1 }

func (WaitStatus) Signal() int { return -1 }

func (WaitStatus) CoreDump() bool { return false }

func (WaitStatus) Stopped() bool { return false }

func (WaitStatus) Continued() bool { return false }

func (WaitStatus) StopSignal() int { return -1 }

func (WaitStatus) Signaled() bool { return false }

func (WaitStatus) TrapCause() int { return -1 }
