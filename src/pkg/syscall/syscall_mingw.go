// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows system calls.

package syscall

import (
	"unsafe"
	"utf16"
)

const OS = "mingw"

/*

small demo to detect version of windows you are running:

package main

import (
	"syscall"
)

func abort(funcname string, err int) {
	panic(funcname + " failed: " + syscall.Errstr(err))
}

func print_version(v uint32) {
	major := byte(v)
	minor := uint8(v >> 8)
	build := uint16(v >> 16)
	print("windows version ", major, ".", minor, " (Build ", build, ")\n")
}

func main() {
	h, err := syscall.LoadLibrary("kernel32.dll")
	if err != 0 {
		abort("LoadLibrary", err)
	}
	defer syscall.FreeLibrary(h)
	proc, err := syscall.GetProcAddress(h, "GetVersion")
	if err != 0 {
		abort("GetProcAddress", err)
	}
	r, _, _ := syscall.Syscall(uintptr(proc), 0, 0, 0)
	print_version(uint32(r))
}

*/

// StringToUTF16 returns the UTF-16 encoding of the UTF-8 string s,
// with a terminating NUL added.
func StringToUTF16(s string) []uint16 { return utf16.Encode([]int(s + "\x00")) }

// UTF16ToString returns the UTF-8 encoding of the UTF-16 sequence s,
// with a terminating NUL removed.
func UTF16ToString(s []uint16) string {
	if n := len(s); n > 0 && s[n-1] == 0 {
		s = s[0 : n-1]
	}
	return string(utf16.Decode(s))
}

// StringToUTF16Ptr returns pointer to the UTF-16 encoding of
// the UTF-8 string s, with a terminating NUL added.
func StringToUTF16Ptr(s string) *uint16 { return &StringToUTF16(s)[0] }

// dll helpers

// implemented in ../pkg/runtime/mingw/syscall.cgo
func Syscall9(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, lasterr uintptr)
func loadlibraryex(filename uintptr) (handle uint32)
func getprocaddress(handle uint32, procname uintptr) (proc uintptr)

func loadDll(fname string) uint32 {
	m := loadlibraryex(uintptr(unsafe.Pointer(StringBytePtr(fname))))
	if m == 0 {
		panic("syscall: could not LoadLibraryEx " + fname)
	}
	return m
}

func getSysProcAddr(m uint32, pname string) uintptr {
	p := getprocaddress(m, uintptr(unsafe.Pointer(StringBytePtr(pname))))
	if p == 0 {
		panic("syscall: could not GetProcAddress for " + pname)
	}
	return p
}

// windows api calls

//sys	GetLastError() (lasterrno int)
//sys	LoadLibrary(libname string) (handle uint32, errno int) = LoadLibraryW
//sys	FreeLibrary(handle uint32) (ok bool, errno int)
//sys	GetProcAddress(module uint32, procname string) (proc uint32, errno int)
//sys	GetVersion() (ver uint32, errno int)
//sys	FormatMessage(flags uint32, msgsrc uint32, msgid uint32, langid uint32, buf []uint16, args *byte) (n uint32, errno int) = FormatMessageW

func Errstr(errno int) string {
	if errno == EMINGW {
		return "not supported by windows"
	}
	var b = make([]uint16, 300)
	n, err := FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_ARGUMENT_ARRAY, 0, uint32(errno), 0, b, nil)
	if err != 0 {
		return "error " + str(errno) + " (FormatMessage failed with err=" + str(err) + ")"
	}
	return UTF16ToString(b[0 : n-1])
}

// TODO(brainman): fix all this meaningless code, it is here to compile exec.go

func Pipe(p []int) (errno int) { return EMINGW }

func Close(fd int) (errno int) { return EMINGW }
func read(fd int, buf *byte, nbuf int) (n int, errno int) {
	return 0, EMINGW
}

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
