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
//sys	ExitProcess(exitcode uint32)
//sys	CreateFile(name *uint16, access uint32, mode uint32, sa *byte, createmode uint32, attrs uint32, templatefile int32) (handle int32, errno int) [failretval=-1] = CreateFileW
//sys	ReadFile(handle int32, buf []byte, done *uint32, overlapped *Overlapped) (ok bool, errno int)
//sys	WriteFile(handle int32, buf []byte, done *uint32, overlapped *Overlapped) (ok bool, errno int)
//sys	SetFilePointer(handle int32, lowoffset int32, highoffsetptr *int32, whence uint32) (newlowoffset uint32, errno int) [failretval=0xffffffff]
//sys	CloseHandle(handle int32) (ok bool, errno int)
//sys	GetStdHandle(stdhandle int32) (handle int32, errno int) [failretval=-1]

// syscall interface implementation for other packages

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

func Exit(code int) { ExitProcess(uint32(code)) }

func Open(path string, mode int, perm int) (fd int, errno int) {
	if len(path) == 0 {
		return -1, ERROR_FILE_NOT_FOUND
	}
	var access, sharemode uint32
	switch {
	case mode&O_CREAT != 0:
		access = GENERIC_READ | GENERIC_WRITE
		sharemode = 0
	case mode&O_RDWR == O_RDONLY:
		access = GENERIC_READ
		sharemode = FILE_SHARE_READ
	case mode&O_RDWR == O_WRONLY:
		access = GENERIC_WRITE
		sharemode = FILE_SHARE_READ
	case mode&O_RDWR == O_RDWR:
		access = GENERIC_READ | GENERIC_WRITE
		sharemode = FILE_SHARE_READ | FILE_SHARE_WRITE
	}
	var createmode uint32
	switch {
	case mode&O_CREAT != 0:
		if mode&O_EXCL != 0 {
			createmode = CREATE_NEW
		} else {
			createmode = CREATE_ALWAYS
		}
	case mode&O_TRUNC != 0:
		createmode = TRUNCATE_EXISTING
	default:
		createmode = OPEN_EXISTING
	}
	h, e := CreateFile(StringToUTF16Ptr(path), access, sharemode, nil, createmode, FILE_ATTRIBUTE_NORMAL, 0)
	return int(h), int(e)
}

func Read(fd int, p []byte) (n int, errno int) {
	var done uint32
	if ok, e := ReadFile(int32(fd), p, &done, nil); !ok {
		return 0, e
	}
	return int(done), 0
}

// TODO(brainman): ReadFile/WriteFile change file offset, therefore
// i use Seek here to preserve semantics of unix pread/pwrite,
// not sure if I should do that

func Pread(fd int, p []byte, offset int64) (n int, errno int) {
	var o Overlapped
	o.OffsetHigh = uint32(offset >> 32)
	o.Offset = uint32(offset)
	curoffset, e := Seek(fd, 0, 1)
	if e != 0 {
		return 0, e
	}
	var done uint32
	if ok, e := ReadFile(int32(fd), p, &done, &o); !ok {
		return 0, e
	}
	_, e = Seek(fd, curoffset, 0)
	if e != 0 {
		return 0, e
	}
	return int(done), 0
}

func Write(fd int, p []byte) (n int, errno int) {
	var done uint32
	if ok, e := WriteFile(int32(fd), p, &done, nil); !ok {
		return 0, e
	}
	return int(done), 0
}

func Pwrite(fd int, p []byte, offset int64) (n int, errno int) {
	var o Overlapped
	o.OffsetHigh = uint32(offset >> 32)
	o.Offset = uint32(offset)
	curoffset, e := Seek(fd, 0, 1)
	if e != 0 {
		return 0, e
	}
	var done uint32
	if ok, e := WriteFile(int32(fd), p, &done, &o); !ok {
		return 0, e
	}
	_, e = Seek(fd, curoffset, 0)
	if e != 0 {
		return 0, e
	}
	return int(done), 0
}

func Seek(fd int, offset int64, whence int) (newoffset int64, errno int) {
	var w uint32
	switch whence {
	case 0:
		w = FILE_BEGIN
	case 1:
		w = FILE_CURRENT
	case 2:
		w = FILE_END
	}
	hi := int32(offset >> 32)
	lo := int32(offset)
	rlo, e := SetFilePointer(int32(fd), lo, &hi, w)
	if e != 0 {
		return 0, e
	}
	return int64(hi)<<32 + int64(rlo), 0
}

func Close(fd int) (errno int) {
	if ok, e := CloseHandle(int32(fd)); !ok {
		return e
	}
	return 0
}

var (
	Stdin  = getStdHandle(STD_INPUT_HANDLE)
	Stdout = getStdHandle(STD_OUTPUT_HANDLE)
	Stderr = getStdHandle(STD_ERROR_HANDLE)
)

func getStdHandle(h int32) (fd int) {
	r, _ := GetStdHandle(h)
	return int(r)
}

// TODO(brainman): fix all needed for os

const (
	SIGTRAP = 5
)

func Getdents(fd int, buf []byte) (n int, errno int) { return 0, EMINGW }

func Getpid() (pid int)   { return -1 }
func Getppid() (ppid int) { return -1 }

func Mkdir(path string, mode int) (errno int)             { return EMINGW }
func Lstat(path string, stat *Stat_t) (errno int)         { return EMINGW }
func Stat(path string, stat *Stat_t) (errno int)          { return EMINGW }
func Fstat(fd int, stat *Stat_t) (errno int)              { return EMINGW }
func Chdir(path string) (errno int)                       { return EMINGW }
func Fchdir(fd int) (errno int)                           { return EMINGW }
func Unlink(path string) (errno int)                      { return EMINGW }
func Rmdir(path string) (errno int)                       { return EMINGW }
func Link(oldpath, newpath string) (errno int)            { return EMINGW }
func Symlink(path, link string) (errno int)               { return EMINGW }
func Readlink(path string, buf []byte) (n int, errno int) { return 0, EMINGW }
func Rename(oldpath, newpath string) (errno int)          { return EMINGW }
func Chmod(path string, mode int) (errno int)             { return EMINGW }
func Fchmod(fd int, mode int) (errno int)                 { return EMINGW }
func Chown(path string, uid int, gid int) (errno int)     { return EMINGW }
func Lchown(path string, uid int, gid int) (errno int)    { return EMINGW }
func Fchown(fd int, uid int, gid int) (errno int)         { return EMINGW }
func Truncate(name string, size int64) (errno int)        { return EMINGW }
func Ftruncate(fd int, length int64) (errno int)          { return EMINGW }

const ImplementsGetwd = true

func Getwd() (wd string, errno int)        { return "", EMINGW }
func Getuid() (uid int)                    { return -1 }
func Geteuid() (euid int)                  { return -1 }
func Getgid() (gid int)                    { return -1 }
func Getegid() (egid int)                  { return -1 }
func Getgroups() (gids []int, errno int)   { return nil, EMINGW }
func Gettimeofday(tv *Timeval) (errno int) { return EMINGW }

// TODO(brainman): fix all this meaningless code, it is here to compile exec.go

func Pipe(p []int) (errno int) { return EMINGW }

func read(fd int, buf *byte, nbuf int) (n int, errno int) {
	return 0, EMINGW
}

func fcntl(fd, cmd, arg int) (val int, errno int) {
	return 0, EMINGW
}

const (
	PTRACE_TRACEME = 1 + iota
	WNOHANG
	WSTOPPED
	SYS_CLOSE
	SYS_WRITE
	SYS_EXIT
	SYS_READ
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
