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
	for i, v := range s {
		if v == 0 {
			s = s[0:i]
			break
		}
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
//sys	FindFirstFile(name *uint16, data *Win32finddata) (handle int32, errno int) [failretval=-1] = FindFirstFileW
//sys	FindNextFile(handle int32, data *Win32finddata) (ok bool, errno int) = FindNextFileW
//sys	FindClose(handle int32) (ok bool, errno int)
//sys	GetFileInformationByHandle(handle int32, data *ByHandleFileInformation) (ok bool, errno int)
//sys	GetCurrentDirectory(buflen uint32, buf *uint16) (n uint32, errno int) = GetCurrentDirectoryW
//sys	SetCurrentDirectory(path *uint16) (ok bool, errno int) = SetCurrentDirectoryW
//sys	CreateDirectory(path *uint16, sa *byte) (ok bool, errno int) = CreateDirectoryW
//sys	RemoveDirectory(path *uint16) (ok bool, errno int) = RemoveDirectoryW
//sys	DeleteFile(path *uint16) (ok bool, errno int) = DeleteFileW
//sys	MoveFile(from *uint16, to *uint16) (ok bool, errno int) = MoveFileW
//sys	GetComputerName(buf *uint16, n *uint32) (ok bool, errno int) = GetComputerNameW
//sys	SetEndOfFile(handle int32) (ok bool, errno int)
//sys	GetSystemTimeAsFileTime(time *Filetime)
//sys   sleep(msec uint32) = Sleep

// syscall interface implementation for other packages

func Sleep(nsec int64) (errno int) {
	nsec += 999999 // round up to milliseconds
	msec := uint32(nsec / 1e6)
	sleep(msec)
	errno = 0
	return
}

func Errstr(errno int) string {
	if errno == EMINGW {
		return "not supported by windows"
	}
	b := make([]uint16, 300)
	n, err := FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_ARGUMENT_ARRAY, 0, uint32(errno), 0, b, nil)
	if err != 0 {
		return "error " + str(errno) + " (FormatMessage failed with err=" + str(err) + ")"
	}
	return string(utf16.Decode(b[0 : n-1]))
}

func Exit(code int) { ExitProcess(uint32(code)) }

func Open(path string, mode int, perm int) (fd int, errno int) {
	if len(path) == 0 {
		return -1, ERROR_FILE_NOT_FOUND
	}
	var access uint32
	switch mode & (O_RDONLY | O_WRONLY | O_RDWR) {
	case O_RDONLY:
		access = GENERIC_READ
	case O_WRONLY:
		access = GENERIC_WRITE
	case O_RDWR:
		access = GENERIC_READ | GENERIC_WRITE
	}
	if mode&O_CREAT != 0 {
		access |= GENERIC_WRITE
	}
	sharemode := uint32(FILE_SHARE_READ | FILE_SHARE_WRITE)
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
	curoffset, e := Seek(fd, 0, 1)
	if e != 0 {
		return 0, e
	}
	defer Seek(fd, curoffset, 0)
	var o Overlapped
	o.OffsetHigh = uint32(offset >> 32)
	o.Offset = uint32(offset)
	var done uint32
	if ok, e := ReadFile(int32(fd), p, &done, &o); !ok {
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
	curoffset, e := Seek(fd, 0, 1)
	if e != 0 {
		return 0, e
	}
	defer Seek(fd, curoffset, 0)
	var o Overlapped
	o.OffsetHigh = uint32(offset >> 32)
	o.Offset = uint32(offset)
	var done uint32
	if ok, e := WriteFile(int32(fd), p, &done, &o); !ok {
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

func Stat(path string, stat *Stat_t) (errno int) {
	h, e := FindFirstFile(StringToUTF16Ptr(path), &stat.Windata)
	if e != 0 {
		return e
	}
	defer FindClose(h)
	stat.Mode = 0
	return 0
}

func Lstat(path string, stat *Stat_t) (errno int) {
	// no links on windows, just call Stat
	return Stat(path, stat)
}

const ImplementsGetwd = true

func Getwd() (wd string, errno int) {
	b := make([]uint16, 300)
	n, e := GetCurrentDirectory(uint32(len(b)), &b[0])
	if e != 0 {
		return "", e
	}
	return string(utf16.Decode(b[0:n])), 0
}

func Chdir(path string) (errno int) {
	if ok, e := SetCurrentDirectory(&StringToUTF16(path)[0]); !ok {
		return e
	}
	return 0
}

func Mkdir(path string, mode int) (errno int) {
	if ok, e := CreateDirectory(&StringToUTF16(path)[0], nil); !ok {
		return e
	}
	return 0
}

func Rmdir(path string) (errno int) {
	if ok, e := RemoveDirectory(&StringToUTF16(path)[0]); !ok {
		return e
	}
	return 0
}

func Unlink(path string) (errno int) {
	if ok, e := DeleteFile(&StringToUTF16(path)[0]); !ok {
		return e
	}
	return 0
}

func Rename(oldpath, newpath string) (errno int) {
	from := &StringToUTF16(oldpath)[0]
	to := &StringToUTF16(newpath)[0]
	if ok, e := MoveFile(from, to); !ok {
		return e
	}
	return 0
}

func ComputerName() (name string, errno int) {
	var n uint32 = MAX_COMPUTERNAME_LENGTH + 1
	b := make([]uint16, n)
	if ok, e := GetComputerName(&b[0], &n); !ok {
		return "", e
	}
	return string(utf16.Decode(b[0:n])), 0
}

func Ftruncate(fd int, length int64) (errno int) {
	curoffset, e := Seek(fd, 0, 1)
	if e != 0 {
		return e
	}
	defer Seek(fd, curoffset, 0)
	if _, e := Seek(fd, length, 0); e != 0 {
		return e
	}
	if _, e := SetEndOfFile(int32(fd)); e != 0 {
		return e
	}
	return 0
}

func Gettimeofday(tv *Timeval) (errno int) {
	var ft Filetime
	// 100-nanosecond intervals since January 1, 1601
	GetSystemTimeAsFileTime(&ft)
	t := uint64(ft.HighDateTime)<<32 + uint64(ft.LowDateTime)
	// convert into microseconds
	t /= 10
	// change starting time to the Epoch (00:00:00 UTC, January 1, 1970)
	t -= 11644473600000000
	// split into sec / usec
	tv.Sec = int32(t / 1e6)
	tv.Usec = int32(t) - tv.Sec
	return 0
}

// TODO(brainman): fix all needed for os

const (
	SIGTRAP = 5
)

func Getpid() (pid int)   { return -1 }
func Getppid() (ppid int) { return -1 }

func Fchdir(fd int) (errno int)                           { return EMINGW }
func Link(oldpath, newpath string) (errno int)            { return EMINGW }
func Symlink(path, link string) (errno int)               { return EMINGW }
func Readlink(path string, buf []byte) (n int, errno int) { return 0, EMINGW }
func Chmod(path string, mode int) (errno int)             { return EMINGW }
func Fchmod(fd int, mode int) (errno int)                 { return EMINGW }
func Chown(path string, uid int, gid int) (errno int)     { return EMINGW }
func Lchown(path string, uid int, gid int) (errno int)    { return EMINGW }
func Fchown(fd int, uid int, gid int) (errno int)         { return EMINGW }

func Getuid() (uid int)                  { return -1 }
func Geteuid() (euid int)                { return -1 }
func Getgid() (gid int)                  { return -1 }
func Getegid() (egid int)                { return -1 }
func Getgroups() (gids []int, errno int) { return nil, EMINGW }

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
