// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x

// Many of the following syscalls are not available on all versions of z/OS.
// Some missing calls have legacy implementations/simulations but others
// will be missing completely. To achieve consistent failing behaviour on
// legacy systems, we first test the function pointer via a safeloading
// mechanism to see if the function exists on a given system. Then execution
// is branched to either continue the function call, or return an error.

package unix

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

//go:noescape
func initZosLibVec()

//go:noescape
func GetZosLibVec() uintptr

func init() {
	initZosLibVec()
	r0, _, _ := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS_____GETENV_A<<4, uintptr(unsafe.Pointer(&([]byte("__ZOS_XSYSTRACE\x00"))[0])))
	if r0 != 0 {
		n, _, _ := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS___ATOI_A<<4, r0)
		ZosTraceLevel = int(n)
		r0, _, _ := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS_____GETENV_A<<4, uintptr(unsafe.Pointer(&([]byte("__ZOS_XSYSTRACEFD\x00"))[0])))
		if r0 != 0 {
			fd, _, _ := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS___ATOI_A<<4, r0)
			f := os.NewFile(fd, "zostracefile")
			if f != nil {
				ZosTracefile = f
			}
		}

	}
}

//go:noescape
func CallLeFuncWithErr(funcdesc uintptr, parms ...uintptr) (ret, errno2 uintptr, err Errno)

//go:noescape
func CallLeFuncWithPtrReturn(funcdesc uintptr, parms ...uintptr) (ret, errno2 uintptr, err Errno)

// -------------------------------
// pointer validity test
// good pointer returns 0
// bad pointer returns 1
//
//go:nosplit
func ptrtest(uintptr) uint64

// Load memory at ptr location with error handling if the location is invalid
//
//go:noescape
func safeload(ptr uintptr) (value uintptr, error uintptr)

const (
	entrypointLocationOffset = 8 // From function descriptor

	xplinkEyecatcher   = 0x00c300c500c500f1 // ".C.E.E.1"
	eyecatcherOffset   = 16                 // From function entrypoint (negative)
	ppa1LocationOffset = 8                  // From function entrypoint (negative)

	nameLenOffset = 0x14 // From PPA1 start
	nameOffset    = 0x16 // From PPA1 start
)

func getPpaOffset(funcptr uintptr) int64 {
	entrypoint, err := safeload(funcptr + entrypointLocationOffset)
	if err != 0 {
		return -1
	}

	// XPLink functions have ".C.E.E.1" as the first 8 bytes (EBCDIC)
	val, err := safeload(entrypoint - eyecatcherOffset)
	if err != 0 {
		return -1
	}
	if val != xplinkEyecatcher {
		return -1
	}

	ppaoff, err := safeload(entrypoint - ppa1LocationOffset)
	if err != 0 {
		return -1
	}

	ppaoff >>= 32
	return int64(ppaoff)
}

//-------------------------------
// function descriptor pointer validity test
// good pointer returns 0
// bad pointer returns 1

// TODO: currently mksyscall_zos_s390x.go generate empty string for funcName
// have correct funcName pass to the funcptrtest function
func funcptrtest(funcptr uintptr, funcName string) uint64 {
	entrypoint, err := safeload(funcptr + entrypointLocationOffset)
	if err != 0 {
		return 1
	}

	ppaoff := getPpaOffset(funcptr)
	if ppaoff == -1 {
		return 1
	}

	// PPA1 offset value is from the start of the entire function block, not the entrypoint
	ppa1 := (entrypoint - eyecatcherOffset) + uintptr(ppaoff)

	nameLen, err := safeload(ppa1 + nameLenOffset)
	if err != 0 {
		return 1
	}

	nameLen >>= 48
	if nameLen > 128 {
		return 1
	}

	// no function name input to argument end here
	if funcName == "" {
		return 0
	}

	var funcname [128]byte
	for i := 0; i < int(nameLen); i += 8 {
		v, err := safeload(ppa1 + nameOffset + uintptr(i))
		if err != 0 {
			return 1
		}
		funcname[i] = byte(v >> 56)
		funcname[i+1] = byte(v >> 48)
		funcname[i+2] = byte(v >> 40)
		funcname[i+3] = byte(v >> 32)
		funcname[i+4] = byte(v >> 24)
		funcname[i+5] = byte(v >> 16)
		funcname[i+6] = byte(v >> 8)
		funcname[i+7] = byte(v)
	}

	runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___E2A_L<<4, // __e2a_l
		[]uintptr{uintptr(unsafe.Pointer(&funcname[0])), nameLen})

	name := string(funcname[:nameLen])
	if name != funcName {
		return 1
	}

	return 0
}

// For detection of capabilities on a system.
// Is function descriptor f a valid function?
func isValidLeFunc(f uintptr) error {
	ret := funcptrtest(f, "")
	if ret != 0 {
		return fmt.Errorf("Bad pointer, not an LE function ")
	}
	return nil
}

// Retrieve function name from descriptor
func getLeFuncName(f uintptr) (string, error) {
	// assume it has been checked, only check ppa1 validity here
	entry := ((*[2]uintptr)(unsafe.Pointer(f)))[1]
	preamp := ((*[4]uint32)(unsafe.Pointer(entry - eyecatcherOffset)))

	offsetPpa1 := preamp[2]
	if offsetPpa1 > 0x0ffff {
		return "", fmt.Errorf("PPA1 offset seems too big 0x%x\n", offsetPpa1)
	}

	ppa1 := uintptr(unsafe.Pointer(preamp)) + uintptr(offsetPpa1)
	res := ptrtest(ppa1)
	if res != 0 {
		return "", fmt.Errorf("PPA1 address not valid")
	}

	size := *(*uint16)(unsafe.Pointer(ppa1 + nameLenOffset))
	if size > 128 {
		return "", fmt.Errorf("Function name seems too long, length=%d\n", size)
	}

	var name [128]byte
	funcname := (*[128]byte)(unsafe.Pointer(ppa1 + nameOffset))
	copy(name[0:size], funcname[0:size])

	runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___E2A_L<<4, // __e2a_l
		[]uintptr{uintptr(unsafe.Pointer(&name[0])), uintptr(size)})

	return string(name[:size]), nil
}

// Check z/OS version
func zosLeVersion() (version, release uint32) {
	p1 := (*(*uintptr)(unsafe.Pointer(uintptr(1208)))) >> 32
	p1 = *(*uintptr)(unsafe.Pointer(uintptr(p1 + 88)))
	p1 = *(*uintptr)(unsafe.Pointer(uintptr(p1 + 8)))
	p1 = *(*uintptr)(unsafe.Pointer(uintptr(p1 + 984)))
	vrm := *(*uint32)(unsafe.Pointer(p1 + 80))
	version = (vrm & 0x00ff0000) >> 16
	release = (vrm & 0x0000ff00) >> 8
	return
}

// returns a zos C FILE * for stdio fd 0, 1, 2
func ZosStdioFilep(fd int32) uintptr {
	return uintptr(*(*uint64)(unsafe.Pointer(uintptr(*(*uint64)(unsafe.Pointer(uintptr(*(*uint64)(unsafe.Pointer(uintptr(uint64(*(*uint32)(unsafe.Pointer(uintptr(1208)))) + 80))) + uint64((fd+2)<<3))))))))
}

func copyStat(stat *Stat_t, statLE *Stat_LE_t) {
	stat.Dev = uint64(statLE.Dev)
	stat.Ino = uint64(statLE.Ino)
	stat.Nlink = uint64(statLE.Nlink)
	stat.Mode = uint32(statLE.Mode)
	stat.Uid = uint32(statLE.Uid)
	stat.Gid = uint32(statLE.Gid)
	stat.Rdev = uint64(statLE.Rdev)
	stat.Size = statLE.Size
	stat.Atim.Sec = int64(statLE.Atim)
	stat.Atim.Nsec = 0 //zos doesn't return nanoseconds
	stat.Mtim.Sec = int64(statLE.Mtim)
	stat.Mtim.Nsec = 0 //zos doesn't return nanoseconds
	stat.Ctim.Sec = int64(statLE.Ctim)
	stat.Ctim.Nsec = 0 //zos doesn't return nanoseconds
	stat.Blksize = int64(statLE.Blksize)
	stat.Blocks = statLE.Blocks
}

func svcCall(fnptr unsafe.Pointer, argv *unsafe.Pointer, dsa *uint64)
func svcLoad(name *byte) unsafe.Pointer
func svcUnload(name *byte, fnptr unsafe.Pointer) int64

func (d *Dirent) NameString() string {
	if d == nil {
		return ""
	}
	s := string(d.Name[:])
	idx := strings.IndexByte(s, 0)
	if idx == -1 {
		return s
	} else {
		return s[:idx]
	}
}

func DecodeData(dest []byte, sz int, val uint64) {
	for i := 0; i < sz; i++ {
		dest[sz-1-i] = byte((val >> (uint64(i * 8))) & 0xff)
	}
}

func EncodeData(data []byte) uint64 {
	var value uint64
	sz := len(data)
	for i := 0; i < sz; i++ {
		value |= uint64(data[i]) << uint64(((sz - i - 1) * 8))
	}
	return value
}

func (sa *SockaddrInet4) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Len = SizeofSockaddrInet4
	sa.raw.Family = AF_INET
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return unsafe.Pointer(&sa.raw), _Socklen(sa.raw.Len), nil
}

func (sa *SockaddrInet6) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Len = SizeofSockaddrInet6
	sa.raw.Family = AF_INET6
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	sa.raw.Scope_id = sa.ZoneId
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return unsafe.Pointer(&sa.raw), _Socklen(sa.raw.Len), nil
}

func (sa *SockaddrUnix) sockaddr() (unsafe.Pointer, _Socklen, error) {
	name := sa.Name
	n := len(name)
	if n >= len(sa.raw.Path) || n == 0 {
		return nil, 0, EINVAL
	}
	sa.raw.Len = byte(3 + n) // 2 for Family, Len; 1 for NUL
	sa.raw.Family = AF_UNIX
	for i := 0; i < n; i++ {
		sa.raw.Path[i] = int8(name[i])
	}
	return unsafe.Pointer(&sa.raw), _Socklen(sa.raw.Len), nil
}

func anyToSockaddr(_ int, rsa *RawSockaddrAny) (Sockaddr, error) {
	// TODO(neeilan): Implement use of first param (fd)
	switch rsa.Addr.Family {
	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa))
		sa := new(SockaddrUnix)
		// For z/OS, only replace NUL with @ when the
		// length is not zero.
		if pp.Len != 0 && pp.Path[0] == 0 {
			// "Abstract" Unix domain socket.
			// Rewrite leading NUL as @ for textual display.
			// (This is the standard convention.)
			// Not friendly to overwrite in place,
			// but the callers below don't care.
			pp.Path[0] = '@'
		}

		// Assume path ends at NUL.
		//
		// For z/OS, the length of the name is a field
		// in the structure. To be on the safe side, we
		// will still scan the name for a NUL but only
		// to the length provided in the structure.
		//
		// This is not technically the Linux semantics for
		// abstract Unix domain sockets--they are supposed
		// to be uninterpreted fixed-size binary blobs--but
		// everyone uses this convention.
		n := 0
		for n < int(pp.Len) && pp.Path[n] != 0 {
			n++
		}
		sa.Name = string(unsafe.Slice((*byte)(unsafe.Pointer(&pp.Path[0])), n))
		return sa, nil

	case AF_INET:
		pp := (*RawSockaddrInet4)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet4)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i]
		}
		return sa, nil

	case AF_INET6:
		pp := (*RawSockaddrInet6)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet6)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		sa.ZoneId = pp.Scope_id
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i]
		}
		return sa, nil
	}
	return nil, EAFNOSUPPORT
}

func Accept(fd int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept(fd, &rsa, &len)
	if err != nil {
		return
	}
	// TODO(neeilan): Remove 0 in call
	sa, err = anyToSockaddr(0, &rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

func Accept4(fd int, flags int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept4(fd, &rsa, &len, flags)
	if err != nil {
		return
	}
	if len > SizeofSockaddrAny {
		panic("RawSockaddrAny too small")
	}
	// TODO(neeilan): Remove 0 in call
	sa, err = anyToSockaddr(0, &rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

func Ctermid() (tty string, err error) {
	var termdev [1025]byte
	runtime.EnterSyscall()
	r0, err2, err1 := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS___CTERMID_A<<4, uintptr(unsafe.Pointer(&termdev[0])))
	runtime.ExitSyscall()
	if r0 == 0 {
		return "", fmt.Errorf("%s (errno2=0x%x)\n", err1.Error(), err2)
	}
	s := string(termdev[:])
	idx := strings.Index(s, string(rune(0)))
	if idx == -1 {
		tty = s
	} else {
		tty = s[:idx]
	}
	return
}

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint64(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = int32(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = int32(length)
}

//sys   fcntl(fd int, cmd int, arg int) (val int, err error)
//sys   Flistxattr(fd int, dest []byte) (sz int, err error) = SYS___FLISTXATTR_A
//sys   Fremovexattr(fd int, attr string) (err error) = SYS___FREMOVEXATTR_A
//sys	read(fd int, p []byte) (n int, err error)
//sys	write(fd int, p []byte) (n int, err error)

//sys   Fgetxattr(fd int, attr string, dest []byte) (sz int, err error) = SYS___FGETXATTR_A
//sys   Fsetxattr(fd int, attr string, data []byte, flag int) (err error) = SYS___FSETXATTR_A

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, err error) = SYS___ACCEPT_A
//sys	accept4(s int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (fd int, err error) = SYS___ACCEPT4_A
//sys	bind(s int, addr unsafe.Pointer, addrlen _Socklen) (err error) = SYS___BIND_A
//sys	connect(s int, addr unsafe.Pointer, addrlen _Socklen) (err error) = SYS___CONNECT_A
//sysnb	getgroups(n int, list *_Gid_t) (nn int, err error)
//sysnb	setgroups(n int, list *_Gid_t) (err error)
//sys	getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *_Socklen) (err error)
//sys	setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) (err error)
//sysnb	socket(domain int, typ int, proto int) (fd int, err error)
//sysnb	socketpair(domain int, typ int, proto int, fd *[2]int32) (err error)
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error) = SYS___GETPEERNAME_A
//sysnb	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error) = SYS___GETSOCKNAME_A
//sys   Removexattr(path string, attr string) (err error) = SYS___REMOVEXATTR_A
//sys	recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, err error) = SYS___RECVFROM_A
//sys	sendto(s int, buf []byte, flags int, to unsafe.Pointer, addrlen _Socklen) (err error) = SYS___SENDTO_A
//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, err error) = SYS___RECVMSG_A
//sys	sendmsg(s int, msg *Msghdr, flags int) (n int, err error) = SYS___SENDMSG_A
//sys   mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error) = SYS_MMAP
//sys   munmap(addr uintptr, length uintptr) (err error) = SYS_MUNMAP
//sys   ioctl(fd int, req int, arg uintptr) (err error) = SYS_IOCTL
//sys   ioctlPtr(fd int, req int, arg unsafe.Pointer) (err error) = SYS_IOCTL
//sys	shmat(id int, addr uintptr, flag int) (ret uintptr, err error) = SYS_SHMAT
//sys	shmctl(id int, cmd int, buf *SysvShmDesc) (result int, err error) = SYS_SHMCTL64
//sys	shmdt(addr uintptr) (err error) = SYS_SHMDT
//sys	shmget(key int, size int, flag int) (id int, err error) = SYS_SHMGET

//sys   Access(path string, mode uint32) (err error) = SYS___ACCESS_A
//sys   Chdir(path string) (err error) = SYS___CHDIR_A
//sys	Chown(path string, uid int, gid int) (err error) = SYS___CHOWN_A
//sys	Chmod(path string, mode uint32) (err error) = SYS___CHMOD_A
//sys   Creat(path string, mode uint32) (fd int, err error) = SYS___CREAT_A
//sys	Dup(oldfd int) (fd int, err error)
//sys	Dup2(oldfd int, newfd int) (err error)
//sys	Dup3(oldfd int, newfd int, flags int) (err error) = SYS_DUP3
//sys	Dirfd(dirp uintptr) (fd int, err error) = SYS_DIRFD
//sys	EpollCreate(size int) (fd int, err error) = SYS_EPOLL_CREATE
//sys	EpollCreate1(flags int) (fd int, err error) = SYS_EPOLL_CREATE1
//sys	EpollCtl(epfd int, op int, fd int, event *EpollEvent) (err error) = SYS_EPOLL_CTL
//sys	EpollPwait(epfd int, events []EpollEvent, msec int, sigmask *int) (n int, err error) = SYS_EPOLL_PWAIT
//sys	EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error) = SYS_EPOLL_WAIT
//sys	Errno2() (er2 int) = SYS___ERRNO2
//sys	Eventfd(initval uint, flags int) (fd int, err error) = SYS_EVENTFD
//sys	Exit(code int)
//sys	Faccessat(dirfd int, path string, mode uint32, flags int) (err error) = SYS___FACCESSAT_A

func Faccessat2(dirfd int, path string, mode uint32, flags int) (err error) {
	return Faccessat(dirfd, path, mode, flags)
}

//sys	Fchdir(fd int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchmodat(dirfd int, path string, mode uint32, flags int) (err error) = SYS___FCHMODAT_A
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fchownat(fd int, path string, uid int, gid int, flags int) (err error) = SYS___FCHOWNAT_A
//sys	FcntlInt(fd uintptr, cmd int, arg int) (retval int, err error) = SYS_FCNTL
//sys	Fdatasync(fd int) (err error) = SYS_FDATASYNC
//sys	fstat(fd int, stat *Stat_LE_t) (err error)
//sys	fstatat(dirfd int, path string, stat *Stat_LE_t, flags int) (err error) = SYS___FSTATAT_A

func Fstat(fd int, stat *Stat_t) (err error) {
	var statLE Stat_LE_t
	err = fstat(fd, &statLE)
	copyStat(stat, &statLE)
	return
}

func Fstatat(dirfd int, path string, stat *Stat_t, flags int) (err error) {
	var statLE Stat_LE_t
	err = fstatat(dirfd, path, &statLE, flags)
	copyStat(stat, &statLE)
	return
}

func impl_Getxattr(path string, attr string, dest []byte) (sz int, err error) {
	var _p0 *byte
	_p0, err = BytePtrFromString(path)
	if err != nil {
		return
	}
	var _p1 *byte
	_p1, err = BytePtrFromString(attr)
	if err != nil {
		return
	}
	var _p2 unsafe.Pointer
	if len(dest) > 0 {
		_p2 = unsafe.Pointer(&dest[0])
	} else {
		_p2 = unsafe.Pointer(&_zero)
	}
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___GETXATTR_A<<4, uintptr(unsafe.Pointer(_p0)), uintptr(unsafe.Pointer(_p1)), uintptr(_p2), uintptr(len(dest)))
	sz = int(r0)
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//go:nosplit
func get_GetxattrAddr() *(func(path string, attr string, dest []byte) (sz int, err error))

var Getxattr = enter_Getxattr

func enter_Getxattr(path string, attr string, dest []byte) (sz int, err error) {
	funcref := get_GetxattrAddr()
	if validGetxattr() {
		*funcref = impl_Getxattr
	} else {
		*funcref = error_Getxattr
	}
	return (*funcref)(path, attr, dest)
}

func error_Getxattr(path string, attr string, dest []byte) (sz int, err error) {
	return -1, ENOSYS
}

func validGetxattr() bool {
	if funcptrtest(GetZosLibVec()+SYS___GETXATTR_A<<4, "") == 0 {
		if name, err := getLeFuncName(GetZosLibVec() + SYS___GETXATTR_A<<4); err == nil {
			return name == "__getxattr_a"
		}
	}
	return false
}

//sys   Lgetxattr(link string, attr string, dest []byte) (sz int, err error) = SYS___LGETXATTR_A
//sys   Lsetxattr(path string, attr string, data []byte, flags int) (err error) = SYS___LSETXATTR_A

func impl_Setxattr(path string, attr string, data []byte, flags int) (err error) {
	var _p0 *byte
	_p0, err = BytePtrFromString(path)
	if err != nil {
		return
	}
	var _p1 *byte
	_p1, err = BytePtrFromString(attr)
	if err != nil {
		return
	}
	var _p2 unsafe.Pointer
	if len(data) > 0 {
		_p2 = unsafe.Pointer(&data[0])
	} else {
		_p2 = unsafe.Pointer(&_zero)
	}
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___SETXATTR_A<<4, uintptr(unsafe.Pointer(_p0)), uintptr(unsafe.Pointer(_p1)), uintptr(_p2), uintptr(len(data)), uintptr(flags))
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//go:nosplit
func get_SetxattrAddr() *(func(path string, attr string, data []byte, flags int) (err error))

var Setxattr = enter_Setxattr

func enter_Setxattr(path string, attr string, data []byte, flags int) (err error) {
	funcref := get_SetxattrAddr()
	if validSetxattr() {
		*funcref = impl_Setxattr
	} else {
		*funcref = error_Setxattr
	}
	return (*funcref)(path, attr, data, flags)
}

func error_Setxattr(path string, attr string, data []byte, flags int) (err error) {
	return ENOSYS
}

func validSetxattr() bool {
	if funcptrtest(GetZosLibVec()+SYS___SETXATTR_A<<4, "") == 0 {
		if name, err := getLeFuncName(GetZosLibVec() + SYS___SETXATTR_A<<4); err == nil {
			return name == "__setxattr_a"
		}
	}
	return false
}

//sys	Fstatfs(fd int, buf *Statfs_t) (err error) = SYS_FSTATFS
//sys	Fstatvfs(fd int, stat *Statvfs_t) (err error) = SYS_FSTATVFS
//sys	Fsync(fd int) (err error)
//sys	Futimes(fd int, tv []Timeval) (err error) = SYS_FUTIMES
//sys	Futimesat(dirfd int, path string, tv []Timeval) (err error) = SYS___FUTIMESAT_A
//sys	Ftruncate(fd int, length int64) (err error)
//sys	Getrandom(buf []byte, flags int) (n int, err error) = SYS_GETRANDOM
//sys	InotifyInit() (fd int, err error) = SYS_INOTIFY_INIT
//sys	InotifyInit1(flags int) (fd int, err error) = SYS_INOTIFY_INIT1
//sys	InotifyAddWatch(fd int, pathname string, mask uint32) (watchdesc int, err error) = SYS___INOTIFY_ADD_WATCH_A
//sys	InotifyRmWatch(fd int, watchdesc uint32) (success int, err error) = SYS_INOTIFY_RM_WATCH
//sys   Listxattr(path string, dest []byte) (sz int, err error) = SYS___LISTXATTR_A
//sys   Llistxattr(path string, dest []byte) (sz int, err error) = SYS___LLISTXATTR_A
//sys   Lremovexattr(path string, attr string) (err error) = SYS___LREMOVEXATTR_A
//sys	Lutimes(path string, tv []Timeval) (err error) = SYS___LUTIMES_A
//sys   Mprotect(b []byte, prot int) (err error) = SYS_MPROTECT
//sys   Msync(b []byte, flags int) (err error) = SYS_MSYNC
//sys   Console2(cmsg *ConsMsg2, modstr *byte, concmd *uint32) (err error) = SYS___CONSOLE2

// Pipe2 begin

//go:nosplit
func getPipe2Addr() *(func([]int, int) error)

var Pipe2 = pipe2Enter

func pipe2Enter(p []int, flags int) (err error) {
	if funcptrtest(GetZosLibVec()+SYS_PIPE2<<4, "") == 0 {
		*getPipe2Addr() = pipe2Impl
	} else {
		*getPipe2Addr() = pipe2Error
	}
	return (*getPipe2Addr())(p, flags)
}

func pipe2Impl(p []int, flags int) (err error) {
	var pp [2]_C_int
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_PIPE2<<4, uintptr(unsafe.Pointer(&pp[0])), uintptr(flags))
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	} else {
		p[0] = int(pp[0])
		p[1] = int(pp[1])
	}
	return
}
func pipe2Error(p []int, flags int) (err error) {
	return fmt.Errorf("Pipe2 is not available on this system")
}

// Pipe2 end

//sys   Poll(fds []PollFd, timeout int) (n int, err error) = SYS_POLL

func Readdir(dir uintptr) (dirent *Dirent, err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___READDIR_A<<4, uintptr(dir))
	runtime.ExitSyscall()
	dirent = (*Dirent)(unsafe.Pointer(r0))
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//sys	Readdir_r(dirp uintptr, entry *direntLE, result **direntLE) (err error) = SYS___READDIR_R_A
//sys	Statfs(path string, buf *Statfs_t) (err error) = SYS___STATFS_A
//sys	Syncfs(fd int) (err error) = SYS_SYNCFS
//sys   Times(tms *Tms) (ticks uintptr, err error) = SYS_TIMES
//sys   W_Getmntent(buff *byte, size int) (lastsys int, err error) = SYS_W_GETMNTENT
//sys   W_Getmntent_A(buff *byte, size int) (lastsys int, err error) = SYS___W_GETMNTENT_A

//sys   mount_LE(path string, filesystem string, fstype string, mtm uint32, parmlen int32, parm string) (err error) = SYS___MOUNT_A
//sys   unmount_LE(filesystem string, mtm int) (err error) = SYS___UMOUNT_A
//sys   Chroot(path string) (err error) = SYS___CHROOT_A
//sys   Select(nmsgsfds int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (ret int, err error) = SYS_SELECT
//sysnb Uname(buf *Utsname) (err error) = SYS_____OSNAME_A
//sys   Unshare(flags int) (err error) = SYS_UNSHARE

func Ptsname(fd int) (name string, err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS___PTSNAME_A<<4, uintptr(fd))
	runtime.ExitSyscall()
	if r0 == 0 {
		err = errnoErr2(e1, e2)
	} else {
		name = u2s(unsafe.Pointer(r0))
	}
	return
}

func u2s(cstr unsafe.Pointer) string {
	str := (*[1024]uint8)(cstr)
	i := 0
	for str[i] != 0 {
		i++
	}
	return string(str[:i])
}

func Close(fd int) (err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_CLOSE<<4, uintptr(fd))
	runtime.ExitSyscall()
	for i := 0; e1 == EAGAIN && i < 10; i++ {
		runtime.EnterSyscall()
		CallLeFuncWithErr(GetZosLibVec()+SYS_USLEEP<<4, uintptr(10))
		runtime.ExitSyscall()
		runtime.EnterSyscall()
		r0, e2, e1 = CallLeFuncWithErr(GetZosLibVec()+SYS_CLOSE<<4, uintptr(fd))
		runtime.ExitSyscall()
	}
	if r0 != 0 {
		err = errnoErr2(e1, e2)
	}
	return
}

// Dummy function: there are no semantics for Madvise on z/OS
func Madvise(b []byte, advice int) (err error) {
	return
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return mapper.Munmap(b)
}

//sys   Gethostname(buf []byte) (err error) = SYS___GETHOSTNAME_A
//sysnb	Getgid() (gid int)
//sysnb	Getpid() (pid int)
//sysnb	Getpgid(pid int) (pgid int, err error) = SYS_GETPGID

func Getpgrp() (pid int) {
	pid, _ = Getpgid(0)
	return
}

//sysnb	Getppid() (pid int)
//sys	Getpriority(which int, who int) (prio int, err error)
//sysnb	Getrlimit(resource int, rlim *Rlimit) (err error) = SYS_GETRLIMIT

//sysnb getrusage(who int, rusage *rusage_zos) (err error) = SYS_GETRUSAGE

func Getrusage(who int, rusage *Rusage) (err error) {
	var ruz rusage_zos
	err = getrusage(who, &ruz)
	//Only the first two fields of Rusage are set
	rusage.Utime.Sec = ruz.Utime.Sec
	rusage.Utime.Usec = int64(ruz.Utime.Usec)
	rusage.Stime.Sec = ruz.Stime.Sec
	rusage.Stime.Usec = int64(ruz.Stime.Usec)
	return
}

//sys	Getegid() (egid int) = SYS_GETEGID
//sys	Geteuid() (euid int) = SYS_GETEUID
//sysnb Getsid(pid int) (sid int, err error) = SYS_GETSID
//sysnb	Getuid() (uid int)
//sysnb	Kill(pid int, sig Signal) (err error)
//sys	Lchown(path string, uid int, gid int) (err error) = SYS___LCHOWN_A
//sys	Link(path string, link string) (err error) = SYS___LINK_A
//sys	Linkat(oldDirFd int, oldPath string, newDirFd int, newPath string, flags int) (err error) = SYS___LINKAT_A
//sys	Listen(s int, n int) (err error)
//sys	lstat(path string, stat *Stat_LE_t) (err error) = SYS___LSTAT_A

func Lstat(path string, stat *Stat_t) (err error) {
	var statLE Stat_LE_t
	err = lstat(path, &statLE)
	copyStat(stat, &statLE)
	return
}

// for checking symlinks begins with $VERSION/ $SYSNAME/ $SYSSYMR/ $SYSSYMA/
func isSpecialPath(path []byte) (v bool) {
	var special = [4][8]byte{
		{'V', 'E', 'R', 'S', 'I', 'O', 'N', '/'},
		{'S', 'Y', 'S', 'N', 'A', 'M', 'E', '/'},
		{'S', 'Y', 'S', 'S', 'Y', 'M', 'R', '/'},
		{'S', 'Y', 'S', 'S', 'Y', 'M', 'A', '/'}}

	var i, j int
	for i = 0; i < len(special); i++ {
		for j = 0; j < len(special[i]); j++ {
			if path[j] != special[i][j] {
				break
			}
		}
		if j == len(special[i]) {
			return true
		}
	}
	return false
}

func realpath(srcpath string, abspath []byte) (pathlen int, errno int) {
	var source [1024]byte
	copy(source[:], srcpath)
	source[len(srcpath)] = 0
	ret := runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___REALPATH_A<<4, //__realpath_a()
		[]uintptr{uintptr(unsafe.Pointer(&source[0])),
			uintptr(unsafe.Pointer(&abspath[0]))})
	if ret != 0 {
		index := bytes.IndexByte(abspath[:], byte(0))
		if index != -1 {
			return index, 0
		}
	} else {
		errptr := (*int)(unsafe.Pointer(runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___ERRNO<<4, []uintptr{}))) //__errno()
		return 0, *errptr
	}
	return 0, 245 // EBADDATA   245
}

func Readlink(path string, buf []byte) (n int, err error) {
	var _p0 *byte
	_p0, err = BytePtrFromString(path)
	if err != nil {
		return
	}
	var _p1 unsafe.Pointer
	if len(buf) > 0 {
		_p1 = unsafe.Pointer(&buf[0])
	} else {
		_p1 = unsafe.Pointer(&_zero)
	}
	n = int(runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___READLINK_A<<4,
		[]uintptr{uintptr(unsafe.Pointer(_p0)), uintptr(_p1), uintptr(len(buf))}))
	runtime.KeepAlive(unsafe.Pointer(_p0))
	if n == -1 {
		value := *(*int32)(unsafe.Pointer(runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___ERRNO<<4, []uintptr{})))
		err = errnoErr(Errno(value))
	} else {
		if buf[0] == '$' {
			if isSpecialPath(buf[1:9]) {
				cnt, err1 := realpath(path, buf)
				if err1 == 0 {
					n = cnt
				}
			}
		}
	}
	return
}

func impl_Readlinkat(dirfd int, path string, buf []byte) (n int, err error) {
	var _p0 *byte
	_p0, err = BytePtrFromString(path)
	if err != nil {
		return
	}
	var _p1 unsafe.Pointer
	if len(buf) > 0 {
		_p1 = unsafe.Pointer(&buf[0])
	} else {
		_p1 = unsafe.Pointer(&_zero)
	}
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___READLINKAT_A<<4, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(_p1), uintptr(len(buf)))
	runtime.ExitSyscall()
	n = int(r0)
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
		return n, err
	} else {
		if buf[0] == '$' {
			if isSpecialPath(buf[1:9]) {
				cnt, err1 := realpath(path, buf)
				if err1 == 0 {
					n = cnt
				}
			}
		}
	}
	return
}

//go:nosplit
func get_ReadlinkatAddr() *(func(dirfd int, path string, buf []byte) (n int, err error))

var Readlinkat = enter_Readlinkat

func enter_Readlinkat(dirfd int, path string, buf []byte) (n int, err error) {
	funcref := get_ReadlinkatAddr()
	if funcptrtest(GetZosLibVec()+SYS___READLINKAT_A<<4, "") == 0 {
		*funcref = impl_Readlinkat
	} else {
		*funcref = error_Readlinkat
	}
	return (*funcref)(dirfd, path, buf)
}

func error_Readlinkat(dirfd int, path string, buf []byte) (n int, err error) {
	n = -1
	err = ENOSYS
	return
}

//sys	Mkdir(path string, mode uint32) (err error) = SYS___MKDIR_A
//sys	Mkdirat(dirfd int, path string, mode uint32) (err error) = SYS___MKDIRAT_A
//sys   Mkfifo(path string, mode uint32) (err error) = SYS___MKFIFO_A
//sys	Mknod(path string, mode uint32, dev int) (err error) = SYS___MKNOD_A
//sys	Mknodat(dirfd int, path string, mode uint32, dev int) (err error) = SYS___MKNODAT_A
//sys	PivotRoot(newroot string, oldroot string) (err error) = SYS___PIVOT_ROOT_A
//sys	Pread(fd int, p []byte, offset int64) (n int, err error)
//sys	Pwrite(fd int, p []byte, offset int64) (n int, err error)
//sys	Prctl(option int, arg2 uintptr, arg3 uintptr, arg4 uintptr, arg5 uintptr) (err error) = SYS___PRCTL_A
//sysnb	Prlimit(pid int, resource int, newlimit *Rlimit, old *Rlimit) (err error) = SYS_PRLIMIT
//sys	Rename(from string, to string) (err error) = SYS___RENAME_A
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error) = SYS___RENAMEAT_A
//sys	Renameat2(olddirfd int, oldpath string, newdirfd int, newpath string, flags uint) (err error) = SYS___RENAMEAT2_A
//sys	Rmdir(path string) (err error) = SYS___RMDIR_A
//sys   Seek(fd int, offset int64, whence int) (off int64, err error) = SYS_LSEEK
//sys	Setegid(egid int) (err error) = SYS_SETEGID
//sys	Seteuid(euid int) (err error) = SYS_SETEUID
//sys	Sethostname(p []byte) (err error) = SYS___SETHOSTNAME_A
//sys   Setns(fd int, nstype int) (err error) = SYS_SETNS
//sys	Setpriority(which int, who int, prio int) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error) = SYS_SETPGID
//sysnb	Setrlimit(resource int, lim *Rlimit) (err error)
//sysnb	Setregid(rgid int, egid int) (err error) = SYS_SETREGID
//sysnb	Setreuid(ruid int, euid int) (err error) = SYS_SETREUID
//sysnb	Setsid() (pid int, err error) = SYS_SETSID
//sys	Setuid(uid int) (err error) = SYS_SETUID
//sys	Setgid(uid int) (err error) = SYS_SETGID
//sys	Shutdown(fd int, how int) (err error)
//sys	stat(path string, statLE *Stat_LE_t) (err error) = SYS___STAT_A

func Stat(path string, sta *Stat_t) (err error) {
	var statLE Stat_LE_t
	err = stat(path, &statLE)
	copyStat(sta, &statLE)
	return
}

//sys	Symlink(path string, link string) (err error) = SYS___SYMLINK_A
//sys	Symlinkat(oldPath string, dirfd int, newPath string) (err error) = SYS___SYMLINKAT_A
//sys	Sync() = SYS_SYNC
//sys	Truncate(path string, length int64) (err error) = SYS___TRUNCATE_A
//sys	Tcgetattr(fildes int, termptr *Termios) (err error) = SYS_TCGETATTR
//sys	Tcsetattr(fildes int, when int, termptr *Termios) (err error) = SYS_TCSETATTR
//sys	Umask(mask int) (oldmask int)
//sys	Unlink(path string) (err error) = SYS___UNLINK_A
//sys	Unlinkat(dirfd int, path string, flags int) (err error) = SYS___UNLINKAT_A
//sys	Utime(path string, utim *Utimbuf) (err error) = SYS___UTIME_A

//sys	open(path string, mode int, perm uint32) (fd int, err error) = SYS___OPEN_A

func Open(path string, mode int, perm uint32) (fd int, err error) {
	if mode&O_ACCMODE == 0 {
		mode |= O_RDONLY
	}
	return open(path, mode, perm)
}

//sys	openat(dirfd int, path string, flags int, mode uint32) (fd int, err error) = SYS___OPENAT_A

func Openat(dirfd int, path string, flags int, mode uint32) (fd int, err error) {
	if flags&O_ACCMODE == 0 {
		flags |= O_RDONLY
	}
	return openat(dirfd, path, flags, mode)
}

//sys	openat2(dirfd int, path string, open_how *OpenHow, size int) (fd int, err error) = SYS___OPENAT2_A

func Openat2(dirfd int, path string, how *OpenHow) (fd int, err error) {
	if how.Flags&O_ACCMODE == 0 {
		how.Flags |= O_RDONLY
	}
	return openat2(dirfd, path, how, SizeofOpenHow)
}

func ZosFdToPath(dirfd int) (path string, err error) {
	var buffer [1024]byte
	runtime.EnterSyscall()
	ret, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_W_IOCTL<<4, uintptr(dirfd), 17, 1024, uintptr(unsafe.Pointer(&buffer[0])))
	runtime.ExitSyscall()
	if ret == 0 {
		zb := bytes.IndexByte(buffer[:], 0)
		if zb == -1 {
			zb = len(buffer)
		}
		CallLeFuncWithErr(GetZosLibVec()+SYS___E2A_L<<4, uintptr(unsafe.Pointer(&buffer[0])), uintptr(zb))
		return string(buffer[:zb]), nil
	}
	return "", errnoErr2(e1, e2)
}

//sys	remove(path string) (err error)

func Remove(path string) error {
	return remove(path)
}

const ImplementsGetwd = true

func Getcwd(buf []byte) (n int, err error) {
	var p unsafe.Pointer
	if len(buf) > 0 {
		p = unsafe.Pointer(&buf[0])
	} else {
		p = unsafe.Pointer(&_zero)
	}
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS___GETCWD_A<<4, uintptr(p), uintptr(len(buf)))
	runtime.ExitSyscall()
	n = clen(buf) + 1
	if r0 == 0 {
		err = errnoErr2(e1, e2)
	}
	return
}

func Getwd() (wd string, err error) {
	var buf [PathMax]byte
	n, err := Getcwd(buf[0:])
	if err != nil {
		return "", err
	}
	// Getcwd returns the number of bytes written to buf, including the NUL.
	if n < 1 || n > len(buf) || buf[n-1] != 0 {
		return "", EINVAL
	}
	return string(buf[0 : n-1]), nil
}

func Getgroups() (gids []int, err error) {
	n, err := getgroups(0, nil)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}

	// Sanity check group count.  Max is 1<<16 on Linux.
	if n < 0 || n > 1<<20 {
		return nil, EINVAL
	}

	a := make([]_Gid_t, n)
	n, err = getgroups(n, &a[0])
	if err != nil {
		return nil, err
	}
	gids = make([]int, n)
	for i, v := range a[0:n] {
		gids[i] = int(v)
	}
	return
}

func Setgroups(gids []int) (err error) {
	if len(gids) == 0 {
		return setgroups(0, nil)
	}

	a := make([]_Gid_t, len(gids))
	for i, v := range gids {
		a[i] = _Gid_t(v)
	}
	return setgroups(len(a), &a[0])
}

func gettid() uint64

func Gettid() (tid int) {
	return int(gettid())
}

type WaitStatus uint32

// Wait status is 7 bits at bottom, either 0 (exited),
// 0x7F (stopped), or a signal number that caused an exit.
// The 0x80 bit is whether there was a core dump.
// An extra number (exit code, signal causing a stop)
// is in the high bits.  At least that's the idea.
// There are various irregularities.  For example, the
// "continued" status is 0xFFFF, distinguishing itself
// from stopped via the core dump bit.

const (
	mask    = 0x7F
	core    = 0x80
	exited  = 0x00
	stopped = 0x7F
	shift   = 8
)

func (w WaitStatus) Exited() bool { return w&mask == exited }

func (w WaitStatus) Signaled() bool { return w&mask != stopped && w&mask != exited }

func (w WaitStatus) Stopped() bool { return w&0xFF == stopped }

func (w WaitStatus) Continued() bool { return w == 0xFFFF }

func (w WaitStatus) CoreDump() bool { return w.Signaled() && w&core != 0 }

func (w WaitStatus) ExitStatus() int {
	if !w.Exited() {
		return -1
	}
	return int(w>>shift) & 0xFF
}

func (w WaitStatus) Signal() Signal {
	if !w.Signaled() {
		return -1
	}
	return Signal(w & mask)
}

func (w WaitStatus) StopSignal() Signal {
	if !w.Stopped() {
		return -1
	}
	return Signal(w>>shift) & 0xFF
}

func (w WaitStatus) TrapCause() int { return -1 }

//sys	waitid(idType int, id int, info *Siginfo, options int) (err error)

func Waitid(idType int, id int, info *Siginfo, options int, rusage *Rusage) (err error) {
	return waitid(idType, id, info, options)
}

//sys	waitpid(pid int, wstatus *_C_int, options int) (wpid int, err error)

func impl_Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_WAIT4<<4, uintptr(pid), uintptr(unsafe.Pointer(wstatus)), uintptr(options), uintptr(unsafe.Pointer(rusage)))
	runtime.ExitSyscall()
	wpid = int(r0)
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//go:nosplit
func get_Wait4Addr() *(func(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error))

var Wait4 = enter_Wait4

func enter_Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	funcref := get_Wait4Addr()
	if funcptrtest(GetZosLibVec()+SYS_WAIT4<<4, "") == 0 {
		*funcref = impl_Wait4
	} else {
		*funcref = legacyWait4
	}
	return (*funcref)(pid, wstatus, options, rusage)
}

func legacyWait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	// TODO(mundaym): z/OS doesn't have wait4. I don't think getrusage does what we want.
	// At the moment rusage will not be touched.
	var status _C_int
	wpid, err = waitpid(pid, &status, options)
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return
}

//sysnb	gettimeofday(tv *timeval_zos) (err error)

func Gettimeofday(tv *Timeval) (err error) {
	var tvz timeval_zos
	err = gettimeofday(&tvz)
	tv.Sec = tvz.Sec
	tv.Usec = int64(tvz.Usec)
	return
}

func Time(t *Time_t) (tt Time_t, err error) {
	var tv Timeval
	err = Gettimeofday(&tv)
	if err != nil {
		return 0, err
	}
	if t != nil {
		*t = Time_t(tv.Sec)
	}
	return Time_t(tv.Sec), nil
}

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: sec, Nsec: nsec}
}

func setTimeval(sec, usec int64) Timeval { //fix
	return Timeval{Sec: sec, Usec: usec}
}

//sysnb pipe(p *[2]_C_int) (err error)

func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err = pipe(&pp)
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return
}

//sys	utimes(path string, timeval *[2]Timeval) (err error) = SYS___UTIMES_A

func Utimes(path string, tv []Timeval) (err error) {
	if tv == nil {
		return utimes(path, nil)
	}
	if len(tv) != 2 {
		return EINVAL
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	utimensat(dirfd int, path string, ts *[2]Timespec, flags int) (err error) = SYS___UTIMENSAT_A

func validUtimensat() bool {
	if funcptrtest(GetZosLibVec()+SYS___UTIMENSAT_A<<4, "") == 0 {
		if name, err := getLeFuncName(GetZosLibVec() + SYS___UTIMENSAT_A<<4); err == nil {
			return name == "__utimensat_a"
		}
	}
	return false
}

// Begin UtimesNano

//go:nosplit
func get_UtimesNanoAddr() *(func(path string, ts []Timespec) (err error))

var UtimesNano = enter_UtimesNano

func enter_UtimesNano(path string, ts []Timespec) (err error) {
	funcref := get_UtimesNanoAddr()
	if validUtimensat() {
		*funcref = utimesNanoImpl
	} else {
		*funcref = legacyUtimesNano
	}
	return (*funcref)(path, ts)
}

func utimesNanoImpl(path string, ts []Timespec) (err error) {
	if ts == nil {
		return utimensat(AT_FDCWD, path, nil, 0)
	}
	if len(ts) != 2 {
		return EINVAL
	}
	return utimensat(AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
}

func legacyUtimesNano(path string, ts []Timespec) (err error) {
	if len(ts) != 2 {
		return EINVAL
	}
	// Not as efficient as it could be because Timespec and
	// Timeval have different types in the different OSes
	tv := [2]Timeval{
		NsecToTimeval(TimespecToNsec(ts[0])),
		NsecToTimeval(TimespecToNsec(ts[1])),
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

// End UtimesNano

// Begin UtimesNanoAt

//go:nosplit
func get_UtimesNanoAtAddr() *(func(dirfd int, path string, ts []Timespec, flags int) (err error))

var UtimesNanoAt = enter_UtimesNanoAt

func enter_UtimesNanoAt(dirfd int, path string, ts []Timespec, flags int) (err error) {
	funcref := get_UtimesNanoAtAddr()
	if validUtimensat() {
		*funcref = utimesNanoAtImpl
	} else {
		*funcref = legacyUtimesNanoAt
	}
	return (*funcref)(dirfd, path, ts, flags)
}

func utimesNanoAtImpl(dirfd int, path string, ts []Timespec, flags int) (err error) {
	if ts == nil {
		return utimensat(dirfd, path, nil, flags)
	}
	if len(ts) != 2 {
		return EINVAL
	}
	return utimensat(dirfd, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), flags)
}

func legacyUtimesNanoAt(dirfd int, path string, ts []Timespec, flags int) (err error) {
	if path[0] != '/' {
		dirPath, err := ZosFdToPath(dirfd)
		if err != nil {
			return err
		}
		path = dirPath + "/" + path
	}
	if flags == AT_SYMLINK_NOFOLLOW {
		if len(ts) != 2 {
			return EINVAL
		}

		if ts[0].Nsec >= 5e8 {
			ts[0].Sec++
		}
		ts[0].Nsec = 0
		if ts[1].Nsec >= 5e8 {
			ts[1].Sec++
		}
		ts[1].Nsec = 0

		// Not as efficient as it could be because Timespec and
		// Timeval have different types in the different OSes
		tv := []Timeval{
			NsecToTimeval(TimespecToNsec(ts[0])),
			NsecToTimeval(TimespecToNsec(ts[1])),
		}
		return Lutimes(path, tv)
	}
	return UtimesNano(path, ts)
}

// End UtimesNanoAt

func Getsockname(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if err = getsockname(fd, &rsa, &len); err != nil {
		return
	}
	// TODO(neeilan) : Remove this 0 ( added to get sys/unix compiling on z/OS )
	return anyToSockaddr(0, &rsa)
}

const (
	// identifier constants
	nwmHeaderIdentifier    = 0xd5e6d4c8
	nwmFilterIdentifier    = 0xd5e6d4c6
	nwmTCPConnIdentifier   = 0xd5e6d4c3
	nwmRecHeaderIdentifier = 0xd5e6d4d9
	nwmIPStatsIdentifier   = 0xd5e6d4c9d7e2e340
	nwmIPGStatsIdentifier  = 0xd5e6d4c9d7c7e2e3
	nwmTCPStatsIdentifier  = 0xd5e6d4e3c3d7e2e3
	nwmUDPStatsIdentifier  = 0xd5e6d4e4c4d7e2e3
	nwmICMPGStatsEntry     = 0xd5e6d4c9c3d4d7c7
	nwmICMPTStatsEntry     = 0xd5e6d4c9c3d4d7e3

	// nwmHeader constants
	nwmVersion1   = 1
	nwmVersion2   = 2
	nwmCurrentVer = 2

	nwmTCPConnType     = 1
	nwmGlobalStatsType = 14

	// nwmFilter constants
	nwmFilterLclAddrMask = 0x20000000 // Local address
	nwmFilterSrcAddrMask = 0x20000000 // Source address
	nwmFilterLclPortMask = 0x10000000 // Local port
	nwmFilterSrcPortMask = 0x10000000 // Source port

	// nwmConnEntry constants
	nwmTCPStateClosed   = 1
	nwmTCPStateListen   = 2
	nwmTCPStateSynSent  = 3
	nwmTCPStateSynRcvd  = 4
	nwmTCPStateEstab    = 5
	nwmTCPStateFinWait1 = 6
	nwmTCPStateFinWait2 = 7
	nwmTCPStateClosWait = 8
	nwmTCPStateLastAck  = 9
	nwmTCPStateClosing  = 10
	nwmTCPStateTimeWait = 11
	nwmTCPStateDeletTCB = 12

	// Existing constants on linux
	BPF_TCP_CLOSE        = 1
	BPF_TCP_LISTEN       = 2
	BPF_TCP_SYN_SENT     = 3
	BPF_TCP_SYN_RECV     = 4
	BPF_TCP_ESTABLISHED  = 5
	BPF_TCP_FIN_WAIT1    = 6
	BPF_TCP_FIN_WAIT2    = 7
	BPF_TCP_CLOSE_WAIT   = 8
	BPF_TCP_LAST_ACK     = 9
	BPF_TCP_CLOSING      = 10
	BPF_TCP_TIME_WAIT    = 11
	BPF_TCP_NEW_SYN_RECV = -1
	BPF_TCP_MAX_STATES   = -2
)

type nwmTriplet struct {
	offset uint32
	length uint32
	number uint32
}

type nwmQuadruplet struct {
	offset uint32
	length uint32
	number uint32
	match  uint32
}

type nwmHeader struct {
	ident       uint32
	length      uint32
	version     uint16
	nwmType     uint16
	bytesNeeded uint32
	options     uint32
	_           [16]byte
	inputDesc   nwmTriplet
	outputDesc  nwmQuadruplet
}

type nwmFilter struct {
	ident         uint32
	flags         uint32
	resourceName  [8]byte
	resourceId    uint32
	listenerId    uint32
	local         [28]byte // union of sockaddr4 and sockaddr6
	remote        [28]byte // union of sockaddr4 and sockaddr6
	_             uint16
	_             uint16
	asid          uint16
	_             [2]byte
	tnLuName      [8]byte
	tnMonGrp      uint32
	tnAppl        [8]byte
	applData      [40]byte
	nInterface    [16]byte
	dVipa         [16]byte
	dVipaPfx      uint16
	dVipaPort     uint16
	dVipaFamily   byte
	_             [3]byte
	destXCF       [16]byte
	destXCFPfx    uint16
	destXCFFamily byte
	_             [1]byte
	targIP        [16]byte
	targIPPfx     uint16
	targIPFamily  byte
	_             [1]byte
	_             [20]byte
}

type nwmRecHeader struct {
	ident  uint32
	length uint32
	number byte
	_      [3]byte
}

type nwmTCPStatsEntry struct {
	ident             uint64
	currEstab         uint32
	activeOpened      uint32
	passiveOpened     uint32
	connClosed        uint32
	estabResets       uint32
	attemptFails      uint32
	passiveDrops      uint32
	timeWaitReused    uint32
	inSegs            uint64
	predictAck        uint32
	predictData       uint32
	inDupAck          uint32
	inBadSum          uint32
	inBadLen          uint32
	inShort           uint32
	inDiscOldTime     uint32
	inAllBeforeWin    uint32
	inSomeBeforeWin   uint32
	inAllAfterWin     uint32
	inSomeAfterWin    uint32
	inOutOfOrder      uint32
	inAfterClose      uint32
	inWinProbes       uint32
	inWinUpdates      uint32
	outWinUpdates     uint32
	outSegs           uint64
	outDelayAcks      uint32
	outRsts           uint32
	retransSegs       uint32
	retransTimeouts   uint32
	retransDrops      uint32
	pmtuRetrans       uint32
	pmtuErrors        uint32
	outWinProbes      uint32
	probeDrops        uint32
	keepAliveProbes   uint32
	keepAliveDrops    uint32
	finwait2Drops     uint32
	acceptCount       uint64
	inBulkQSegs       uint64
	inDiscards        uint64
	connFloods        uint32
	connStalls        uint32
	cfgEphemDef       uint16
	ephemInUse        uint16
	ephemHiWater      uint16
	flags             byte
	_                 [1]byte
	ephemExhaust      uint32
	smcRCurrEstabLnks uint32
	smcRLnkActTimeOut uint32
	smcRActLnkOpened  uint32
	smcRPasLnkOpened  uint32
	smcRLnksClosed    uint32
	smcRCurrEstab     uint32
	smcRActiveOpened  uint32
	smcRPassiveOpened uint32
	smcRConnClosed    uint32
	smcRInSegs        uint64
	smcROutSegs       uint64
	smcRInRsts        uint32
	smcROutRsts       uint32
	smcDCurrEstabLnks uint32
	smcDActLnkOpened  uint32
	smcDPasLnkOpened  uint32
	smcDLnksClosed    uint32
	smcDCurrEstab     uint32
	smcDActiveOpened  uint32
	smcDPassiveOpened uint32
	smcDConnClosed    uint32
	smcDInSegs        uint64
	smcDOutSegs       uint64
	smcDInRsts        uint32
	smcDOutRsts       uint32
}

type nwmConnEntry struct {
	ident             uint32
	local             [28]byte // union of sockaddr4 and sockaddr6
	remote            [28]byte // union of sockaddr4 and sockaddr6
	startTime         [8]byte  // uint64, changed to prevent padding from being inserted
	lastActivity      [8]byte  // uint64
	bytesIn           [8]byte  // uint64
	bytesOut          [8]byte  // uint64
	inSegs            [8]byte  // uint64
	outSegs           [8]byte  // uint64
	state             uint16
	activeOpen        byte
	flag01            byte
	outBuffered       uint32
	inBuffered        uint32
	maxSndWnd         uint32
	reXmtCount        uint32
	congestionWnd     uint32
	ssThresh          uint32
	roundTripTime     uint32
	roundTripVar      uint32
	sendMSS           uint32
	sndWnd            uint32
	rcvBufSize        uint32
	sndBufSize        uint32
	outOfOrderCount   uint32
	lcl0WindowCount   uint32
	rmt0WindowCount   uint32
	dupacks           uint32
	flag02            byte
	sockOpt6Cont      byte
	asid              uint16
	resourceName      [8]byte
	resourceId        uint32
	subtask           uint32
	sockOpt           byte
	sockOpt6          byte
	clusterConnFlag   byte
	proto             byte
	targetAppl        [8]byte
	luName            [8]byte
	clientUserId      [8]byte
	logMode           [8]byte
	timeStamp         uint32
	timeStampAge      uint32
	serverResourceId  uint32
	intfName          [16]byte
	ttlsStatPol       byte
	ttlsStatConn      byte
	ttlsSSLProt       uint16
	ttlsNegCiph       [2]byte
	ttlsSecType       byte
	ttlsFIPS140Mode   byte
	ttlsUserID        [8]byte
	applData          [40]byte
	inOldestTime      [8]byte // uint64
	outOldestTime     [8]byte // uint64
	tcpTrustedPartner byte
	_                 [3]byte
	bulkDataIntfName  [16]byte
	ttlsNegCiph4      [4]byte
	smcReason         uint32
	lclSMCLinkId      uint32
	rmtSMCLinkId      uint32
	smcStatus         byte
	smcFlags          byte
	_                 [2]byte
	rcvWnd            uint32
	lclSMCBufSz       uint32
	rmtSMCBufSz       uint32
	ttlsSessID        [32]byte
	ttlsSessIDLen     int16
	_                 [1]byte
	smcDStatus        byte
	smcDReason        uint32
}

var svcNameTable [][]byte = [][]byte{
	[]byte("\xc5\xe9\xc2\xd5\xd4\xc9\xc6\xf4"), // svc_EZBNMIF4
}

const (
	svc_EZBNMIF4 = 0
)

func GetsockoptTCPInfo(fd, level, opt int) (*TCPInfo, error) {
	jobname := []byte("\x5c\x40\x40\x40\x40\x40\x40\x40") // "*"
	responseBuffer := [4096]byte{0}
	var bufferAlet, reasonCode uint32 = 0, 0
	var bufferLen, returnValue, returnCode int32 = 4096, 0, 0

	dsa := [18]uint64{0}
	var argv [7]unsafe.Pointer
	argv[0] = unsafe.Pointer(&jobname[0])
	argv[1] = unsafe.Pointer(&responseBuffer[0])
	argv[2] = unsafe.Pointer(&bufferAlet)
	argv[3] = unsafe.Pointer(&bufferLen)
	argv[4] = unsafe.Pointer(&returnValue)
	argv[5] = unsafe.Pointer(&returnCode)
	argv[6] = unsafe.Pointer(&reasonCode)

	request := (*struct {
		header nwmHeader
		filter nwmFilter
	})(unsafe.Pointer(&responseBuffer[0]))

	EZBNMIF4 := svcLoad(&svcNameTable[svc_EZBNMIF4][0])
	if EZBNMIF4 == nil {
		return nil, errnoErr(EINVAL)
	}

	// GetGlobalStats EZBNMIF4 call
	request.header.ident = nwmHeaderIdentifier
	request.header.length = uint32(unsafe.Sizeof(request.header))
	request.header.version = nwmCurrentVer
	request.header.nwmType = nwmGlobalStatsType
	request.header.options = 0x80000000

	svcCall(EZBNMIF4, &argv[0], &dsa[0])

	// outputDesc field is filled by EZBNMIF4 on success
	if returnCode != 0 || request.header.outputDesc.offset == 0 {
		return nil, errnoErr(EINVAL)
	}

	// Check that EZBNMIF4 returned a nwmRecHeader
	recHeader := (*nwmRecHeader)(unsafe.Pointer(&responseBuffer[request.header.outputDesc.offset]))
	if recHeader.ident != nwmRecHeaderIdentifier {
		return nil, errnoErr(EINVAL)
	}

	// Parse nwmTriplets to get offsets of returned entries
	var sections []*uint64
	var sectionDesc *nwmTriplet = (*nwmTriplet)(unsafe.Pointer(&responseBuffer[0]))
	for i := uint32(0); i < uint32(recHeader.number); i++ {
		offset := request.header.outputDesc.offset + uint32(unsafe.Sizeof(*recHeader)) + i*uint32(unsafe.Sizeof(*sectionDesc))
		sectionDesc = (*nwmTriplet)(unsafe.Pointer(&responseBuffer[offset]))
		for j := uint32(0); j < sectionDesc.number; j++ {
			offset = request.header.outputDesc.offset + sectionDesc.offset + j*sectionDesc.length
			sections = append(sections, (*uint64)(unsafe.Pointer(&responseBuffer[offset])))
		}
	}

	// Find nwmTCPStatsEntry in returned entries
	var tcpStats *nwmTCPStatsEntry = nil
	for _, ptr := range sections {
		switch *ptr {
		case nwmTCPStatsIdentifier:
			if tcpStats != nil {
				return nil, errnoErr(EINVAL)
			}
			tcpStats = (*nwmTCPStatsEntry)(unsafe.Pointer(ptr))
		case nwmIPStatsIdentifier:
		case nwmIPGStatsIdentifier:
		case nwmUDPStatsIdentifier:
		case nwmICMPGStatsEntry:
		case nwmICMPTStatsEntry:
		default:
			return nil, errnoErr(EINVAL)
		}
	}
	if tcpStats == nil {
		return nil, errnoErr(EINVAL)
	}

	// GetConnectionDetail EZBNMIF4 call
	responseBuffer = [4096]byte{0}
	dsa = [18]uint64{0}
	bufferAlet, reasonCode = 0, 0
	bufferLen, returnValue, returnCode = 4096, 0, 0
	nameptr := (*uint32)(unsafe.Pointer(uintptr(0x21c))) // Get jobname of current process
	nameptr = (*uint32)(unsafe.Pointer(uintptr(*nameptr + 12)))
	argv[0] = unsafe.Pointer(uintptr(*nameptr))

	request.header.ident = nwmHeaderIdentifier
	request.header.length = uint32(unsafe.Sizeof(request.header))
	request.header.version = nwmCurrentVer
	request.header.nwmType = nwmTCPConnType
	request.header.options = 0x80000000

	request.filter.ident = nwmFilterIdentifier

	var localSockaddr RawSockaddrAny
	socklen := _Socklen(SizeofSockaddrAny)
	err := getsockname(fd, &localSockaddr, &socklen)
	if err != nil {
		return nil, errnoErr(EINVAL)
	}
	if localSockaddr.Addr.Family == AF_INET {
		localSockaddr := (*RawSockaddrInet4)(unsafe.Pointer(&localSockaddr.Addr))
		localSockFilter := (*RawSockaddrInet4)(unsafe.Pointer(&request.filter.local[0]))
		localSockFilter.Family = AF_INET
		var i int
		for i = 0; i < 4; i++ {
			if localSockaddr.Addr[i] != 0 {
				break
			}
		}
		if i != 4 {
			request.filter.flags |= nwmFilterLclAddrMask
			for i = 0; i < 4; i++ {
				localSockFilter.Addr[i] = localSockaddr.Addr[i]
			}
		}
		if localSockaddr.Port != 0 {
			request.filter.flags |= nwmFilterLclPortMask
			localSockFilter.Port = localSockaddr.Port
		}
	} else if localSockaddr.Addr.Family == AF_INET6 {
		localSockaddr := (*RawSockaddrInet6)(unsafe.Pointer(&localSockaddr.Addr))
		localSockFilter := (*RawSockaddrInet6)(unsafe.Pointer(&request.filter.local[0]))
		localSockFilter.Family = AF_INET6
		var i int
		for i = 0; i < 16; i++ {
			if localSockaddr.Addr[i] != 0 {
				break
			}
		}
		if i != 16 {
			request.filter.flags |= nwmFilterLclAddrMask
			for i = 0; i < 16; i++ {
				localSockFilter.Addr[i] = localSockaddr.Addr[i]
			}
		}
		if localSockaddr.Port != 0 {
			request.filter.flags |= nwmFilterLclPortMask
			localSockFilter.Port = localSockaddr.Port
		}
	}

	svcCall(EZBNMIF4, &argv[0], &dsa[0])

	// outputDesc field is filled by EZBNMIF4 on success
	if returnCode != 0 || request.header.outputDesc.offset == 0 {
		return nil, errnoErr(EINVAL)
	}

	// Check that EZBNMIF4 returned a nwmConnEntry
	conn := (*nwmConnEntry)(unsafe.Pointer(&responseBuffer[request.header.outputDesc.offset]))
	if conn.ident != nwmTCPConnIdentifier {
		return nil, errnoErr(EINVAL)
	}

	// Copy data from the returned data structures into tcpInfo
	// Stats from nwmConnEntry are specific to that connection.
	// Stats from nwmTCPStatsEntry are global (to the interface?)
	// Fields may not be an exact match. Some fields have no equivalent.
	var tcpinfo TCPInfo
	tcpinfo.State = uint8(conn.state)
	tcpinfo.Ca_state = 0 // dummy
	tcpinfo.Retransmits = uint8(tcpStats.retransSegs)
	tcpinfo.Probes = uint8(tcpStats.outWinProbes)
	tcpinfo.Backoff = 0 // dummy
	tcpinfo.Options = 0 // dummy
	tcpinfo.Rto = tcpStats.retransTimeouts
	tcpinfo.Ato = tcpStats.outDelayAcks
	tcpinfo.Snd_mss = conn.sendMSS
	tcpinfo.Rcv_mss = conn.sendMSS // dummy
	tcpinfo.Unacked = 0            // dummy
	tcpinfo.Sacked = 0             // dummy
	tcpinfo.Lost = 0               // dummy
	tcpinfo.Retrans = conn.reXmtCount
	tcpinfo.Fackets = 0 // dummy
	tcpinfo.Last_data_sent = uint32(*(*uint64)(unsafe.Pointer(&conn.lastActivity[0])))
	tcpinfo.Last_ack_sent = uint32(*(*uint64)(unsafe.Pointer(&conn.outOldestTime[0])))
	tcpinfo.Last_data_recv = uint32(*(*uint64)(unsafe.Pointer(&conn.inOldestTime[0])))
	tcpinfo.Last_ack_recv = uint32(*(*uint64)(unsafe.Pointer(&conn.inOldestTime[0])))
	tcpinfo.Pmtu = conn.sendMSS // dummy, NWMIfRouteMtu is a candidate
	tcpinfo.Rcv_ssthresh = conn.ssThresh
	tcpinfo.Rtt = conn.roundTripTime
	tcpinfo.Rttvar = conn.roundTripVar
	tcpinfo.Snd_ssthresh = conn.ssThresh // dummy
	tcpinfo.Snd_cwnd = conn.congestionWnd
	tcpinfo.Advmss = conn.sendMSS        // dummy
	tcpinfo.Reordering = 0               // dummy
	tcpinfo.Rcv_rtt = conn.roundTripTime // dummy
	tcpinfo.Rcv_space = conn.sendMSS     // dummy
	tcpinfo.Total_retrans = conn.reXmtCount

	svcUnload(&svcNameTable[svc_EZBNMIF4][0], EZBNMIF4)

	return &tcpinfo, nil
}

// GetsockoptString returns the string value of the socket option opt for the
// socket associated with fd at the given socket level.
func GetsockoptString(fd, level, opt int) (string, error) {
	buf := make([]byte, 256)
	vallen := _Socklen(len(buf))
	err := getsockopt(fd, level, opt, unsafe.Pointer(&buf[0]), &vallen)
	if err != nil {
		return "", err
	}

	return ByteSliceToString(buf[:vallen]), nil
}

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn int, recvflags int, from Sockaddr, err error) {
	var msg Msghdr
	var rsa RawSockaddrAny
	msg.Name = (*byte)(unsafe.Pointer(&rsa))
	msg.Namelen = SizeofSockaddrAny
	var iov Iovec
	if len(p) > 0 {
		iov.Base = (*byte)(unsafe.Pointer(&p[0]))
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		// receive at least one normal byte
		if len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = (*byte)(unsafe.Pointer(&oob[0]))
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = recvmsg(fd, &msg, flags); err != nil {
		return
	}
	oobn = int(msg.Controllen)
	recvflags = int(msg.Flags)
	// source address is only specified if the socket is unconnected
	if rsa.Addr.Family != AF_UNSPEC {
		// TODO(neeilan): Remove 0 arg added to get this compiling on z/OS
		from, err = anyToSockaddr(0, &rsa)
	}
	return
}

func Sendmsg(fd int, p, oob []byte, to Sockaddr, flags int) (err error) {
	_, err = SendmsgN(fd, p, oob, to, flags)
	return
}

func SendmsgN(fd int, p, oob []byte, to Sockaddr, flags int) (n int, err error) {
	var ptr unsafe.Pointer
	var salen _Socklen
	if to != nil {
		var err error
		ptr, salen, err = to.sockaddr()
		if err != nil {
			return 0, err
		}
	}
	var msg Msghdr
	msg.Name = (*byte)(unsafe.Pointer(ptr))
	msg.Namelen = int32(salen)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = (*byte)(unsafe.Pointer(&p[0]))
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		// send at least one normal byte
		if len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = (*byte)(unsafe.Pointer(&oob[0]))
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = sendmsg(fd, &msg, flags); err != nil {
		return 0, err
	}
	if len(oob) > 0 && len(p) == 0 {
		n = 0
	}
	return n, nil
}

func Opendir(name string) (uintptr, error) {
	p, err := BytePtrFromString(name)
	if err != nil {
		return 0, err
	}
	err = nil
	runtime.EnterSyscall()
	dir, e2, e1 := CallLeFuncWithPtrReturn(GetZosLibVec()+SYS___OPENDIR_A<<4, uintptr(unsafe.Pointer(p)))
	runtime.ExitSyscall()
	runtime.KeepAlive(unsafe.Pointer(p))
	if dir == 0 {
		err = errnoErr2(e1, e2)
	}
	return dir, err
}

// clearsyscall.Errno resets the errno value to 0.
func clearErrno()

func Closedir(dir uintptr) error {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_CLOSEDIR<<4, dir)
	runtime.ExitSyscall()
	if r0 != 0 {
		return errnoErr2(e1, e2)
	}
	return nil
}

func Seekdir(dir uintptr, pos int) {
	runtime.EnterSyscall()
	CallLeFuncWithErr(GetZosLibVec()+SYS_SEEKDIR<<4, dir, uintptr(pos))
	runtime.ExitSyscall()
}

func Telldir(dir uintptr) (int, error) {
	p, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_TELLDIR<<4, dir)
	pos := int(p)
	if int64(p) == -1 {
		return pos, errnoErr2(e1, e2)
	}
	return pos, nil
}

// FcntlFlock performs a fcntl syscall for the F_GETLK, F_SETLK or F_SETLKW command.
func FcntlFlock(fd uintptr, cmd int, lk *Flock_t) error {
	// struct flock is packed on z/OS. We can't emulate that in Go so
	// instead we pack it here.
	var flock [24]byte
	*(*int16)(unsafe.Pointer(&flock[0])) = lk.Type
	*(*int16)(unsafe.Pointer(&flock[2])) = lk.Whence
	*(*int64)(unsafe.Pointer(&flock[4])) = lk.Start
	*(*int64)(unsafe.Pointer(&flock[12])) = lk.Len
	*(*int32)(unsafe.Pointer(&flock[20])) = lk.Pid
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_FCNTL<<4, fd, uintptr(cmd), uintptr(unsafe.Pointer(&flock)))
	runtime.ExitSyscall()
	lk.Type = *(*int16)(unsafe.Pointer(&flock[0]))
	lk.Whence = *(*int16)(unsafe.Pointer(&flock[2]))
	lk.Start = *(*int64)(unsafe.Pointer(&flock[4]))
	lk.Len = *(*int64)(unsafe.Pointer(&flock[12]))
	lk.Pid = *(*int32)(unsafe.Pointer(&flock[20]))
	if r0 == 0 {
		return nil
	}
	return errnoErr2(e1, e2)
}

func impl_Flock(fd int, how int) (err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_FLOCK<<4, uintptr(fd), uintptr(how))
	runtime.ExitSyscall()
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//go:nosplit
func get_FlockAddr() *(func(fd int, how int) (err error))

var Flock = enter_Flock

func validFlock(fp uintptr) bool {
	if funcptrtest(GetZosLibVec()+SYS_FLOCK<<4, "") == 0 {
		if name, err := getLeFuncName(GetZosLibVec() + SYS_FLOCK<<4); err == nil {
			return name == "flock"
		}
	}
	return false
}

func enter_Flock(fd int, how int) (err error) {
	funcref := get_FlockAddr()
	if validFlock(GetZosLibVec() + SYS_FLOCK<<4) {
		*funcref = impl_Flock
	} else {
		*funcref = legacyFlock
	}
	return (*funcref)(fd, how)
}

func legacyFlock(fd int, how int) error {

	var flock_type int16
	var fcntl_cmd int

	switch how {
	case LOCK_SH | LOCK_NB:
		flock_type = F_RDLCK
		fcntl_cmd = F_SETLK
	case LOCK_EX | LOCK_NB:
		flock_type = F_WRLCK
		fcntl_cmd = F_SETLK
	case LOCK_EX:
		flock_type = F_WRLCK
		fcntl_cmd = F_SETLKW
	case LOCK_UN:
		flock_type = F_UNLCK
		fcntl_cmd = F_SETLKW
	default:
	}

	flock := Flock_t{
		Type:   int16(flock_type),
		Whence: int16(0),
		Start:  int64(0),
		Len:    int64(0),
		Pid:    int32(Getppid()),
	}

	err := FcntlFlock(uintptr(fd), fcntl_cmd, &flock)
	return err
}

func Mlock(b []byte) (err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___MLOCKALL<<4, _BPX_NONSWAP)
	runtime.ExitSyscall()
	if r0 != 0 {
		err = errnoErr2(e1, e2)
	}
	return
}

func Mlock2(b []byte, flags int) (err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___MLOCKALL<<4, _BPX_NONSWAP)
	runtime.ExitSyscall()
	if r0 != 0 {
		err = errnoErr2(e1, e2)
	}
	return
}

func Mlockall(flags int) (err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___MLOCKALL<<4, _BPX_NONSWAP)
	runtime.ExitSyscall()
	if r0 != 0 {
		err = errnoErr2(e1, e2)
	}
	return
}

func Munlock(b []byte) (err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___MLOCKALL<<4, _BPX_SWAP)
	runtime.ExitSyscall()
	if r0 != 0 {
		err = errnoErr2(e1, e2)
	}
	return
}

func Munlockall() (err error) {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___MLOCKALL<<4, _BPX_SWAP)
	runtime.ExitSyscall()
	if r0 != 0 {
		err = errnoErr2(e1, e2)
	}
	return
}

func ClockGettime(clockid int32, ts *Timespec) error {

	var ticks_per_sec uint32 = 100 //TODO(kenan): value is currently hardcoded; need sysconf() call otherwise
	var nsec_per_sec int64 = 1000000000

	if ts == nil {
		return EFAULT
	}
	if clockid == CLOCK_REALTIME || clockid == CLOCK_MONOTONIC {
		var nanotime int64 = runtime.Nanotime1()
		ts.Sec = nanotime / nsec_per_sec
		ts.Nsec = nanotime % nsec_per_sec
	} else if clockid == CLOCK_PROCESS_CPUTIME_ID || clockid == CLOCK_THREAD_CPUTIME_ID {
		var tm Tms
		_, err := Times(&tm)
		if err != nil {
			return EFAULT
		}
		ts.Sec = int64(tm.Utime / ticks_per_sec)
		ts.Nsec = int64(tm.Utime) * nsec_per_sec / int64(ticks_per_sec)
	} else {
		return EINVAL
	}
	return nil
}

// Chtag

//go:nosplit
func get_ChtagAddr() *(func(path string, ccsid uint64, textbit uint64) error)

var Chtag = enter_Chtag

func enter_Chtag(path string, ccsid uint64, textbit uint64) error {
	funcref := get_ChtagAddr()
	if validSetxattr() {
		*funcref = impl_Chtag
	} else {
		*funcref = legacy_Chtag
	}
	return (*funcref)(path, ccsid, textbit)
}

func legacy_Chtag(path string, ccsid uint64, textbit uint64) error {
	tag := ccsid<<16 | textbit<<15
	var tag_buff [8]byte
	DecodeData(tag_buff[:], 8, tag)
	return Setxattr(path, "filetag", tag_buff[:], XATTR_REPLACE)
}

func impl_Chtag(path string, ccsid uint64, textbit uint64) error {
	tag := ccsid<<16 | textbit<<15
	var tag_buff [4]byte
	DecodeData(tag_buff[:], 4, tag)
	return Setxattr(path, "system.filetag", tag_buff[:], XATTR_REPLACE)
}

// End of Chtag

// Nanosleep

//go:nosplit
func get_NanosleepAddr() *(func(time *Timespec, leftover *Timespec) error)

var Nanosleep = enter_Nanosleep

func enter_Nanosleep(time *Timespec, leftover *Timespec) error {
	funcref := get_NanosleepAddr()
	if funcptrtest(GetZosLibVec()+SYS_NANOSLEEP<<4, "") == 0 {
		*funcref = impl_Nanosleep
	} else {
		*funcref = legacyNanosleep
	}
	return (*funcref)(time, leftover)
}

func impl_Nanosleep(time *Timespec, leftover *Timespec) error {
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS_NANOSLEEP<<4, uintptr(unsafe.Pointer(time)), uintptr(unsafe.Pointer(leftover)))
	runtime.ExitSyscall()
	if int64(r0) == -1 {
		return errnoErr2(e1, e2)
	}
	return nil
}

func legacyNanosleep(time *Timespec, leftover *Timespec) error {
	t0 := runtime.Nanotime1()
	var secrem uint32
	var nsecrem uint32
	total := time.Sec*1000000000 + time.Nsec
	elapsed := runtime.Nanotime1() - t0
	var rv int32
	var rc int32
	var err error
	// repeatedly sleep for 1 second until less than 1 second left
	for total-elapsed > 1000000000 {
		rv, rc, _ = BpxCondTimedWait(uint32(1), uint32(0), uint32(CW_CONDVAR), &secrem, &nsecrem)
		if rv != 0 && rc != 112 { // 112 is EAGAIN
			if leftover != nil && rc == 120 { // 120 is EINTR
				leftover.Sec = int64(secrem)
				leftover.Nsec = int64(nsecrem)
			}
			err = Errno(rc)
			return err
		}
		elapsed = runtime.Nanotime1() - t0
	}
	// sleep the remainder
	if total > elapsed {
		rv, rc, _ = BpxCondTimedWait(uint32(0), uint32(total-elapsed), uint32(CW_CONDVAR), &secrem, &nsecrem)
	}
	if leftover != nil && rc == 120 {
		leftover.Sec = int64(secrem)
		leftover.Nsec = int64(nsecrem)
	}
	if rv != 0 && rc != 112 {
		err = Errno(rc)
	}
	return err
}

// End of Nanosleep

var (
	Stdin  = 0
	Stdout = 1
	Stderr = 2
)

// Do the interface allocations only once for common
// Errno values.
var (
	errEAGAIN error = syscall.EAGAIN
	errEINVAL error = syscall.EINVAL
	errENOENT error = syscall.ENOENT
)

var ZosTraceLevel int
var ZosTracefile *os.File

var (
	signalNameMapOnce sync.Once
	signalNameMap     map[string]syscall.Signal
)

// errnoErr returns common boxed Errno values, to prevent
// allocations at runtime.
func errnoErr(e Errno) error {
	switch e {
	case 0:
		return nil
	case EAGAIN:
		return errEAGAIN
	case EINVAL:
		return errEINVAL
	case ENOENT:
		return errENOENT
	}
	return e
}

var reg *regexp.Regexp

// enhanced with zos specific errno2
func errnoErr2(e Errno, e2 uintptr) error {
	switch e {
	case 0:
		return nil
	case EAGAIN:
		return errEAGAIN
		/*
			Allow the retrieval of errno2 for EINVAL and ENOENT on zos
				case EINVAL:
					return errEINVAL
				case ENOENT:
					return errENOENT
		*/
	}
	if ZosTraceLevel > 0 {
		var name string
		if reg == nil {
			reg = regexp.MustCompile("(^unix\\.[^/]+$|.*\\/unix\\.[^/]+$)")
		}
		i := 1
		pc, file, line, ok := runtime.Caller(i)
		if ok {
			name = runtime.FuncForPC(pc).Name()
		}
		for ok && reg.MatchString(runtime.FuncForPC(pc).Name()) {
			i += 1
			pc, file, line, ok = runtime.Caller(i)
		}
		if ok {
			if ZosTracefile == nil {
				ZosConsolePrintf("From %s:%d\n", file, line)
				ZosConsolePrintf("%s: %s (errno2=0x%x)\n", name, e.Error(), e2)
			} else {
				fmt.Fprintf(ZosTracefile, "From %s:%d\n", file, line)
				fmt.Fprintf(ZosTracefile, "%s: %s (errno2=0x%x)\n", name, e.Error(), e2)
			}
		} else {
			if ZosTracefile == nil {
				ZosConsolePrintf("%s (errno2=0x%x)\n", e.Error(), e2)
			} else {
				fmt.Fprintf(ZosTracefile, "%s (errno2=0x%x)\n", e.Error(), e2)
			}
		}
	}
	return e
}

// ErrnoName returns the error name for error number e.
func ErrnoName(e Errno) string {
	i := sort.Search(len(errorList), func { i -> errorList[i].num >= e })
	if i < len(errorList) && errorList[i].num == e {
		return errorList[i].name
	}
	return ""
}

// SignalName returns the signal name for signal number s.
func SignalName(s syscall.Signal) string {
	i := sort.Search(len(signalList), func { i -> signalList[i].num >= s })
	if i < len(signalList) && signalList[i].num == s {
		return signalList[i].name
	}
	return ""
}

// SignalNum returns the syscall.Signal for signal named s,
// or 0 if a signal with such name is not found.
// The signal name should start with "SIG".
func SignalNum(s string) syscall.Signal {
	signalNameMapOnce.Do(func() {
		signalNameMap = make(map[string]syscall.Signal, len(signalList))
		for _, signal := range signalList {
			signalNameMap[signal.name] = signal.num
		}
	})
	return signalNameMap[s]
}

// clen returns the index of the first NULL byte in n or len(n) if n contains no NULL byte.
func clen(n []byte) int {
	i := bytes.IndexByte(n, 0)
	if i == -1 {
		i = len(n)
	}
	return i
}

// Mmap manager, for use by operating system-specific implementations.

type mmapper struct {
	sync.Mutex
	active map[*byte][]byte // active mappings; key is last byte in mapping
	mmap   func(addr, length uintptr, prot, flags, fd int, offset int64) (uintptr, error)
	munmap func(addr uintptr, length uintptr) error
}

func (m *mmapper) Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	if length <= 0 {
		return nil, EINVAL
	}

	// Set __MAP_64 by default
	flags |= __MAP_64

	// Map the requested memory.
	addr, errno := m.mmap(0, uintptr(length), prot, flags, fd, offset)
	if errno != nil {
		return nil, errno
	}

	// Slice memory layout
	var sl = struct {
		addr uintptr
		len  int
		cap  int
	}{addr, length, length}

	// Use unsafe to turn sl into a []byte.
	b := *(*[]byte)(unsafe.Pointer(&sl))

	// Register mapping in m and return it.
	p := &b[cap(b)-1]
	m.Lock()
	defer m.Unlock()
	m.active[p] = b
	return b, nil
}

func (m *mmapper) Munmap(data []byte) (err error) {
	if len(data) == 0 || len(data) != cap(data) {
		return EINVAL
	}

	// Find the base of the mapping.
	p := &data[cap(data)-1]
	m.Lock()
	defer m.Unlock()
	b := m.active[p]
	if b == nil || &b[0] != &data[0] {
		return EINVAL
	}

	// Unmap the memory and update m.
	if errno := m.munmap(uintptr(unsafe.Pointer(&b[0])), uintptr(len(b))); errno != nil {
		return errno
	}
	delete(m.active, p)
	return nil
}

func Read(fd int, p []byte) (n int, err error) {
	n, err = read(fd, p)
	if raceenabled {
		if n > 0 {
			raceWriteRange(unsafe.Pointer(&p[0]), n)
		}
		if err == nil {
			raceAcquire(unsafe.Pointer(&ioSync))
		}
	}
	return
}

func Write(fd int, p []byte) (n int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = write(fd, p)
	if raceenabled && n > 0 {
		raceReadRange(unsafe.Pointer(&p[0]), n)
	}
	return
}

// For testing: clients can set this flag to force
// creation of IPv6 sockets to return EAFNOSUPPORT.
var SocketDisableIPv6 bool

// Sockaddr represents a socket address.
type Sockaddr interface {
	sockaddr() (ptr unsafe.Pointer, len _Socklen, err error) // lowercase; only we can define Sockaddrs
}

// SockaddrInet4 implements the Sockaddr interface for AF_INET type sockets.
type SockaddrInet4 struct {
	Port int
	Addr [4]byte
	raw  RawSockaddrInet4
}

// SockaddrInet6 implements the Sockaddr interface for AF_INET6 type sockets.
type SockaddrInet6 struct {
	Port   int
	ZoneId uint32
	Addr   [16]byte
	raw    RawSockaddrInet6
}

// SockaddrUnix implements the Sockaddr interface for AF_UNIX type sockets.
type SockaddrUnix struct {
	Name string
	raw  RawSockaddrUnix
}

func Bind(fd int, sa Sockaddr) (err error) {
	ptr, n, err := sa.sockaddr()
	if err != nil {
		return err
	}
	return bind(fd, ptr, n)
}

func Connect(fd int, sa Sockaddr) (err error) {
	ptr, n, err := sa.sockaddr()
	if err != nil {
		return err
	}
	return connect(fd, ptr, n)
}

func Getpeername(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if err = getpeername(fd, &rsa, &len); err != nil {
		return
	}
	return anyToSockaddr(fd, &rsa)
}

func GetsockoptByte(fd, level, opt int) (value byte, err error) {
	var n byte
	vallen := _Socklen(1)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&n), &vallen)
	return n, err
}

func GetsockoptInt(fd, level, opt int) (value int, err error) {
	var n int32
	vallen := _Socklen(4)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&n), &vallen)
	return int(n), err
}

func GetsockoptInet4Addr(fd, level, opt int) (value [4]byte, err error) {
	vallen := _Socklen(4)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&value[0]), &vallen)
	return value, err
}

func GetsockoptIPMreq(fd, level, opt int) (*IPMreq, error) {
	var value IPMreq
	vallen := _Socklen(SizeofIPMreq)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptIPv6Mreq(fd, level, opt int) (*IPv6Mreq, error) {
	var value IPv6Mreq
	vallen := _Socklen(SizeofIPv6Mreq)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptIPv6MTUInfo(fd, level, opt int) (*IPv6MTUInfo, error) {
	var value IPv6MTUInfo
	vallen := _Socklen(SizeofIPv6MTUInfo)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptICMPv6Filter(fd, level, opt int) (*ICMPv6Filter, error) {
	var value ICMPv6Filter
	vallen := _Socklen(SizeofICMPv6Filter)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptLinger(fd, level, opt int) (*Linger, error) {
	var linger Linger
	vallen := _Socklen(SizeofLinger)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&linger), &vallen)
	return &linger, err
}

func GetsockoptTimeval(fd, level, opt int) (*Timeval, error) {
	var tv Timeval
	vallen := _Socklen(unsafe.Sizeof(tv))
	err := getsockopt(fd, level, opt, unsafe.Pointer(&tv), &vallen)
	return &tv, err
}

func GetsockoptUint64(fd, level, opt int) (value uint64, err error) {
	var n uint64
	vallen := _Socklen(8)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&n), &vallen)
	return n, err
}

func Recvfrom(fd int, p []byte, flags int) (n int, from Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if n, err = recvfrom(fd, p, flags, &rsa, &len); err != nil {
		return
	}
	if rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(fd, &rsa)
	}
	return
}

func Sendto(fd int, p []byte, flags int, to Sockaddr) (err error) {
	ptr, n, err := to.sockaddr()
	if err != nil {
		return err
	}
	return sendto(fd, p, flags, ptr, n)
}

func SetsockoptByte(fd, level, opt int, value byte) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(&value), 1)
}

func SetsockoptInt(fd, level, opt int, value int) (err error) {
	var n = int32(value)
	return setsockopt(fd, level, opt, unsafe.Pointer(&n), 4)
}

func SetsockoptInet4Addr(fd, level, opt int, value [4]byte) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(&value[0]), 4)
}

func SetsockoptIPMreq(fd, level, opt int, mreq *IPMreq) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), SizeofIPMreq)
}

func SetsockoptIPv6Mreq(fd, level, opt int, mreq *IPv6Mreq) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), SizeofIPv6Mreq)
}

func SetsockoptICMPv6Filter(fd, level, opt int, filter *ICMPv6Filter) error {
	return setsockopt(fd, level, opt, unsafe.Pointer(filter), SizeofICMPv6Filter)
}

func SetsockoptLinger(fd, level, opt int, l *Linger) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(l), SizeofLinger)
}

func SetsockoptString(fd, level, opt int, s string) (err error) {
	var p unsafe.Pointer
	if len(s) > 0 {
		p = unsafe.Pointer(&[]byte(s)[0])
	}
	return setsockopt(fd, level, opt, p, uintptr(len(s)))
}

func SetsockoptTimeval(fd, level, opt int, tv *Timeval) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(tv), unsafe.Sizeof(*tv))
}

func SetsockoptUint64(fd, level, opt int, value uint64) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(&value), 8)
}

func Socket(domain, typ, proto int) (fd int, err error) {
	if domain == AF_INET6 && SocketDisableIPv6 {
		return -1, EAFNOSUPPORT
	}
	fd, err = socket(domain, typ, proto)
	return
}

func Socketpair(domain, typ, proto int) (fd [2]int, err error) {
	var fdx [2]int32
	err = socketpair(domain, typ, proto, &fdx)
	if err == nil {
		fd[0] = int(fdx[0])
		fd[1] = int(fdx[1])
	}
	return
}

var ioSync int64

func CloseOnExec(fd int) { fcntl(fd, F_SETFD, FD_CLOEXEC) }

func SetNonblock(fd int, nonblocking bool) (err error) {
	flag, err := fcntl(fd, F_GETFL, 0)
	if err != nil {
		return err
	}
	if nonblocking {
		flag |= O_NONBLOCK
	} else {
		flag &= ^O_NONBLOCK
	}
	_, err = fcntl(fd, F_SETFL, flag)
	return err
}

// Exec calls execve(2), which replaces the calling executable in the process
// tree. argv0 should be the full path to an executable ("/bin/ls") and the
// executable name should also be the first argument in argv (["ls", "-l"]).
// envv are the environment variables that should be passed to the new
// process (["USER=go", "PWD=/tmp"]).
func Exec(argv0 string, argv []string, envv []string) error {
	return syscall.Exec(argv0, argv, envv)
}

func Getag(path string) (ccsid uint16, flag uint16, err error) {
	var val [8]byte
	sz, err := Getxattr(path, "ccsid", val[:])
	if err != nil {
		return
	}
	ccsid = uint16(EncodeData(val[0:sz]))
	sz, err = Getxattr(path, "flags", val[:])
	if err != nil {
		return
	}
	flag = uint16(EncodeData(val[0:sz]) >> 15)
	return
}

// Mount begin
func impl_Mount(source string, target string, fstype string, flags uintptr, data string) (err error) {
	var _p0 *byte
	_p0, err = BytePtrFromString(source)
	if err != nil {
		return
	}
	var _p1 *byte
	_p1, err = BytePtrFromString(target)
	if err != nil {
		return
	}
	var _p2 *byte
	_p2, err = BytePtrFromString(fstype)
	if err != nil {
		return
	}
	var _p3 *byte
	_p3, err = BytePtrFromString(data)
	if err != nil {
		return
	}
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___MOUNT1_A<<4, uintptr(unsafe.Pointer(_p0)), uintptr(unsafe.Pointer(_p1)), uintptr(unsafe.Pointer(_p2)), uintptr(flags), uintptr(unsafe.Pointer(_p3)))
	runtime.ExitSyscall()
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//go:nosplit
func get_MountAddr() *(func(source string, target string, fstype string, flags uintptr, data string) (err error))

var Mount = enter_Mount

func enter_Mount(source string, target string, fstype string, flags uintptr, data string) (err error) {
	funcref := get_MountAddr()
	if validMount() {
		*funcref = impl_Mount
	} else {
		*funcref = legacyMount
	}
	return (*funcref)(source, target, fstype, flags, data)
}

func legacyMount(source string, target string, fstype string, flags uintptr, data string) (err error) {
	if needspace := 8 - len(fstype); needspace <= 0 {
		fstype = fstype[0:8]
	} else {
		fstype += "        "[0:needspace]
	}
	return mount_LE(target, source, fstype, uint32(flags), int32(len(data)), data)
}

func validMount() bool {
	if funcptrtest(GetZosLibVec()+SYS___MOUNT1_A<<4, "") == 0 {
		if name, err := getLeFuncName(GetZosLibVec() + SYS___MOUNT1_A<<4); err == nil {
			return name == "__mount1_a"
		}
	}
	return false
}

// Mount end

// Unmount begin
func impl_Unmount(target string, flags int) (err error) {
	var _p0 *byte
	_p0, err = BytePtrFromString(target)
	if err != nil {
		return
	}
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___UMOUNT2_A<<4, uintptr(unsafe.Pointer(_p0)), uintptr(flags))
	runtime.ExitSyscall()
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//go:nosplit
func get_UnmountAddr() *(func(target string, flags int) (err error))

var Unmount = enter_Unmount

func enter_Unmount(target string, flags int) (err error) {
	funcref := get_UnmountAddr()
	if funcptrtest(GetZosLibVec()+SYS___UMOUNT2_A<<4, "") == 0 {
		*funcref = impl_Unmount
	} else {
		*funcref = legacyUnmount
	}
	return (*funcref)(target, flags)
}

func legacyUnmount(name string, mtm int) (err error) {
	// mountpoint is always a full path and starts with a '/'
	// check if input string is not a mountpoint but a filesystem name
	if name[0] != '/' {
		return unmount_LE(name, mtm)
	}
	// treat name as mountpoint
	b2s := func(arr []byte) string {
		var str string
		for i := 0; i < len(arr); i++ {
			if arr[i] == 0 {
				str = string(arr[:i])
				break
			}
		}
		return str
	}
	var buffer struct {
		header W_Mnth
		fsinfo [64]W_Mntent
	}
	fs_count, err := W_Getmntent_A((*byte)(unsafe.Pointer(&buffer)), int(unsafe.Sizeof(buffer)))
	if err == nil {
		err = EINVAL
		for i := 0; i < fs_count; i++ {
			if b2s(buffer.fsinfo[i].Mountpoint[:]) == name {
				err = unmount_LE(b2s(buffer.fsinfo[i].Fsname[:]), mtm)
				break
			}
		}
	} else if fs_count == 0 {
		err = EINVAL
	}
	return err
}

// Unmount end

func direntIno(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Ino), unsafe.Sizeof(Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Reclen), unsafe.Sizeof(Dirent{}.Reclen))
}

func direntNamlen(buf []byte) (uint64, bool) {
	reclen, ok := direntReclen(buf)
	if !ok {
		return 0, false
	}
	return reclen - uint64(unsafe.Offsetof(Dirent{}.Name)), true
}

func direntLeToDirentUnix(dirent *direntLE, dir uintptr, path string) (Dirent, error) {
	var d Dirent

	d.Ino = uint64(dirent.Ino)
	offset, err := Telldir(dir)
	if err != nil {
		return d, err
	}

	d.Off = int64(offset)
	s := string(bytes.Split(dirent.Name[:], []byte{0})[0])
	copy(d.Name[:], s)

	d.Reclen = uint16(24 + len(d.NameString()))
	var st Stat_t
	path = path + "/" + s
	err = Lstat(path, &st)
	if err != nil {
		return d, err
	}

	d.Type = uint8(st.Mode >> 24)
	return d, err
}

func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	// Simulation of Getdirentries port from the Darwin implementation.
	// COMMENTS FROM DARWIN:
	// It's not the full required semantics, but should handle the case
	// of calling Getdirentries or ReadDirent repeatedly.
	// It won't handle assigning the results of lseek to *basep, or handle
	// the directory being edited underfoot.

	skip, err := Seek(fd, 0, 1 /* SEEK_CUR */)
	if err != nil {
		return 0, err
	}

	// Get path from fd to avoid unavailable call (fdopendir)
	path, err := ZosFdToPath(fd)
	if err != nil {
		return 0, err
	}
	d, err := Opendir(path)
	if err != nil {
		return 0, err
	}
	defer Closedir(d)

	var cnt int64
	for {
		var entryLE direntLE
		var entrypLE *direntLE
		e := Readdir_r(d, &entryLE, &entrypLE)
		if e != nil {
			return n, e
		}
		if entrypLE == nil {
			break
		}
		if skip > 0 {
			skip--
			cnt++
			continue
		}

		// Dirent on zos has a different structure
		entry, e := direntLeToDirentUnix(&entryLE, d, path)
		if e != nil {
			return n, e
		}

		reclen := int(entry.Reclen)
		if reclen > len(buf) {
			// Not enough room. Return for now.
			// The counter will let us know where we should start up again.
			// Note: this strategy for suspending in the middle and
			// restarting is O(n^2) in the length of the directory. Oh well.
			break
		}

		// Copy entry into return buffer.
		s := unsafe.Slice((*byte)(unsafe.Pointer(&entry)), reclen)
		copy(buf, s)

		buf = buf[reclen:]
		n += reclen
		cnt++
	}
	// Set the seek offset of the input fd to record
	// how many files we've already returned.
	_, err = Seek(fd, cnt, 0 /* SEEK_SET */)
	if err != nil {
		return n, err
	}

	return n, nil
}

func Err2ad() (eadd *int) {
	r0, _, _ := CallLeFuncWithErr(GetZosLibVec() + SYS___ERR2AD<<4)
	eadd = (*int)(unsafe.Pointer(r0))
	return
}

func ZosConsolePrintf(format string, v ...interface{}) (int, error) {
	type __cmsg struct {
		_            uint16
		_            [2]uint8
		__msg_length uint32
		__msg        uintptr
		_            [4]uint8
	}
	msg := fmt.Sprintf(format, v...)
	strptr := unsafe.Pointer((*reflect.StringHeader)(unsafe.Pointer(&msg)).Data)
	len := (*reflect.StringHeader)(unsafe.Pointer(&msg)).Len
	cmsg := __cmsg{__msg_length: uint32(len), __msg: uintptr(strptr)}
	cmd := uint32(0)
	runtime.EnterSyscall()
	rc, err2, err1 := CallLeFuncWithErr(GetZosLibVec()+SYS_____CONSOLE_A<<4, uintptr(unsafe.Pointer(&cmsg)), 0, uintptr(unsafe.Pointer(&cmd)))
	runtime.ExitSyscall()
	if rc != 0 {
		return 0, fmt.Errorf("%s (errno2=0x%x)\n", err1.Error(), err2)
	}
	return 0, nil
}
func ZosStringToEbcdicBytes(str string, nullterm bool) (ebcdicBytes []byte) {
	if nullterm {
		ebcdicBytes = []byte(str + "\x00")
	} else {
		ebcdicBytes = []byte(str)
	}
	A2e(ebcdicBytes)
	return
}
func ZosEbcdicBytesToString(b []byte, trimRight bool) (str string) {
	res := make([]byte, len(b))
	copy(res, b)
	E2a(res)
	if trimRight {
		str = string(bytes.TrimRight(res, " \x00"))
	} else {
		str = string(res)
	}
	return
}

func fdToPath(dirfd int) (path string, err error) {
	var buffer [1024]byte
	// w_ctrl()
	ret := runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS_W_IOCTL<<4,
		[]uintptr{uintptr(dirfd), 17, 1024, uintptr(unsafe.Pointer(&buffer[0]))})
	if ret == 0 {
		zb := bytes.IndexByte(buffer[:], 0)
		if zb == -1 {
			zb = len(buffer)
		}
		// __e2a_l()
		runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___E2A_L<<4,
			[]uintptr{uintptr(unsafe.Pointer(&buffer[0])), uintptr(zb)})
		return string(buffer[:zb]), nil
	}
	// __errno()
	errno := int(*(*int32)(unsafe.Pointer(runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___ERRNO<<4,
		[]uintptr{}))))
	// __errno2()
	errno2 := int(runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS___ERRNO2<<4,
		[]uintptr{}))
	// strerror_r()
	ret = runtime.CallLeFuncByPtr(runtime.XplinkLibvec+SYS_STRERROR_R<<4,
		[]uintptr{uintptr(errno), uintptr(unsafe.Pointer(&buffer[0])), 1024})
	if ret == 0 {
		zb := bytes.IndexByte(buffer[:], 0)
		if zb == -1 {
			zb = len(buffer)
		}
		return "", fmt.Errorf("%s (errno2=0x%x)", buffer[:zb], errno2)
	} else {
		return "", fmt.Errorf("fdToPath errno %d (errno2=0x%x)", errno, errno2)
	}
}

func impl_Mkfifoat(dirfd int, path string, mode uint32) (err error) {
	var _p0 *byte
	_p0, err = BytePtrFromString(path)
	if err != nil {
		return
	}
	runtime.EnterSyscall()
	r0, e2, e1 := CallLeFuncWithErr(GetZosLibVec()+SYS___MKFIFOAT_A<<4, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode))
	runtime.ExitSyscall()
	if int64(r0) == -1 {
		err = errnoErr2(e1, e2)
	}
	return
}

//go:nosplit
func get_MkfifoatAddr() *(func(dirfd int, path string, mode uint32) (err error))

var Mkfifoat = enter_Mkfifoat

func enter_Mkfifoat(dirfd int, path string, mode uint32) (err error) {
	funcref := get_MkfifoatAddr()
	if funcptrtest(GetZosLibVec()+SYS___MKFIFOAT_A<<4, "") == 0 {
		*funcref = impl_Mkfifoat
	} else {
		*funcref = legacy_Mkfifoat
	}
	return (*funcref)(dirfd, path, mode)
}

func legacy_Mkfifoat(dirfd int, path string, mode uint32) (err error) {
	dirname, err := ZosFdToPath(dirfd)
	if err != nil {
		return err
	}
	return Mkfifo(dirname+"/"+path, mode)
}

//sys	Posix_openpt(oflag int) (fd int, err error) = SYS_POSIX_OPENPT
//sys	Grantpt(fildes int) (rc int, err error) = SYS_GRANTPT
//sys	Unlockpt(fildes int) (rc int, err error) = SYS_UNLOCKPT
