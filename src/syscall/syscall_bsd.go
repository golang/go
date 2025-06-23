// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

// BSD system call wrappers shared by *BSD based systems
// including OS X (Darwin) and FreeBSD.  Like the other
// syscall_*.go files it is compiled as Go code but also
// used as input to mksyscall which parses the //sys
// lines and generates system call stubs.

package syscall

import (
	"runtime"
	"unsafe"
)

const ImplementsGetwd = true

func Getwd() (string, error) {
	var buf [pathMax]byte
	_, err := getcwd(buf[:])
	if err != nil {
		return "", err
	}
	n := clen(buf[:])
	if n < 1 {
		return "", EINVAL
	}
	return string(buf[:n]), nil
}

/*
 * Wrapped
 */

//sysnb	getgroups(ngid int, gid *_Gid_t) (n int, err error)
//sysnb	setgroups(ngid int, gid *_Gid_t) (err error)

func Getgroups() (gids []int, err error) {
	n, err := getgroups(0, nil)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}

	// Sanity check group count. Max is 16 on BSD.
	if n < 0 || n > 1000 {
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

func ReadDirent(fd int, buf []byte) (n int, err error) {
	// Final argument is (basep *uintptr) and the syscall doesn't take nil.
	// 64 bits should be enough. (32 bits isn't even on 386). Since the
	// actual system call is getdirentries64, 64 is a good guess.
	// TODO(rsc): Can we use a single global basep for all calls?
	var base = (*uintptr)(unsafe.Pointer(new(uint64)))
	return Getdirentries(fd, buf, base)
}

// Wait status is 7 bits at bottom, either 0 (exited),
// 0x7F (stopped), or a signal number that caused an exit.
// The 0x80 bit is whether there was a core dump.
// An extra number (exit code, signal causing a stop)
// is in the high bits.

type WaitStatus uint32

const (
	mask  = 0x7F
	core  = 0x80
	shift = 8

	exited  = 0
	stopped = 0x7F
)

func (w WaitStatus) Exited() bool { return w&mask == exited }

func (w WaitStatus) ExitStatus() int {
	if w&mask != exited {
		return -1
	}
	return int(w >> shift)
}

func (w WaitStatus) Signaled() bool { return w&mask != stopped && w&mask != 0 }

func (w WaitStatus) Signal() Signal {
	sig := Signal(w & mask)
	if sig == stopped || sig == 0 {
		return -1
	}
	return sig
}

func (w WaitStatus) CoreDump() bool { return w.Signaled() && w&core != 0 }

func (w WaitStatus) Stopped() bool { return w&mask == stopped && Signal(w>>shift) != SIGSTOP }

func (w WaitStatus) Continued() bool { return w&mask == stopped && Signal(w>>shift) == SIGSTOP }

func (w WaitStatus) StopSignal() Signal {
	if !w.Stopped() {
		return -1
	}
	return Signal(w>>shift) & 0xFF
}

func (w WaitStatus) TrapCause() int { return -1 }

//sys	wait4(pid int, wstatus *_C_int, options int, rusage *Rusage) (wpid int, err error)

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	var status _C_int
	wpid, err = wait4(pid, &status, options, rusage)
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return
}

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, err error)
//sys	bind(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sys	connect(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sysnb	socket(domain int, typ int, proto int) (fd int, err error)
//sys	getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *_Socklen) (err error)
//sys	setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) (err error)
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sysnb	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sys	Shutdown(s int, how int) (err error)

func (sa *SockaddrInet4) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Len = SizeofSockaddrInet4
	sa.raw.Family = AF_INET
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	sa.raw.Addr = sa.Addr
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
	sa.raw.Addr = sa.Addr
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

func (sa *SockaddrDatalink) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Index == 0 {
		return nil, 0, EINVAL
	}
	sa.raw.Len = sa.Len
	sa.raw.Family = AF_LINK
	sa.raw.Index = sa.Index
	sa.raw.Type = sa.Type
	sa.raw.Nlen = sa.Nlen
	sa.raw.Alen = sa.Alen
	sa.raw.Slen = sa.Slen
	sa.raw.Data = sa.Data
	return unsafe.Pointer(&sa.raw), SizeofSockaddrDatalink, nil
}

func anyToSockaddr(rsa *RawSockaddrAny) (Sockaddr, error) {
	switch rsa.Addr.Family {
	case AF_LINK:
		pp := (*RawSockaddrDatalink)(unsafe.Pointer(rsa))
		sa := new(SockaddrDatalink)
		sa.Len = pp.Len
		sa.Family = pp.Family
		sa.Index = pp.Index
		sa.Type = pp.Type
		sa.Nlen = pp.Nlen
		sa.Alen = pp.Alen
		sa.Slen = pp.Slen
		sa.Data = pp.Data
		return sa, nil

	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa))
		if pp.Len < 2 || pp.Len > SizeofSockaddrUnix {
			return nil, EINVAL
		}
		sa := new(SockaddrUnix)

		// Some BSDs include the trailing NUL in the length, whereas
		// others do not. Work around this by subtracting the leading
		// family and len. The path is then scanned to see if a NUL
		// terminator still exists within the length.
		n := int(pp.Len) - 2 // subtract leading Family, Len
		for i := 0; i < n; i++ {
			if pp.Path[i] == 0 {
				// found early NUL; assume Len included the NUL
				// or was overestimating.
				n = i
				break
			}
		}
		sa.Name = string(unsafe.Slice((*byte)(unsafe.Pointer(&pp.Path[0])), n))
		return sa, nil

	case AF_INET:
		pp := (*RawSockaddrInet4)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet4)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		sa.Addr = pp.Addr
		return sa, nil

	case AF_INET6:
		pp := (*RawSockaddrInet6)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet6)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		sa.ZoneId = pp.Scope_id
		sa.Addr = pp.Addr
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
	if (runtime.GOOS == "darwin" || runtime.GOOS == "ios") && len == 0 {
		// Accepted socket has no address.
		// This is likely due to a bug in xnu kernels,
		// where instead of ECONNABORTED error socket
		// is accepted, but has no address.
		Close(nfd)
		return 0, nil, ECONNABORTED
	}
	sa, err = anyToSockaddr(&rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

func Getsockname(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if err = getsockname(fd, &rsa, &len); err != nil {
		return
	}
	// TODO(jsing): DragonFly has a "bug" (see issue 3349), which should be
	// reported upstream.
	if runtime.GOOS == "dragonfly" && rsa.Addr.Family == AF_UNSPEC && rsa.Addr.Len == 0 {
		rsa.Addr.Family = AF_UNIX
		rsa.Addr.Len = SizeofSockaddrUnix
	}
	return anyToSockaddr(&rsa)
}

//sysnb socketpair(domain int, typ int, proto int, fd *[2]int32) (err error)

func GetsockoptByte(fd, level, opt int) (value byte, err error) {
	var n byte
	vallen := _Socklen(1)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&n), &vallen)
	return n, err
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

//sys   recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, err error)
//sys   sendto(s int, buf []byte, flags int, to unsafe.Pointer, addrlen _Socklen) (err error)
//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, err error)

func recvmsgRaw(fd int, p, oob []byte, flags int, rsa *RawSockaddrAny) (n, oobn int, recvflags int, err error) {
	var msg Msghdr
	msg.Name = (*byte)(unsafe.Pointer(rsa))
	msg.Namelen = uint32(SizeofSockaddrAny)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = &p[0]
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		// receive at least one normal byte
		if len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = &oob[0]
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = recvmsg(fd, &msg, flags); err != nil {
		return
	}
	oobn = int(msg.Controllen)
	recvflags = int(msg.Flags)
	return
}

//sys	sendmsg(s int, msg *Msghdr, flags int) (n int, err error)

func sendmsgN(fd int, p, oob []byte, ptr unsafe.Pointer, salen _Socklen, flags int) (n int, err error) {
	var msg Msghdr
	msg.Name = (*byte)(ptr)
	msg.Namelen = uint32(salen)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = &p[0]
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		// send at least one normal byte
		if len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = &oob[0]
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

//sys	kevent(kq int, change unsafe.Pointer, nchange int, event unsafe.Pointer, nevent int, timeout *Timespec) (n int, err error)

func Kevent(kq int, changes, events []Kevent_t, timeout *Timespec) (n int, err error) {
	var change, event unsafe.Pointer
	if len(changes) > 0 {
		change = unsafe.Pointer(&changes[0])
	}
	if len(events) > 0 {
		event = unsafe.Pointer(&events[0])
	}
	return kevent(kq, change, len(changes), event, len(events), timeout)
}

func Sysctl(name string) (value string, err error) {
	// Translate name to mib number.
	mib, err := nametomib(name)
	if err != nil {
		return "", err
	}

	// Find size.
	n := uintptr(0)
	if err = sysctl(mib, nil, &n, nil, 0); err != nil {
		return "", err
	}
	if n == 0 {
		return "", nil
	}

	// Read into buffer of that size.
	buf := make([]byte, n)
	if err = sysctl(mib, &buf[0], &n, nil, 0); err != nil {
		return "", err
	}

	// Throw away terminating NUL.
	if n > 0 && buf[n-1] == '\x00' {
		n--
	}
	return string(buf[0:n]), nil
}

func SysctlUint32(name string) (value uint32, err error) {
	// Translate name to mib number.
	mib, err := nametomib(name)
	if err != nil {
		return 0, err
	}

	// Read into buffer of that size.
	n := uintptr(4)
	buf := make([]byte, 4)
	if err = sysctl(mib, &buf[0], &n, nil, 0); err != nil {
		return 0, err
	}
	if n != 4 {
		return 0, EIO
	}
	return *(*uint32)(unsafe.Pointer(&buf[0])), nil
}

//sys	utimes(path string, timeval *[2]Timeval) (err error)

func Utimes(path string, tv []Timeval) (err error) {
	if len(tv) != 2 {
		return EINVAL
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

func UtimesNano(path string, ts []Timespec) error {
	if len(ts) != 2 {
		return EINVAL
	}
	err := utimensat(_AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
	if err != ENOSYS {
		return err
	}
	// Not as efficient as it could be because Timespec and
	// Timeval have different types in the different OSes
	tv := [2]Timeval{
		NsecToTimeval(TimespecToNsec(ts[0])),
		NsecToTimeval(TimespecToNsec(ts[1])),
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	futimes(fd int, timeval *[2]Timeval) (err error)

func Futimes(fd int, tv []Timeval) (err error) {
	if len(tv) != 2 {
		return EINVAL
	}
	return futimes(fd, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	fcntl(fd int, cmd int, arg int) (val int, err error)
//sys	fcntlPtr(fd int, cmd int, arg unsafe.Pointer) (val int, err error) = SYS_FCNTL
//sysnb ioctl(fd int, req int, arg int) (err error)
//sysnb ioctlPtr(fd int, req uint, arg unsafe.Pointer) (err error) = SYS_IOCTL

var mapper = &mmapper{
	active: make(map[*byte][]byte),
	mmap:   mmap,
	munmap: munmap,
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return mapper.Munmap(b)
}
