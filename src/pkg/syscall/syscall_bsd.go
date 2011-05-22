// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// BSD system call wrappers shared by *BSD based systems
// including OS X (Darwin) and FreeBSD.  Like the other
// syscall_*.go files it is compiled as Go code but also
// used as input to mksyscall which parses the //sys
// lines and generates system call stubs.

package syscall

import "unsafe"

/*
 * Pseudo-system calls
 */
// The const provides a compile-time constant so clients
// can adjust to whether there is a working Getwd and avoid
// even linking this function into the binary.  See ../os/getwd.go.
const ImplementsGetwd = false

func Getwd() (string, int) { return "", ENOTSUP }


/*
 * Wrapped
 */

//sysnb	getgroups(ngid int, gid *_Gid_t) (n int, errno int)
//sysnb	setgroups(ngid int, gid *_Gid_t) (errno int)

func Getgroups() (gids []int, errno int) {
	n, err := getgroups(0, nil)
	if err != 0 {
		return nil, errno
	}
	if n == 0 {
		return nil, 0
	}

	// Sanity check group count.  Max is 16 on BSD.
	if n < 0 || n > 1000 {
		return nil, EINVAL
	}

	a := make([]_Gid_t, n)
	n, err = getgroups(n, &a[0])
	if err != 0 {
		return nil, errno
	}
	gids = make([]int, n)
	for i, v := range a[0:n] {
		gids[i] = int(v)
	}
	return
}

func Setgroups(gids []int) (errno int) {
	if len(gids) == 0 {
		return setgroups(0, nil)
	}

	a := make([]_Gid_t, len(gids))
	for i, v := range gids {
		a[i] = _Gid_t(v)
	}
	return setgroups(len(a), &a[0])
}

func ReadDirent(fd int, buf []byte) (n int, errno int) {
	// Final argument is (basep *uintptr) and the syscall doesn't take nil.
	// TODO(rsc): Can we use a single global basep for all calls?
	return Getdirentries(fd, buf, new(uintptr))
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

func (w WaitStatus) Signal() int {
	sig := int(w & mask)
	if sig == stopped || sig == 0 {
		return -1
	}
	return sig
}

func (w WaitStatus) CoreDump() bool { return w.Signaled() && w&core != 0 }

func (w WaitStatus) Stopped() bool { return w&mask == stopped && w>>shift != SIGSTOP }

func (w WaitStatus) Continued() bool { return w&mask == stopped && w>>shift == SIGSTOP }

func (w WaitStatus) StopSignal() int {
	if !w.Stopped() {
		return -1
	}
	return int(w>>shift) & 0xFF
}

func (w WaitStatus) TrapCause() int { return -1 }

//sys	wait4(pid int, wstatus *_C_int, options int, rusage *Rusage) (wpid int, errno int)

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, errno int) {
	var status _C_int
	wpid, errno = wait4(pid, &status, options, rusage)
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return
}

//sysnb	pipe() (r int, w int, errno int)

func Pipe(p []int) (errno int) {
	if len(p) != 2 {
		return EINVAL
	}
	p[0], p[1], errno = pipe()
	return
}

func Sleep(ns int64) (errno int) {
	tv := NsecToTimeval(ns)
	return Select(0, nil, nil, nil, &tv)
}

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int)
//sys	bind(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	connect(s int, addr uintptr, addrlen _Socklen) (errno int)
//sysnb	socket(domain int, typ int, proto int) (fd int, errno int)
//sys	getsockopt(s int, level int, name int, val uintptr, vallen *_Socklen) (errno int)
//sys	setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int)
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)
//sysnb	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)
//sys	Shutdown(s int, how int) (errno int)

// For testing: clients can set this flag to force
// creation of IPv6 sockets to return EAFNOSUPPORT.
var SocketDisableIPv6 bool

type Sockaddr interface {
	sockaddr() (ptr uintptr, len _Socklen, errno int) // lowercase; only we can define Sockaddrs
}

type SockaddrInet4 struct {
	Port int
	Addr [4]byte
	raw  RawSockaddrInet4
}

func (sa *SockaddrInet4) sockaddr() (uintptr, _Socklen, int) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return 0, 0, EINVAL
	}
	sa.raw.Len = SizeofSockaddrInet4
	sa.raw.Family = AF_INET
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return uintptr(unsafe.Pointer(&sa.raw)), _Socklen(sa.raw.Len), 0
}

type SockaddrInet6 struct {
	Port   int
	ZoneId uint32
	Addr   [16]byte
	raw    RawSockaddrInet6
}

func (sa *SockaddrInet6) sockaddr() (uintptr, _Socklen, int) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return 0, 0, EINVAL
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
	return uintptr(unsafe.Pointer(&sa.raw)), _Socklen(sa.raw.Len), 0
}

type SockaddrUnix struct {
	Name string
	raw  RawSockaddrUnix
}

func (sa *SockaddrUnix) sockaddr() (uintptr, _Socklen, int) {
	name := sa.Name
	n := len(name)
	if n >= len(sa.raw.Path) || n == 0 {
		return 0, 0, EINVAL
	}
	sa.raw.Len = byte(3 + n) // 2 for Family, Len; 1 for NUL
	sa.raw.Family = AF_UNIX
	for i := 0; i < n; i++ {
		sa.raw.Path[i] = int8(name[i])
	}
	return uintptr(unsafe.Pointer(&sa.raw)), _Socklen(sa.raw.Len), 0
}

func (sa *SockaddrDatalink) sockaddr() (uintptr, _Socklen, int) {
	if sa.Index == 0 {
		return 0, 0, EINVAL
	}
	sa.raw.Len = sa.Len
	sa.raw.Family = AF_LINK
	sa.raw.Index = sa.Index
	sa.raw.Type = sa.Type
	sa.raw.Nlen = sa.Nlen
	sa.raw.Alen = sa.Alen
	sa.raw.Slen = sa.Slen
	for i := 0; i < len(sa.raw.Data); i++ {
		sa.raw.Data[i] = sa.Data[i]
	}
	return uintptr(unsafe.Pointer(&sa.raw)), SizeofSockaddrDatalink, 0
}

func anyToSockaddr(rsa *RawSockaddrAny) (Sockaddr, int) {
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
		for i := 0; i < len(sa.Data); i++ {
			sa.Data[i] = pp.Data[i]
		}
		return sa, 0

	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa))
		if pp.Len < 3 || pp.Len > SizeofSockaddrUnix {
			return nil, EINVAL
		}
		sa := new(SockaddrUnix)
		n := int(pp.Len) - 3 // subtract leading Family, Len, terminating NUL
		for i := 0; i < n; i++ {
			if pp.Path[i] == 0 {
				// found early NUL; assume Len is overestimating
				n = i
				break
			}
		}
		bytes := (*[10000]byte)(unsafe.Pointer(&pp.Path[0]))[0:n]
		sa.Name = string(bytes)
		return sa, 0

	case AF_INET:
		pp := (*RawSockaddrInet4)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet4)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i]
		}
		return sa, 0

	case AF_INET6:
		pp := (*RawSockaddrInet6)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet6)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		sa.ZoneId = pp.Scope_id
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i]
		}
		return sa, 0
	}
	return nil, EAFNOSUPPORT
}

func Accept(fd int) (nfd int, sa Sockaddr, errno int) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, errno = accept(fd, &rsa, &len)
	if errno != 0 {
		return
	}
	sa, errno = anyToSockaddr(&rsa)
	if errno != 0 {
		Close(nfd)
		nfd = 0
	}
	return
}

func Getsockname(fd int) (sa Sockaddr, errno int) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if errno = getsockname(fd, &rsa, &len); errno != 0 {
		return
	}
	return anyToSockaddr(&rsa)
}

func Getpeername(fd int) (sa Sockaddr, errno int) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if errno = getpeername(fd, &rsa, &len); errno != 0 {
		return
	}
	return anyToSockaddr(&rsa)
}

func Bind(fd int, sa Sockaddr) (errno int) {
	ptr, n, err := sa.sockaddr()
	if err != 0 {
		return err
	}
	return bind(fd, ptr, n)
}

func Connect(fd int, sa Sockaddr) (errno int) {
	ptr, n, err := sa.sockaddr()
	if err != 0 {
		return err
	}
	return connect(fd, ptr, n)
}

func Socket(domain, typ, proto int) (fd, errno int) {
	if domain == AF_INET6 && SocketDisableIPv6 {
		return -1, EAFNOSUPPORT
	}
	fd, errno = socket(domain, typ, proto)
	return
}

//sysnb socketpair(domain int, typ int, proto int, fd *[2]int) (errno int)

func Socketpair(domain, typ, proto int) (fd [2]int, errno int) {
	errno = socketpair(domain, typ, proto, &fd)
	return
}

func GetsockoptInt(fd, level, opt int) (value, errno int) {
	var n int32
	vallen := _Socklen(4)
	errno = getsockopt(fd, level, opt, uintptr(unsafe.Pointer(&n)), &vallen)
	return int(n), errno
}

func SetsockoptInt(fd, level, opt int, value int) (errno int) {
	var n = int32(value)
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(&n)), 4)
}

func SetsockoptTimeval(fd, level, opt int, tv *Timeval) (errno int) {
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(tv)), unsafe.Sizeof(*tv))
}

func SetsockoptLinger(fd, level, opt int, l *Linger) (errno int) {
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(l)), unsafe.Sizeof(*l))
}

func SetsockoptIpMreq(fd, level, opt int, mreq *IpMreq) (errno int) {
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(mreq)), unsafe.Sizeof(*mreq))
}

func SetsockoptString(fd, level, opt int, s string) (errno int) {
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(&[]byte(s)[0])), len(s))
}

//sys recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, errno int)

func Recvfrom(fd int, p []byte, flags int) (n int, from Sockaddr, errno int) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if n, errno = recvfrom(fd, p, flags, &rsa, &len); errno != 0 {
		return
	}
	from, errno = anyToSockaddr(&rsa)
	return
}

//sys sendto(s int, buf []byte, flags int, to uintptr, addrlen _Socklen) (errno int)

func Sendto(fd int, p []byte, flags int, to Sockaddr) (errno int) {
	ptr, n, err := to.sockaddr()
	if err != 0 {
		return err
	}
	return sendto(fd, p, flags, ptr, n)
}

// TODO:
// FreeBSD has IP_SENDIF.  Darwin probably needs BSDLLCTest, see:
// http://developer.apple.com/mac/library/samplecode/BSDLLCTest/index.html

// BindToDevice binds the socket associated with fd to device.
func BindToDevice(fd int, device string) (errno int) {
	return ENOSYS
}

//sys	kevent(kq int, change uintptr, nchange int, event uintptr, nevent int, timeout *Timespec) (n int, errno int)

func Kevent(kq int, changes, events []Kevent_t, timeout *Timespec) (n int, errno int) {
	var change, event uintptr
	if len(changes) > 0 {
		change = uintptr(unsafe.Pointer(&changes[0]))
	}
	if len(events) > 0 {
		event = uintptr(unsafe.Pointer(&events[0]))
	}
	return kevent(kq, change, len(changes), event, len(events), timeout)
}

//sys	sysctl(mib []_C_int, old *byte, oldlen *uintptr, new *byte, newlen uintptr) (errno int) = SYS___SYSCTL

// Translate "kern.hostname" to []_C_int{0,1,2,3}.
func nametomib(name string) (mib []_C_int, errno int) {
	const siz = uintptr(unsafe.Sizeof(mib[0]))

	// NOTE(rsc): It seems strange to set the buffer to have
	// size CTL_MAXNAME+2 but use only CTL_MAXNAME
	// as the size.  I don't know why the +2 is here, but the
	// kernel uses +2 for its own implementation of this function.
	// I am scared that if we don't include the +2 here, the kernel
	// will silently write 2 words farther than we specify
	// and we'll get memory corruption.
	var buf [CTL_MAXNAME + 2]_C_int
	n := uintptr(CTL_MAXNAME) * siz

	p := (*byte)(unsafe.Pointer(&buf[0]))
	bytes := StringByteSlice(name)

	// Magic sysctl: "setting" 0.3 to a string name
	// lets you read back the array of integers form.
	if errno = sysctl([]_C_int{0, 3}, p, &n, &bytes[0], uintptr(len(name))); errno != 0 {
		return nil, errno
	}
	return buf[0 : n/siz], 0
}

func Sysctl(name string) (value string, errno int) {
	// Translate name to mib number.
	mib, errno := nametomib(name)
	if errno != 0 {
		return "", errno
	}

	// Find size.
	n := uintptr(0)
	if errno = sysctl(mib, nil, &n, nil, 0); errno != 0 {
		return "", errno
	}
	if n == 0 {
		return "", 0
	}

	// Read into buffer of that size.
	buf := make([]byte, n)
	if errno = sysctl(mib, &buf[0], &n, nil, 0); errno != 0 {
		return "", errno
	}

	// Throw away terminating NUL.
	if n > 0 && buf[n-1] == '\x00' {
		n--
	}
	return string(buf[0:n]), 0
}

func SysctlUint32(name string) (value uint32, errno int) {
	// Translate name to mib number.
	mib, errno := nametomib(name)
	if errno != 0 {
		return 0, errno
	}

	// Read into buffer of that size.
	n := uintptr(4)
	buf := make([]byte, 4)
	if errno = sysctl(mib, &buf[0], &n, nil, 0); errno != 0 {
		return 0, errno
	}
	if n != 4 {
		return 0, EIO
	}
	return *(*uint32)(unsafe.Pointer(&buf[0])), 0
}

func SysctlNetRoute(fourth, fifth, sixth int) (value []byte, errno int) {
	mib := []_C_int{CTL_NET, AF_ROUTE, 0, _C_int(fourth), _C_int(fifth), _C_int(sixth)}

	// Find size.
	n := uintptr(0)
	if errno = sysctl(mib, nil, &n, nil, 0); errno != 0 {
		return nil, errno
	}
	if n == 0 {
		return nil, 0
	}

	// Read into buffer of that size.
	b := make([]byte, n)
	if errno = sysctl(mib, &b[0], &n, nil, 0); errno != 0 {
		return nil, errno
	}

	return b[0:n], 0
}

//sys	utimes(path string, timeval *[2]Timeval) (errno int)
func Utimes(path string, tv []Timeval) (errno int) {
	if len(tv) != 2 {
		return EINVAL
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	futimes(fd int, timeval *[2]Timeval) (errno int)
func Futimes(fd int, tv []Timeval) (errno int) {
	if len(tv) != 2 {
		return EINVAL
	}
	return futimes(fd, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	fcntl(fd int, cmd int, arg int) (val int, errno int)

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn int, recvflags int, from Sockaddr, errno int) {
	return 0, 0, 0, nil, EAFNOSUPPORT
}

func Sendmsg(fd int, p, oob []byte, to Sockaddr, flags int) (errno int) {
	return EAFNOSUPPORT
}

// TODO: wrap
//	Acct(name nil-string) (errno int)
//	Gethostuuid(uuid *byte, timeout *Timespec) (errno int)
//	Madvise(addr *byte, len int, behav int) (errno int)
//	Mprotect(addr *byte, len int, prot int) (errno int)
//	Msync(addr *byte, len int, flags int) (errno int)
//	Ptrace(req int, pid int, addr uintptr, data int) (ret uintptr, errno int)

//sys	mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, errno int)
//sys	munmap(addr uintptr, length uintptr) (errno int)

var mapper = &mmapper{
	active: make(map[*byte][]byte),
	mmap:   mmap,
	munmap: munmap,
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, errno int) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (errno int) {
	return mapper.Munmap(b)
}
