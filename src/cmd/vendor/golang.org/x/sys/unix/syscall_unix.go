// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package unix

import (
	"bytes"
	"sort"
	"sync"
	"syscall"
	"unsafe"
)

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

var (
	signalNameMapOnce sync.Once
	signalNameMap     map[string]syscall.Signal
)

// errnoErr returns common boxed Errno values, to prevent
// allocations at runtime.
func errnoErr(e syscall.Errno) error {
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

// ErrnoName returns the error name for error number e.
func ErrnoName(e syscall.Errno) string {
	i := sort.Search(len(errorList), func(i int) bool {
		return errorList[i].num >= e
	})
	if i < len(errorList) && errorList[i].num == e {
		return errorList[i].name
	}
	return ""
}

// SignalName returns the signal name for signal number s.
func SignalName(s syscall.Signal) string {
	i := sort.Search(len(signalList), func(i int) bool {
		return signalList[i].num >= s
	})
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

	// Map the requested memory.
	addr, errno := m.mmap(0, uintptr(length), prot, flags, fd, offset)
	if errno != nil {
		return nil, errno
	}

	// Use unsafe to convert addr into a []byte.
	b := unsafe.Slice((*byte)(unsafe.Pointer(addr)), length)

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

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return mapper.Munmap(b)
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

func Pread(fd int, p []byte, offset int64) (n int, err error) {
	n, err = pread(fd, p, offset)
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

func Pwrite(fd int, p []byte, offset int64) (n int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = pwrite(fd, p, offset)
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

// Recvmsg receives a message from a socket using the recvmsg system call. The
// received non-control data will be written to p, and any "out of band"
// control data will be written to oob. The flags are passed to recvmsg.
//
// The results are:
//   - n is the number of non-control data bytes read into p
//   - oobn is the number of control data bytes read into oob; this may be interpreted using [ParseSocketControlMessage]
//   - recvflags is flags returned by recvmsg
//   - from is the address of the sender
//
// If the underlying socket type is not SOCK_DGRAM, a received message
// containing oob data and a single '\0' of non-control data is treated as if
// the message contained only control data, i.e. n will be zero on return.
func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn int, recvflags int, from Sockaddr, err error) {
	var iov [1]Iovec
	if len(p) > 0 {
		iov[0].Base = &p[0]
		iov[0].SetLen(len(p))
	}
	var rsa RawSockaddrAny
	n, oobn, recvflags, err = recvmsgRaw(fd, iov[:], oob, flags, &rsa)
	// source address is only specified if the socket is unconnected
	if rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(fd, &rsa)
	}
	return
}

// RecvmsgBuffers receives a message from a socket using the recvmsg system
// call. This function is equivalent to Recvmsg, but non-control data read is
// scattered into the buffers slices.
func RecvmsgBuffers(fd int, buffers [][]byte, oob []byte, flags int) (n, oobn int, recvflags int, from Sockaddr, err error) {
	iov := make([]Iovec, len(buffers))
	for i := range buffers {
		if len(buffers[i]) > 0 {
			iov[i].Base = &buffers[i][0]
			iov[i].SetLen(len(buffers[i]))
		} else {
			iov[i].Base = (*byte)(unsafe.Pointer(&_zero))
		}
	}
	var rsa RawSockaddrAny
	n, oobn, recvflags, err = recvmsgRaw(fd, iov, oob, flags, &rsa)
	if err == nil && rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(fd, &rsa)
	}
	return
}

// Sendmsg sends a message on a socket to an address using the sendmsg system
// call. This function is equivalent to SendmsgN, but does not return the
// number of bytes actually sent.
func Sendmsg(fd int, p, oob []byte, to Sockaddr, flags int) (err error) {
	_, err = SendmsgN(fd, p, oob, to, flags)
	return
}

// SendmsgN sends a message on a socket to an address using the sendmsg system
// call. p contains the non-control data to send, and oob contains the "out of
// band" control data. The flags are passed to sendmsg. The number of
// non-control bytes actually written to the socket is returned.
//
// Some socket types do not support sending control data without accompanying
// non-control data. If p is empty, and oob contains control data, and the
// underlying socket type is not SOCK_DGRAM, p will be treated as containing a
// single '\0' and the return value will indicate zero bytes sent.
//
// The Go function Recvmsg, if called with an empty p and a non-empty oob,
// will read and ignore this additional '\0'.  If the message is received by
// code that does not use Recvmsg, or that does not use Go at all, that code
// will need to be written to expect and ignore the additional '\0'.
//
// If you need to send non-empty oob with p actually empty, and if the
// underlying socket type supports it, you can do so via a raw system call as
// follows:
//
//	msg := &unix.Msghdr{
//	    Control: &oob[0],
//	}
//	msg.SetControllen(len(oob))
//	n, _, errno := unix.Syscall(unix.SYS_SENDMSG, uintptr(fd), uintptr(unsafe.Pointer(msg)), flags)
func SendmsgN(fd int, p, oob []byte, to Sockaddr, flags int) (n int, err error) {
	var iov [1]Iovec
	if len(p) > 0 {
		iov[0].Base = &p[0]
		iov[0].SetLen(len(p))
	}
	var ptr unsafe.Pointer
	var salen _Socklen
	if to != nil {
		ptr, salen, err = to.sockaddr()
		if err != nil {
			return 0, err
		}
	}
	return sendmsgN(fd, iov[:], oob, ptr, salen, flags)
}

// SendmsgBuffers sends a message on a socket to an address using the sendmsg
// system call. This function is equivalent to SendmsgN, but the non-control
// data is gathered from buffers.
func SendmsgBuffers(fd int, buffers [][]byte, oob []byte, to Sockaddr, flags int) (n int, err error) {
	iov := make([]Iovec, len(buffers))
	for i := range buffers {
		if len(buffers[i]) > 0 {
			iov[i].Base = &buffers[i][0]
			iov[i].SetLen(len(buffers[i]))
		} else {
			iov[i].Base = (*byte)(unsafe.Pointer(&_zero))
		}
	}
	var ptr unsafe.Pointer
	var salen _Socklen
	if to != nil {
		ptr, salen, err = to.sockaddr()
		if err != nil {
			return 0, err
		}
	}
	return sendmsgN(fd, iov, oob, ptr, salen, flags)
}

func Send(s int, buf []byte, flags int) (err error) {
	return sendto(s, buf, flags, nil, 0)
}

func Sendto(fd int, p []byte, flags int, to Sockaddr) (err error) {
	var ptr unsafe.Pointer
	var salen _Socklen
	if to != nil {
		ptr, salen, err = to.sockaddr()
		if err != nil {
			return err
		}
	}
	return sendto(fd, p, flags, ptr, salen)
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
	if (flag&O_NONBLOCK != 0) == nonblocking {
		return nil
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

// Lutimes sets the access and modification times tv on path. If path refers to
// a symlink, it is not dereferenced and the timestamps are set on the symlink.
// If tv is nil, the access and modification times are set to the current time.
// Otherwise tv must contain exactly 2 elements, with access time as the first
// element and modification time as the second element.
func Lutimes(path string, tv []Timeval) error {
	if tv == nil {
		return UtimesNanoAt(AT_FDCWD, path, nil, AT_SYMLINK_NOFOLLOW)
	}
	if len(tv) != 2 {
		return EINVAL
	}
	ts := []Timespec{
		NsecToTimespec(TimevalToNsec(tv[0])),
		NsecToTimespec(TimevalToNsec(tv[1])),
	}
	return UtimesNanoAt(AT_FDCWD, path, ts, AT_SYMLINK_NOFOLLOW)
}

// emptyIovecs reports whether there are no bytes in the slice of Iovec.
func emptyIovecs(iov []Iovec) bool {
	for i := range iov {
		if iov[i].Len > 0 {
			return false
		}
	}
	return true
}

// Setrlimit sets a resource limit.
func Setrlimit(resource int, rlim *Rlimit) error {
	// Just call the syscall version, because as of Go 1.21
	// it will affect starting a new process.
	return syscall.Setrlimit(resource, (*syscall.Rlimit)(rlim))
}
