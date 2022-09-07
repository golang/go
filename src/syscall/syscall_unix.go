// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall

import (
	"internal/bytealg"
	"internal/itoa"
	"internal/oserror"
	"internal/race"
	"runtime"
	"sync"
	"unsafe"
)

var (
	Stdin  = 0
	Stdout = 1
	Stderr = 2
)

const (
	darwin64Bit = (runtime.GOOS == "darwin" || runtime.GOOS == "ios") && sizeofPtr == 8
	netbsd32Bit = runtime.GOOS == "netbsd" && sizeofPtr == 4
)

// clen returns the index of the first NULL byte in n or len(n) if n contains no NULL byte.
func clen(n []byte) int {
	if i := bytealg.IndexByte(n, 0); i != -1 {
		return i
	}
	return len(n)
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

	// Use unsafe to turn addr into a []byte.
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

// An Errno is an unsigned number describing an error condition.
// It implements the error interface. The zero Errno is by convention
// a non-error, so code to convert from Errno to error should use:
//
//	err = nil
//	if errno != 0 {
//		err = errno
//	}
//
// Errno values can be tested against error values from the os package
// using errors.Is. For example:
//
//	_, _, err := syscall.Syscall(...)
//	if errors.Is(err, fs.ErrNotExist) ...
type Errno uintptr

func (e Errno) Error() string {
	if 0 <= int(e) && int(e) < len(errors) {
		s := errors[e]
		if s != "" {
			return s
		}
	}
	return "errno " + itoa.Itoa(int(e))
}

func (e Errno) Is(target error) bool {
	switch target {
	case oserror.ErrPermission:
		return e == EACCES || e == EPERM
	case oserror.ErrExist:
		return e == EEXIST || e == ENOTEMPTY
	case oserror.ErrNotExist:
		return e == ENOENT
	}
	return false
}

func (e Errno) Temporary() bool {
	return e == EINTR || e == EMFILE || e == ENFILE || e.Timeout()
}

func (e Errno) Timeout() bool {
	return e == EAGAIN || e == EWOULDBLOCK || e == ETIMEDOUT
}

// Do the interface allocations only once for common
// Errno values.
var (
	errEAGAIN error = EAGAIN
	errEINVAL error = EINVAL
	errENOENT error = ENOENT
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

// A Signal is a number describing a process signal.
// It implements the os.Signal interface.
type Signal int

func (s Signal) Signal() {}

func (s Signal) String() string {
	if 0 <= s && int(s) < len(signals) {
		str := signals[s]
		if str != "" {
			return str
		}
	}
	return "signal " + itoa.Itoa(int(s))
}

func Read(fd int, p []byte) (n int, err error) {
	n, err = read(fd, p)
	if race.Enabled {
		if n > 0 {
			race.WriteRange(unsafe.Pointer(&p[0]), n)
		}
		if err == nil {
			race.Acquire(unsafe.Pointer(&ioSync))
		}
	}
	if msanenabled && n > 0 {
		msanWrite(unsafe.Pointer(&p[0]), n)
	}
	if asanenabled && n > 0 {
		asanWrite(unsafe.Pointer(&p[0]), n)
	}
	return
}

func Write(fd int, p []byte) (n int, err error) {
	if race.Enabled {
		race.ReleaseMerge(unsafe.Pointer(&ioSync))
	}
	if faketime && (fd == 1 || fd == 2) {
		n = faketimeWrite(fd, p)
		if n < 0 {
			n, err = 0, errnoErr(Errno(-n))
		}
	} else {
		n, err = write(fd, p)
	}
	if race.Enabled && n > 0 {
		race.ReadRange(unsafe.Pointer(&p[0]), n)
	}
	if msanenabled && n > 0 {
		msanRead(unsafe.Pointer(&p[0]), n)
	}
	if asanenabled && n > 0 {
		asanRead(unsafe.Pointer(&p[0]), n)
	}
	return
}

func Pread(fd int, p []byte, offset int64) (n int, err error) {
	n, err = pread(fd, p, offset)
	if race.Enabled {
		if n > 0 {
			race.WriteRange(unsafe.Pointer(&p[0]), n)
		}
		if err == nil {
			race.Acquire(unsafe.Pointer(&ioSync))
		}
	}
	if msanenabled && n > 0 {
		msanWrite(unsafe.Pointer(&p[0]), n)
	}
	if asanenabled && n > 0 {
		asanWrite(unsafe.Pointer(&p[0]), n)
	}
	return
}

func Pwrite(fd int, p []byte, offset int64) (n int, err error) {
	if race.Enabled {
		race.ReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = pwrite(fd, p, offset)
	if race.Enabled && n > 0 {
		race.ReadRange(unsafe.Pointer(&p[0]), n)
	}
	if msanenabled && n > 0 {
		msanRead(unsafe.Pointer(&p[0]), n)
	}
	if asanenabled && n > 0 {
		asanRead(unsafe.Pointer(&p[0]), n)
	}
	return
}

// For testing: clients can set this flag to force
// creation of IPv6 sockets to return EAFNOSUPPORT.
var SocketDisableIPv6 bool

type Sockaddr interface {
	sockaddr() (ptr unsafe.Pointer, len _Socklen, err error) // lowercase; only we can define Sockaddrs
}

type SockaddrInet4 struct {
	Port int
	Addr [4]byte
	raw  RawSockaddrInet4
}

type SockaddrInet6 struct {
	Port   int
	ZoneId uint32
	Addr   [16]byte
	raw    RawSockaddrInet6
}

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
	return anyToSockaddr(&rsa)
}

func GetsockoptInt(fd, level, opt int) (value int, err error) {
	var n int32
	vallen := _Socklen(4)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&n), &vallen)
	return int(n), err
}

func Recvfrom(fd int, p []byte, flags int) (n int, from Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if n, err = recvfrom(fd, p, flags, &rsa, &len); err != nil {
		return
	}
	if rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(&rsa)
	}
	return
}

func recvfromInet4(fd int, p []byte, flags int, from *SockaddrInet4) (n int, err error) {
	var rsa RawSockaddrAny
	var socklen _Socklen = SizeofSockaddrAny
	if n, err = recvfrom(fd, p, flags, &rsa, &socklen); err != nil {
		return
	}
	pp := (*RawSockaddrInet4)(unsafe.Pointer(&rsa))
	port := (*[2]byte)(unsafe.Pointer(&pp.Port))
	from.Port = int(port[0])<<8 + int(port[1])
	from.Addr = pp.Addr
	return
}

func recvfromInet6(fd int, p []byte, flags int, from *SockaddrInet6) (n int, err error) {
	var rsa RawSockaddrAny
	var socklen _Socklen = SizeofSockaddrAny
	if n, err = recvfrom(fd, p, flags, &rsa, &socklen); err != nil {
		return
	}
	pp := (*RawSockaddrInet6)(unsafe.Pointer(&rsa))
	port := (*[2]byte)(unsafe.Pointer(&pp.Port))
	from.Port = int(port[0])<<8 + int(port[1])
	from.ZoneId = pp.Scope_id
	from.Addr = pp.Addr
	return
}

func recvmsgInet4(fd int, p, oob []byte, flags int, from *SockaddrInet4) (n, oobn int, recvflags int, err error) {
	var rsa RawSockaddrAny
	n, oobn, recvflags, err = recvmsgRaw(fd, p, oob, flags, &rsa)
	if err != nil {
		return
	}
	pp := (*RawSockaddrInet4)(unsafe.Pointer(&rsa))
	port := (*[2]byte)(unsafe.Pointer(&pp.Port))
	from.Port = int(port[0])<<8 + int(port[1])
	from.Addr = pp.Addr
	return
}

func recvmsgInet6(fd int, p, oob []byte, flags int, from *SockaddrInet6) (n, oobn int, recvflags int, err error) {
	var rsa RawSockaddrAny
	n, oobn, recvflags, err = recvmsgRaw(fd, p, oob, flags, &rsa)
	if err != nil {
		return
	}
	pp := (*RawSockaddrInet6)(unsafe.Pointer(&rsa))
	port := (*[2]byte)(unsafe.Pointer(&pp.Port))
	from.Port = int(port[0])<<8 + int(port[1])
	from.ZoneId = pp.Scope_id
	from.Addr = pp.Addr
	return
}

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn int, recvflags int, from Sockaddr, err error) {
	var rsa RawSockaddrAny
	n, oobn, recvflags, err = recvmsgRaw(fd, p, oob, flags, &rsa)
	// source address is only specified if the socket is unconnected
	if rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(&rsa)
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
		ptr, salen, err = to.sockaddr()
		if err != nil {
			return 0, err
		}
	}
	return sendmsgN(fd, p, oob, ptr, salen, flags)
}

func sendmsgNInet4(fd int, p, oob []byte, to *SockaddrInet4, flags int) (n int, err error) {
	ptr, salen, err := to.sockaddr()
	if err != nil {
		return 0, err
	}
	return sendmsgN(fd, p, oob, ptr, salen, flags)
}

func sendmsgNInet6(fd int, p, oob []byte, to *SockaddrInet6, flags int) (n int, err error) {
	ptr, salen, err := to.sockaddr()
	if err != nil {
		return 0, err
	}
	return sendmsgN(fd, p, oob, ptr, salen, flags)
}

func sendtoInet4(fd int, p []byte, flags int, to *SockaddrInet4) (err error) {
	ptr, n, err := to.sockaddr()
	if err != nil {
		return err
	}
	return sendto(fd, p, flags, ptr, n)
}

func sendtoInet6(fd int, p []byte, flags int, to *SockaddrInet6) (err error) {
	ptr, n, err := to.sockaddr()
	if err != nil {
		return err
	}
	return sendto(fd, p, flags, ptr, n)
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

func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	if race.Enabled {
		race.ReleaseMerge(unsafe.Pointer(&ioSync))
	}
	return sendfile(outfd, infd, offset, count)
}

var ioSync int64
