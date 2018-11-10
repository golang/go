// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A simulated network for use within NaCl.
// The simulation is not particularly tied to NaCl,
// but other systems have real networks.

// All int64 times are UnixNanos.

package syscall

import (
	"sync"
	"sync/atomic"
)

// Interface to timers implemented in package runtime.
// Must be in sync with ../runtime/runtime.h:/^struct.Timer$
// Really for use by package time, but we cannot import time here.

type runtimeTimer struct {
	i      int
	when   int64
	period int64
	f      func(interface{}, uintptr) // NOTE: must not be closure
	arg    interface{}
	seq    uintptr
}

func startTimer(*runtimeTimer)
func stopTimer(*runtimeTimer) bool

type timer struct {
	expired bool
	q       *queue
	r       runtimeTimer
}

func (t *timer) start(q *queue, deadline int64) {
	if deadline == 0 {
		return
	}
	t.q = q
	t.r.when = deadline
	t.r.f = timerExpired
	t.r.arg = t
	startTimer(&t.r)
}

func (t *timer) stop() {
	stopTimer(&t.r)
}

func (t *timer) reset(q *queue, deadline int64) {
	if t.r.f != nil {
		t.stop()
	}
	if deadline == 0 {
		return
	}
	if t.r.f == nil {
		t.q = q
		t.r.f = timerExpired
		t.r.arg = t
	}
	t.r.when = deadline
	startTimer(&t.r)
}

func timerExpired(i interface{}, seq uintptr) {
	t := i.(*timer)
	go func() {
		t.q.Lock()
		defer t.q.Unlock()
		t.expired = true
		t.q.canRead.Broadcast()
		t.q.canWrite.Broadcast()
	}()
}

// Network constants and data structures. These match the traditional values.

const (
	AF_UNSPEC = iota
	AF_UNIX
	AF_INET
	AF_INET6
)

const (
	SHUT_RD = iota
	SHUT_WR
	SHUT_RDWR
)

const (
	SOCK_STREAM = 1 + iota
	SOCK_DGRAM
	SOCK_RAW
	SOCK_SEQPACKET
)

const (
	IPPROTO_IP   = 0
	IPPROTO_IPV4 = 4
	IPPROTO_IPV6 = 0x29
	IPPROTO_TCP  = 6
	IPPROTO_UDP  = 0x11
)

// Misc constants expected by package net but not supported.
const (
	_ = iota
	SOL_SOCKET
	SO_TYPE
	NET_RT_IFLIST
	IFNAMSIZ
	IFF_UP
	IFF_BROADCAST
	IFF_LOOPBACK
	IFF_POINTOPOINT
	IFF_MULTICAST
	IPV6_V6ONLY
	SOMAXCONN
	F_DUPFD_CLOEXEC
	SO_BROADCAST
	SO_REUSEADDR
	SO_REUSEPORT
	SO_RCVBUF
	SO_SNDBUF
	SO_KEEPALIVE
	SO_LINGER
	SO_ERROR
	IP_PORTRANGE
	IP_PORTRANGE_DEFAULT
	IP_PORTRANGE_LOW
	IP_PORTRANGE_HIGH
	IP_MULTICAST_IF
	IP_MULTICAST_LOOP
	IP_ADD_MEMBERSHIP
	IPV6_PORTRANGE
	IPV6_PORTRANGE_DEFAULT
	IPV6_PORTRANGE_LOW
	IPV6_PORTRANGE_HIGH
	IPV6_MULTICAST_IF
	IPV6_MULTICAST_LOOP
	IPV6_JOIN_GROUP
	TCP_NODELAY
	TCP_KEEPINTVL
	TCP_KEEPIDLE

	SYS_FCNTL = 500 // unsupported
)

var SocketDisableIPv6 bool

// A Sockaddr is one of the SockaddrXxx structs.
type Sockaddr interface {
	// copy returns a copy of the underlying data.
	copy() Sockaddr

	// key returns the value of the underlying data,
	// for comparison as a map key.
	key() interface{}
}

type SockaddrInet4 struct {
	Port int
	Addr [4]byte
}

func (sa *SockaddrInet4) copy() Sockaddr {
	sa1 := *sa
	return &sa1
}

func (sa *SockaddrInet4) key() interface{} { return *sa }

func isIPv4Localhost(sa Sockaddr) bool {
	sa4, ok := sa.(*SockaddrInet4)
	return ok && sa4.Addr == [4]byte{127, 0, 0, 1}
}

type SockaddrInet6 struct {
	Port   int
	ZoneId uint32
	Addr   [16]byte
}

func (sa *SockaddrInet6) copy() Sockaddr {
	sa1 := *sa
	return &sa1
}

func (sa *SockaddrInet6) key() interface{} { return *sa }

type SockaddrUnix struct {
	Name string
}

func (sa *SockaddrUnix) copy() Sockaddr {
	sa1 := *sa
	return &sa1
}

func (sa *SockaddrUnix) key() interface{} { return *sa }

type SockaddrDatalink struct {
	Len    uint8
	Family uint8
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [12]int8
}

func (sa *SockaddrDatalink) copy() Sockaddr {
	sa1 := *sa
	return &sa1
}

func (sa *SockaddrDatalink) key() interface{} { return *sa }

// RoutingMessage represents a routing message.
type RoutingMessage interface {
	unimplemented()
}

type IPMreq struct {
	Multiaddr [4]byte /* in_addr */
	Interface [4]byte /* in_addr */
}

type IPv6Mreq struct {
	Multiaddr [16]byte /* in6_addr */
	Interface uint32
}

type Linger struct {
	Onoff  int32
	Linger int32
}

type ICMPv6Filter struct {
	Filt [8]uint32
}

// A queue is the bookkeeping for a synchronized buffered queue.
// We do not use channels because we need to be able to handle
// writes after and during close, and because a chan byte would
// require too many send and receive operations in real use.
type queue struct {
	sync.Mutex
	canRead  sync.Cond
	canWrite sync.Cond
	rtimer   *timer // non-nil if in read
	wtimer   *timer // non-nil if in write
	r        int    // total read index
	w        int    // total write index
	m        int    // index mask
	closed   bool
}

func (q *queue) init(size int) {
	if size&(size-1) != 0 {
		panic("invalid queue size - must be power of two")
	}
	q.canRead.L = &q.Mutex
	q.canWrite.L = &q.Mutex
	q.m = size - 1
}

func past(deadline int64) bool {
	sec, nsec := now()
	return deadline > 0 && deadline < sec*1e9+int64(nsec)
}

func (q *queue) waitRead(n int, deadline int64) (int, error) {
	if past(deadline) {
		return 0, EAGAIN
	}
	var t timer
	t.start(q, deadline)
	q.rtimer = &t
	for q.w-q.r == 0 && !q.closed && !t.expired {
		q.canRead.Wait()
	}
	q.rtimer = nil
	t.stop()
	m := q.w - q.r
	if m == 0 && t.expired {
		return 0, EAGAIN
	}
	if m > n {
		m = n
		q.canRead.Signal() // wake up next reader too
	}
	q.canWrite.Signal()
	return m, nil
}

func (q *queue) waitWrite(n int, deadline int64) (int, error) {
	if past(deadline) {
		return 0, EAGAIN
	}
	var t timer
	t.start(q, deadline)
	q.wtimer = &t
	for q.w-q.r > q.m && !q.closed && !t.expired {
		q.canWrite.Wait()
	}
	q.wtimer = nil
	t.stop()
	m := q.m + 1 - (q.w - q.r)
	if m == 0 && t.expired {
		return 0, EAGAIN
	}
	if m == 0 {
		return 0, EAGAIN
	}
	if m > n {
		m = n
		q.canWrite.Signal() // wake up next writer too
	}
	q.canRead.Signal()
	return m, nil
}

func (q *queue) close() {
	q.Lock()
	defer q.Unlock()
	q.closed = true
	q.canRead.Broadcast()
	q.canWrite.Broadcast()
}

// A byteq is a byte queue.
type byteq struct {
	queue
	data []byte
}

func newByteq() *byteq {
	q := &byteq{
		data: make([]byte, 4096),
	}
	q.init(len(q.data))
	return q
}

func (q *byteq) read(b []byte, deadline int64) (int, error) {
	q.Lock()
	defer q.Unlock()
	n, err := q.waitRead(len(b), deadline)
	if err != nil {
		return 0, err
	}
	b = b[:n]
	for len(b) > 0 {
		m := copy(b, q.data[q.r&q.m:])
		q.r += m
		b = b[m:]
	}
	return n, nil
}

func (q *byteq) write(b []byte, deadline int64) (n int, err error) {
	q.Lock()
	defer q.Unlock()
	for n < len(b) {
		nn, err := q.waitWrite(len(b[n:]), deadline)
		if err != nil {
			return n, err
		}
		bb := b[n : n+nn]
		n += nn
		for len(bb) > 0 {
			m := copy(q.data[q.w&q.m:], bb)
			q.w += m
			bb = bb[m:]
		}
	}
	return n, nil
}

// A msgq is a queue of messages.
type msgq struct {
	queue
	data []interface{}
}

func newMsgq() *msgq {
	q := &msgq{
		data: make([]interface{}, 32),
	}
	q.init(len(q.data))
	return q
}

func (q *msgq) read(deadline int64) (interface{}, error) {
	q.Lock()
	defer q.Unlock()
	n, err := q.waitRead(1, deadline)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	m := q.data[q.r&q.m]
	q.r++
	return m, nil
}

func (q *msgq) write(m interface{}, deadline int64) error {
	q.Lock()
	defer q.Unlock()
	_, err := q.waitWrite(1, deadline)
	if err != nil {
		return err
	}
	q.data[q.w&q.m] = m
	q.w++
	return nil
}

// An addr is a sequence of bytes uniquely identifying a network address.
// It is not human-readable.
type addr string

// A conn is one side of a stream-based network connection.
// That is, a stream-based network connection is a pair of cross-connected conns.
type conn struct {
	rd     *byteq
	wr     *byteq
	local  addr
	remote addr
}

// A pktconn is one side of a packet-based network connection.
// That is, a packet-based network connection is a pair of cross-connected pktconns.
type pktconn struct {
	rd     *msgq
	wr     *msgq
	local  addr
	remote addr
}

// A listener accepts incoming stream-based network connections.
type listener struct {
	rd    *msgq
	local addr
}

// A netFile is an open network file.
type netFile struct {
	defaultFileImpl
	proto      *netproto
	sotype     int
	listener   *msgq
	packet     *msgq
	rd         *byteq
	wr         *byteq
	rddeadline int64
	wrdeadline int64
	addr       Sockaddr
	raddr      Sockaddr
}

// A netAddr is a network address in the global listener map.
// All the fields must have defined == operations.
type netAddr struct {
	proto  *netproto
	sotype int
	addr   interface{}
}

// net records the state of the network.
// It maps a network address to the listener on that address.
var net = struct {
	sync.Mutex
	listener map[netAddr]*netFile
}{
	listener: make(map[netAddr]*netFile),
}

// TODO(rsc): Some day, do a better job with port allocation.
// For playground programs, incrementing is fine.
var nextport = 2

// A netproto contains protocol-specific functionality
// (one for AF_INET, one for AF_INET6 and so on).
// It is a struct instead of an interface because the
// implementation needs no state, and I expect to
// add some data fields at some point.
type netproto struct {
	bind func(*netFile, Sockaddr) error
}

var netprotoAF_INET = &netproto{
	bind: func(f *netFile, sa Sockaddr) error {
		if sa == nil {
			f.addr = &SockaddrInet4{
				Port: nextport,
				Addr: [4]byte{127, 0, 0, 1},
			}
			nextport++
			return nil
		}
		addr, ok := sa.(*SockaddrInet4)
		if !ok {
			return EINVAL
		}
		addr = addr.copy().(*SockaddrInet4)
		if addr.Port == 0 {
			addr.Port = nextport
			nextport++
		}
		f.addr = addr
		return nil
	},
}

var netprotos = map[int]*netproto{
	AF_INET: netprotoAF_INET,
}

// These functions implement the usual BSD socket operations.

func (f *netFile) bind(sa Sockaddr) error {
	if f.addr != nil {
		return EISCONN
	}
	if err := f.proto.bind(f, sa); err != nil {
		return err
	}
	if f.sotype == SOCK_DGRAM {
		_, ok := net.listener[netAddr{f.proto, f.sotype, f.addr.key()}]
		if ok {
			f.addr = nil
			return EADDRINUSE
		}
		net.listener[netAddr{f.proto, f.sotype, f.addr.key()}] = f
		f.packet = newMsgq()
	}
	return nil
}

func (f *netFile) listen(backlog int) error {
	net.Lock()
	defer net.Unlock()
	if f.listener != nil {
		return EINVAL
	}
	old, ok := net.listener[netAddr{f.proto, f.sotype, f.addr.key()}]
	if ok && !old.listenerClosed() {
		return EADDRINUSE
	}
	net.listener[netAddr{f.proto, f.sotype, f.addr.key()}] = f
	f.listener = newMsgq()
	return nil
}

func (f *netFile) accept() (fd int, sa Sockaddr, err error) {
	msg, err := f.listener.read(f.readDeadline())
	if err != nil {
		return -1, nil, err
	}
	newf, ok := msg.(*netFile)
	if !ok {
		// must be eof
		return -1, nil, EAGAIN
	}
	return newFD(newf), newf.raddr.copy(), nil
}

func (f *netFile) connect(sa Sockaddr) error {
	if past(f.writeDeadline()) {
		return EAGAIN
	}
	if f.addr == nil {
		if err := f.bind(nil); err != nil {
			return err
		}
	}
	net.Lock()
	if sa == nil {
		net.Unlock()
		return EINVAL
	}
	sa = sa.copy()
	if f.raddr != nil {
		net.Unlock()
		return EISCONN
	}
	if f.sotype == SOCK_DGRAM {
		net.Unlock()
		f.raddr = sa
		return nil
	}
	if f.listener != nil {
		net.Unlock()
		return EISCONN
	}
	l, ok := net.listener[netAddr{f.proto, f.sotype, sa.key()}]
	if !ok {
		// If we're dialing 127.0.0.1 but found nothing, try
		// 0.0.0.0 also. (Issue 20611)
		if isIPv4Localhost(sa) {
			sa = &SockaddrInet4{Port: sa.(*SockaddrInet4).Port}
			l, ok = net.listener[netAddr{f.proto, f.sotype, sa.key()}]
		}
	}
	if !ok || l.listenerClosed() {
		net.Unlock()
		return ECONNREFUSED
	}
	f.raddr = sa
	f.rd = newByteq()
	f.wr = newByteq()
	newf := &netFile{
		proto:  f.proto,
		sotype: f.sotype,
		addr:   f.raddr,
		raddr:  f.addr,
		rd:     f.wr,
		wr:     f.rd,
	}
	net.Unlock()
	l.listener.write(newf, f.writeDeadline())
	return nil
}

func (f *netFile) read(b []byte) (int, error) {
	if f.rd == nil {
		if f.raddr != nil {
			n, _, err := f.recvfrom(b, 0)
			return n, err
		}
		return 0, ENOTCONN
	}
	return f.rd.read(b, f.readDeadline())
}

func (f *netFile) write(b []byte) (int, error) {
	if f.wr == nil {
		if f.raddr != nil {
			err := f.sendto(b, 0, f.raddr)
			var n int
			if err == nil {
				n = len(b)
			}
			return n, err
		}
		return 0, ENOTCONN
	}
	return f.wr.write(b, f.writeDeadline())
}

type pktmsg struct {
	buf  []byte
	addr Sockaddr
}

func (f *netFile) recvfrom(p []byte, flags int) (n int, from Sockaddr, err error) {
	if f.sotype != SOCK_DGRAM {
		return 0, nil, EINVAL
	}
	if f.packet == nil {
		return 0, nil, ENOTCONN
	}
	msg1, err := f.packet.read(f.readDeadline())
	if err != nil {
		return 0, nil, err
	}
	msg, ok := msg1.(*pktmsg)
	if !ok {
		return 0, nil, EAGAIN
	}
	return copy(p, msg.buf), msg.addr, nil
}

func (f *netFile) sendto(p []byte, flags int, to Sockaddr) error {
	if f.sotype != SOCK_DGRAM {
		return EINVAL
	}
	if f.packet == nil {
		if err := f.bind(nil); err != nil {
			return err
		}
	}
	net.Lock()
	if to == nil {
		net.Unlock()
		return EINVAL
	}
	to = to.copy()
	l, ok := net.listener[netAddr{f.proto, f.sotype, to.key()}]
	if !ok || l.packet == nil {
		net.Unlock()
		return ECONNREFUSED
	}
	net.Unlock()
	msg := &pktmsg{
		buf:  make([]byte, len(p)),
		addr: f.addr,
	}
	copy(msg.buf, p)
	l.packet.write(msg, f.writeDeadline())
	return nil
}

func (f *netFile) listenerClosed() bool {
	f.listener.Lock()
	defer f.listener.Unlock()
	return f.listener.closed
}

func (f *netFile) close() error {
	if f.listener != nil {
		f.listener.close()
	}
	if f.packet != nil {
		f.packet.close()
	}
	if f.rd != nil {
		f.rd.close()
	}
	if f.wr != nil {
		f.wr.close()
	}
	return nil
}

func fdToNetFile(fd int) (*netFile, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return nil, err
	}
	impl := f.impl
	netf, ok := impl.(*netFile)
	if !ok {
		return nil, EINVAL
	}
	return netf, nil
}

func Socket(proto, sotype, unused int) (fd int, err error) {
	p := netprotos[proto]
	if p == nil {
		return -1, EPROTONOSUPPORT
	}
	if sotype != SOCK_STREAM && sotype != SOCK_DGRAM {
		return -1, ESOCKTNOSUPPORT
	}
	f := &netFile{
		proto:  p,
		sotype: sotype,
	}
	return newFD(f), nil
}

func Bind(fd int, sa Sockaddr) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	return f.bind(sa)
}

func StopIO(fd int) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	f.close()
	return nil
}

func Listen(fd int, backlog int) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	return f.listen(backlog)
}

func Accept(fd int) (newfd int, sa Sockaddr, err error) {
	f, err := fdToNetFile(fd)
	if err != nil {
		return 0, nil, err
	}
	return f.accept()
}

func Getsockname(fd int) (sa Sockaddr, err error) {
	f, err := fdToNetFile(fd)
	if err != nil {
		return nil, err
	}
	if f.addr == nil {
		return nil, ENOTCONN
	}
	return f.addr.copy(), nil
}

func Getpeername(fd int) (sa Sockaddr, err error) {
	f, err := fdToNetFile(fd)
	if err != nil {
		return nil, err
	}
	if f.raddr == nil {
		return nil, ENOTCONN
	}
	return f.raddr.copy(), nil
}

func Connect(fd int, sa Sockaddr) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	return f.connect(sa)
}

func Recvfrom(fd int, p []byte, flags int) (n int, from Sockaddr, err error) {
	f, err := fdToNetFile(fd)
	if err != nil {
		return 0, nil, err
	}
	return f.recvfrom(p, flags)
}

func Sendto(fd int, p []byte, flags int, to Sockaddr) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	return f.sendto(p, flags, to)
}

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn, recvflags int, from Sockaddr, err error) {
	f, err := fdToNetFile(fd)
	if err != nil {
		return
	}
	n, from, err = f.recvfrom(p, flags)
	return
}

func Sendmsg(fd int, p, oob []byte, to Sockaddr, flags int) error {
	_, err := SendmsgN(fd, p, oob, to, flags)
	return err
}

func SendmsgN(fd int, p, oob []byte, to Sockaddr, flags int) (n int, err error) {
	f, err := fdToNetFile(fd)
	if err != nil {
		return 0, err
	}
	switch f.sotype {
	case SOCK_STREAM:
		n, err = f.write(p)
	case SOCK_DGRAM:
		n = len(p)
		err = f.sendto(p, flags, to)
	}
	if err != nil {
		return 0, err
	}
	return n, nil
}

func GetsockoptInt(fd, level, opt int) (value int, err error) {
	f, err := fdToNetFile(fd)
	if err != nil {
		return 0, err
	}
	switch {
	case level == SOL_SOCKET && opt == SO_TYPE:
		return f.sotype, nil
	}
	return 0, ENOTSUP
}

func SetsockoptInt(fd, level, opt int, value int) error {
	return nil
}

func SetsockoptByte(fd, level, opt int, value byte) error {
	_, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	return ENOTSUP
}

func SetsockoptLinger(fd, level, opt int, l *Linger) error {
	return nil
}

func SetReadDeadline(fd int, t int64) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	atomic.StoreInt64(&f.rddeadline, t)
	if bq := f.rd; bq != nil {
		bq.Lock()
		if timer := bq.rtimer; timer != nil {
			timer.reset(&bq.queue, t)
		}
		bq.Unlock()
	}
	return nil
}

func (f *netFile) readDeadline() int64 {
	return atomic.LoadInt64(&f.rddeadline)
}

func SetWriteDeadline(fd int, t int64) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	atomic.StoreInt64(&f.wrdeadline, t)
	if bq := f.wr; bq != nil {
		bq.Lock()
		if timer := bq.wtimer; timer != nil {
			timer.reset(&bq.queue, t)
		}
		bq.Unlock()
	}
	return nil
}

func (f *netFile) writeDeadline() int64 {
	return atomic.LoadInt64(&f.wrdeadline)
}

func Shutdown(fd int, how int) error {
	f, err := fdToNetFile(fd)
	if err != nil {
		return err
	}
	switch how {
	case SHUT_RD:
		f.rd.close()
	case SHUT_WR:
		f.wr.close()
	case SHUT_RDWR:
		f.rd.close()
		f.wr.close()
	}
	return nil
}

func SetsockoptICMPv6Filter(fd, level, opt int, filter *ICMPv6Filter) error { panic("SetsockoptICMPv") }
func SetsockoptIPMreq(fd, level, opt int, mreq *IPMreq) error               { panic("SetsockoptIPMreq") }
func SetsockoptIPv6Mreq(fd, level, opt int, mreq *IPv6Mreq) error           { panic("SetsockoptIPv") }
func SetsockoptInet4Addr(fd, level, opt int, value [4]byte) error           { panic("SetsockoptInet") }
func SetsockoptString(fd, level, opt int, s string) error                   { panic("SetsockoptString") }
func SetsockoptTimeval(fd, level, opt int, tv *Timeval) error               { panic("SetsockoptTimeval") }
func Socketpair(domain, typ, proto int) (fd [2]int, err error)              { panic("Socketpair") }

func SetNonblock(fd int, nonblocking bool) error { return nil }
