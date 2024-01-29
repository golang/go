// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fake networking for js/wasm and wasip1/wasm.
// It is intended to allow tests of other package to pass.

//go:build js || wasip1

package net

import (
	"context"
	"errors"
	"io"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

var (
	sockets         sync.Map // fakeSockAddr → *netFD
	fakePorts       sync.Map // int (port #) → *netFD
	nextPortCounter atomic.Int32
)

const defaultBuffer = 65535

type fakeSockAddr struct {
	family  int
	address string
}

func fakeAddr(sa sockaddr) fakeSockAddr {
	return fakeSockAddr{
		family:  sa.family(),
		address: sa.String(),
	}
}

// socket returns a network file descriptor that is ready for
// I/O using the fake network.
func socket(ctx context.Context, net string, family, sotype, proto int, ipv6only bool, laddr, raddr sockaddr, ctrlCtxFn func(context.Context, string, string, syscall.RawConn) error) (*netFD, error) {
	if raddr != nil && ctrlCtxFn != nil {
		return nil, os.NewSyscallError("socket", syscall.ENOTSUP)
	}
	switch sotype {
	case syscall.SOCK_STREAM, syscall.SOCK_SEQPACKET, syscall.SOCK_DGRAM:
	default:
		return nil, os.NewSyscallError("socket", syscall.ENOTSUP)
	}

	fd := &netFD{
		family: family,
		sotype: sotype,
		net:    net,
	}
	fd.fakeNetFD = newFakeNetFD(fd)

	if raddr == nil {
		if err := fakeListen(fd, laddr); err != nil {
			fd.Close()
			return nil, err
		}
		return fd, nil
	}

	if err := fakeConnect(ctx, fd, laddr, raddr); err != nil {
		fd.Close()
		return nil, err
	}
	return fd, nil
}

func validateResolvedAddr(net string, family int, sa sockaddr) error {
	validateIP := func(ip IP) error {
		switch family {
		case syscall.AF_INET:
			if len(ip) != 4 {
				return &AddrError{
					Err:  "non-IPv4 address",
					Addr: ip.String(),
				}
			}
		case syscall.AF_INET6:
			if len(ip) != 16 {
				return &AddrError{
					Err:  "non-IPv6 address",
					Addr: ip.String(),
				}
			}
		default:
			panic("net: unexpected address family in validateResolvedAddr")
		}
		return nil
	}

	switch net {
	case "tcp", "tcp4", "tcp6":
		sa, ok := sa.(*TCPAddr)
		if !ok {
			return &AddrError{
				Err:  "non-TCP address for " + net + " network",
				Addr: sa.String(),
			}
		}
		if err := validateIP(sa.IP); err != nil {
			return err
		}
		if sa.Port <= 0 || sa.Port >= 1<<16 {
			return &AddrError{
				Err:  "port out of range",
				Addr: sa.String(),
			}
		}
		return nil

	case "udp", "udp4", "udp6":
		sa, ok := sa.(*UDPAddr)
		if !ok {
			return &AddrError{
				Err:  "non-UDP address for " + net + " network",
				Addr: sa.String(),
			}
		}
		if err := validateIP(sa.IP); err != nil {
			return err
		}
		if sa.Port <= 0 || sa.Port >= 1<<16 {
			return &AddrError{
				Err:  "port out of range",
				Addr: sa.String(),
			}
		}
		return nil

	case "unix", "unixgram", "unixpacket":
		sa, ok := sa.(*UnixAddr)
		if !ok {
			return &AddrError{
				Err:  "non-Unix address for " + net + " network",
				Addr: sa.String(),
			}
		}
		if sa.Name != "" {
			i := len(sa.Name) - 1
			for i > 0 && !os.IsPathSeparator(sa.Name[i]) {
				i--
			}
			for i > 0 && os.IsPathSeparator(sa.Name[i]) {
				i--
			}
			if i <= 0 {
				return &AddrError{
					Err:  "unix socket name missing path component",
					Addr: sa.Name,
				}
			}
			if _, err := os.Stat(sa.Name[:i+1]); err != nil {
				return &AddrError{
					Err:  err.Error(),
					Addr: sa.Name,
				}
			}
		}
		return nil

	default:
		return &AddrError{
			Err:  syscall.EAFNOSUPPORT.Error(),
			Addr: sa.String(),
		}
	}
}

func matchIPFamily(family int, addr sockaddr) sockaddr {
	convertIP := func(ip IP) IP {
		switch family {
		case syscall.AF_INET:
			return ip.To4()
		case syscall.AF_INET6:
			return ip.To16()
		default:
			return ip
		}
	}

	switch addr := addr.(type) {
	case *TCPAddr:
		ip := convertIP(addr.IP)
		if ip == nil || len(ip) == len(addr.IP) {
			return addr
		}
		return &TCPAddr{IP: ip, Port: addr.Port, Zone: addr.Zone}
	case *UDPAddr:
		ip := convertIP(addr.IP)
		if ip == nil || len(ip) == len(addr.IP) {
			return addr
		}
		return &UDPAddr{IP: ip, Port: addr.Port, Zone: addr.Zone}
	default:
		return addr
	}
}

type fakeNetFD struct {
	fd           *netFD
	assignedPort int // 0 if no port has been assigned for this socket

	queue         *packetQueue // incoming packets
	peer          *netFD       // connected peer (for outgoing packets); nil for listeners and PacketConns
	readDeadline  atomic.Pointer[deadlineTimer]
	writeDeadline atomic.Pointer[deadlineTimer]

	fakeAddr fakeSockAddr // cached fakeSockAddr equivalent of fd.laddr

	// The incoming channels hold incoming connections that have not yet been accepted.
	// All of these channels are 1-buffered.
	incoming      chan []*netFD // holds the queue when it has >0 but <SOMAXCONN pending connections; closed when the Listener is closed
	incomingFull  chan []*netFD // holds the queue when it has SOMAXCONN pending connections
	incomingEmpty chan bool     // holds true when the incoming queue is empty
}

func newFakeNetFD(fd *netFD) *fakeNetFD {
	ffd := &fakeNetFD{fd: fd}
	ffd.readDeadline.Store(newDeadlineTimer(noDeadline))
	ffd.writeDeadline.Store(newDeadlineTimer(noDeadline))
	return ffd
}

func (ffd *fakeNetFD) Read(p []byte) (n int, err error) {
	n, _, err = ffd.queue.recvfrom(ffd.readDeadline.Load(), p, false, nil)
	return n, err
}

func (ffd *fakeNetFD) Write(p []byte) (nn int, err error) {
	peer := ffd.peer
	if peer == nil {
		if ffd.fd.raddr == nil {
			return 0, os.NewSyscallError("write", syscall.ENOTCONN)
		}
		peeri, _ := sockets.Load(fakeAddr(ffd.fd.raddr.(sockaddr)))
		if peeri == nil {
			return 0, os.NewSyscallError("write", syscall.ECONNRESET)
		}
		peer = peeri.(*netFD)
		if peer.queue == nil {
			return 0, os.NewSyscallError("write", syscall.ECONNRESET)
		}
	}

	if peer.fakeNetFD == nil {
		return 0, os.NewSyscallError("write", syscall.EINVAL)
	}
	return peer.queue.write(ffd.writeDeadline.Load(), p, ffd.fd.laddr.(sockaddr))
}

func (ffd *fakeNetFD) Close() (err error) {
	if ffd.fakeAddr != (fakeSockAddr{}) {
		sockets.CompareAndDelete(ffd.fakeAddr, ffd.fd)
	}

	if ffd.queue != nil {
		if closeErr := ffd.queue.closeRead(); err == nil {
			err = closeErr
		}
	}
	if ffd.peer != nil {
		if closeErr := ffd.peer.queue.closeWrite(); err == nil {
			err = closeErr
		}
	}
	ffd.readDeadline.Load().Reset(noDeadline)
	ffd.writeDeadline.Load().Reset(noDeadline)

	if ffd.incoming != nil {
		var (
			incoming []*netFD
			ok       bool
		)
		select {
		case _, ok = <-ffd.incomingEmpty:
		case incoming, ok = <-ffd.incoming:
		case incoming, ok = <-ffd.incomingFull:
		}
		if ok {
			// Sends on ffd.incoming require a receive first.
			// Since we successfully received, no other goroutine may
			// send on it at this point, and we may safely close it.
			close(ffd.incoming)

			for _, c := range incoming {
				c.Close()
			}
		}
	}

	if ffd.assignedPort != 0 {
		fakePorts.CompareAndDelete(ffd.assignedPort, ffd.fd)
	}

	return err
}

func (ffd *fakeNetFD) closeRead() error {
	return ffd.queue.closeRead()
}

func (ffd *fakeNetFD) closeWrite() error {
	if ffd.peer == nil {
		return os.NewSyscallError("closeWrite", syscall.ENOTCONN)
	}
	return ffd.peer.queue.closeWrite()
}

func (ffd *fakeNetFD) accept(laddr Addr) (*netFD, error) {
	if ffd.incoming == nil {
		return nil, os.NewSyscallError("accept", syscall.EINVAL)
	}

	var (
		incoming []*netFD
		ok       bool
	)
	expired := ffd.readDeadline.Load().expired
	select {
	case <-expired:
		return nil, os.ErrDeadlineExceeded
	case incoming, ok = <-ffd.incoming:
		if !ok {
			return nil, ErrClosed
		}
		select {
		case <-expired:
			ffd.incoming <- incoming
			return nil, os.ErrDeadlineExceeded
		default:
		}
	case incoming, ok = <-ffd.incomingFull:
		select {
		case <-expired:
			ffd.incomingFull <- incoming
			return nil, os.ErrDeadlineExceeded
		default:
		}
	}

	peer := incoming[0]
	incoming = incoming[1:]
	if len(incoming) == 0 {
		ffd.incomingEmpty <- true
	} else {
		ffd.incoming <- incoming
	}
	return peer, nil
}

func (ffd *fakeNetFD) SetDeadline(t time.Time) error {
	err1 := ffd.SetReadDeadline(t)
	err2 := ffd.SetWriteDeadline(t)
	if err1 != nil {
		return err1
	}
	return err2
}

func (ffd *fakeNetFD) SetReadDeadline(t time.Time) error {
	dt := ffd.readDeadline.Load()
	if !dt.Reset(t) {
		ffd.readDeadline.Store(newDeadlineTimer(t))
	}
	return nil
}

func (ffd *fakeNetFD) SetWriteDeadline(t time.Time) error {
	dt := ffd.writeDeadline.Load()
	if !dt.Reset(t) {
		ffd.writeDeadline.Store(newDeadlineTimer(t))
	}
	return nil
}

const maxPacketSize = 65535

type packet struct {
	buf       []byte
	bufOffset int
	next      *packet
	from      sockaddr
}

func (p *packet) clear() {
	p.buf = p.buf[:0]
	p.bufOffset = 0
	p.next = nil
	p.from = nil
}

var packetPool = sync.Pool{
	New: func() any { return new(packet) },
}

type packetQueueState struct {
	head, tail      *packet // unqueued packets
	nBytes          int     // number of bytes enqueued in the packet buffers starting from head
	readBufferBytes int     // soft limit on nbytes; no more packets may be enqueued when the limit is exceeded
	readClosed      bool    // true if the reader of the queue has stopped reading
	writeClosed     bool    // true if the writer of the queue has stopped writing; the reader sees either io.EOF or syscall.ECONNRESET when they have read all buffered packets
	noLinger        bool    // if true, the reader sees ECONNRESET instead of EOF
}

// A packetQueue is a set of 1-buffered channels implementing a FIFO queue
// of packets.
type packetQueue struct {
	empty chan packetQueueState // contains configuration parameters when the queue is empty and not closed
	ready chan packetQueueState // contains the packets when non-empty or closed
	full  chan packetQueueState // contains the packets when buffer is full and not closed
}

func newPacketQueue(readBufferBytes int) *packetQueue {
	pq := &packetQueue{
		empty: make(chan packetQueueState, 1),
		ready: make(chan packetQueueState, 1),
		full:  make(chan packetQueueState, 1),
	}
	pq.put(packetQueueState{
		readBufferBytes: readBufferBytes,
	})
	return pq
}

func (pq *packetQueue) get() packetQueueState {
	var q packetQueueState
	select {
	case q = <-pq.empty:
	case q = <-pq.ready:
	case q = <-pq.full:
	}
	return q
}

func (pq *packetQueue) put(q packetQueueState) {
	switch {
	case q.readClosed || q.writeClosed:
		pq.ready <- q
	case q.nBytes >= q.readBufferBytes:
		pq.full <- q
	case q.head == nil:
		if q.nBytes > 0 {
			defer panic("net: put with nil packet list and nonzero nBytes")
		}
		pq.empty <- q
	default:
		pq.ready <- q
	}
}

func (pq *packetQueue) closeRead() error {
	q := pq.get()

	// Discard any unread packets.
	for q.head != nil {
		p := q.head
		q.head = p.next
		p.clear()
		packetPool.Put(p)
	}
	q.nBytes = 0

	q.readClosed = true
	pq.put(q)
	return nil
}

func (pq *packetQueue) closeWrite() error {
	q := pq.get()
	q.writeClosed = true
	pq.put(q)
	return nil
}

func (pq *packetQueue) setLinger(linger bool) error {
	q := pq.get()
	defer func() { pq.put(q) }()

	if q.writeClosed {
		return ErrClosed
	}
	q.noLinger = !linger
	return nil
}

func (pq *packetQueue) write(dt *deadlineTimer, b []byte, from sockaddr) (n int, err error) {
	for {
		dn := len(b)
		if dn > maxPacketSize {
			dn = maxPacketSize
		}

		dn, err = pq.send(dt, b[:dn], from, true)
		n += dn
		if err != nil {
			return n, err
		}

		b = b[dn:]
		if len(b) == 0 {
			return n, nil
		}
	}
}

func (pq *packetQueue) send(dt *deadlineTimer, b []byte, from sockaddr, block bool) (n int, err error) {
	if from == nil {
		return 0, os.NewSyscallError("send", syscall.EINVAL)
	}
	if len(b) > maxPacketSize {
		return 0, os.NewSyscallError("send", syscall.EMSGSIZE)
	}

	var q packetQueueState
	var full chan packetQueueState
	if !block {
		full = pq.full
	}

	// Before we check dt.expired, yield to other goroutines.
	// This may help to prevent starvation of the goroutine that runs the
	// deadlineTimer's time.After callback.
	//
	// TODO(#65178): Remove this when the runtime scheduler no longer starves
	// runnable goroutines.
	runtime.Gosched()

	select {
	case <-dt.expired:
		return 0, os.ErrDeadlineExceeded

	case q = <-full:
		pq.put(q)
		return 0, os.NewSyscallError("send", syscall.ENOBUFS)

	case q = <-pq.empty:
	case q = <-pq.ready:
	}
	defer func() { pq.put(q) }()

	// Don't allow a packet to be sent if the deadline has expired,
	// even if the select above chose a different branch.
	select {
	case <-dt.expired:
		return 0, os.ErrDeadlineExceeded
	default:
	}
	if q.writeClosed {
		return 0, ErrClosed
	} else if q.readClosed {
		return 0, os.NewSyscallError("send", syscall.ECONNRESET)
	}

	p := packetPool.Get().(*packet)
	p.buf = append(p.buf[:0], b...)
	p.from = from

	if q.head == nil {
		q.head = p
	} else {
		q.tail.next = p
	}
	q.tail = p
	q.nBytes += len(p.buf)

	return len(b), nil
}

func (pq *packetQueue) recvfrom(dt *deadlineTimer, b []byte, wholePacket bool, checkFrom func(sockaddr) error) (n int, from sockaddr, err error) {
	var q packetQueueState
	var empty chan packetQueueState
	if len(b) == 0 {
		// For consistency with the implementation on Unix platforms,
		// allow a zero-length Read to proceed if the queue is empty.
		// (Without this, TestZeroByteRead deadlocks.)
		empty = pq.empty
	}

	// Before we check dt.expired, yield to other goroutines.
	// This may help to prevent starvation of the goroutine that runs the
	// deadlineTimer's time.After callback.
	//
	// TODO(#65178): Remove this when the runtime scheduler no longer starves
	// runnable goroutines.
	runtime.Gosched()

	select {
	case <-dt.expired:
		return 0, nil, os.ErrDeadlineExceeded
	case q = <-empty:
	case q = <-pq.ready:
	case q = <-pq.full:
	}
	defer func() { pq.put(q) }()

	p := q.head
	if p == nil {
		switch {
		case q.readClosed:
			return 0, nil, ErrClosed
		case q.writeClosed:
			if q.noLinger {
				return 0, nil, os.NewSyscallError("recvfrom", syscall.ECONNRESET)
			}
			return 0, nil, io.EOF
		case len(b) == 0:
			return 0, nil, nil
		default:
			// This should be impossible: pq.full should only contain a non-empty list,
			// pq.ready should either contain a non-empty list or indicate that the
			// connection is closed, and we should only receive from pq.empty if
			// len(b) == 0.
			panic("net: nil packet list from non-closed packetQueue")
		}
	}

	select {
	case <-dt.expired:
		return 0, nil, os.ErrDeadlineExceeded
	default:
	}

	if checkFrom != nil {
		if err := checkFrom(p.from); err != nil {
			return 0, nil, err
		}
	}

	n = copy(b, p.buf[p.bufOffset:])
	from = p.from
	if wholePacket || p.bufOffset+n == len(p.buf) {
		q.head = p.next
		q.nBytes -= len(p.buf)
		p.clear()
		packetPool.Put(p)
	} else {
		p.bufOffset += n
	}

	return n, from, nil
}

// setReadBuffer sets a soft limit on the number of bytes available to read
// from the pipe.
func (pq *packetQueue) setReadBuffer(bytes int) error {
	if bytes <= 0 {
		return os.NewSyscallError("setReadBuffer", syscall.EINVAL)
	}
	q := pq.get() // Use the queue as a lock.
	q.readBufferBytes = bytes
	pq.put(q)
	return nil
}

type deadlineTimer struct {
	timer   chan *time.Timer
	expired chan struct{}
}

func newDeadlineTimer(deadline time.Time) *deadlineTimer {
	dt := &deadlineTimer{
		timer:   make(chan *time.Timer, 1),
		expired: make(chan struct{}),
	}
	dt.timer <- nil
	dt.Reset(deadline)
	return dt
}

// Reset attempts to reset the timer.
// If the timer has already expired, Reset returns false.
func (dt *deadlineTimer) Reset(deadline time.Time) bool {
	timer := <-dt.timer
	defer func() { dt.timer <- timer }()

	if deadline.Equal(noDeadline) {
		if timer != nil && timer.Stop() {
			timer = nil
		}
		return timer == nil
	}

	d := time.Until(deadline)
	if d < 0 {
		// Ensure that a deadline in the past takes effect immediately.
		defer func() { <-dt.expired }()
	}

	if timer == nil {
		timer = time.AfterFunc(d, func() { close(dt.expired) })
		return true
	}
	if !timer.Stop() {
		return false
	}
	timer.Reset(d)
	return true
}

func sysSocket(family, sotype, proto int) (int, error) {
	return 0, os.NewSyscallError("sysSocket", syscall.ENOSYS)
}

func fakeListen(fd *netFD, laddr sockaddr) (err error) {
	wrapErr := func(err error) error {
		if errno, ok := err.(syscall.Errno); ok {
			err = os.NewSyscallError("listen", errno)
		}
		if errors.Is(err, syscall.EADDRINUSE) {
			return err
		}
		if laddr != nil {
			if _, ok := err.(*AddrError); !ok {
				err = &AddrError{
					Err:  err.Error(),
					Addr: laddr.String(),
				}
			}
		}
		return err
	}

	ffd := newFakeNetFD(fd)
	defer func() {
		if fd.fakeNetFD != ffd {
			// Failed to register listener; clean up.
			ffd.Close()
		}
	}()

	if err := ffd.assignFakeAddr(matchIPFamily(fd.family, laddr)); err != nil {
		return wrapErr(err)
	}

	ffd.fakeAddr = fakeAddr(fd.laddr.(sockaddr))
	switch fd.sotype {
	case syscall.SOCK_STREAM, syscall.SOCK_SEQPACKET:
		ffd.incoming = make(chan []*netFD, 1)
		ffd.incomingFull = make(chan []*netFD, 1)
		ffd.incomingEmpty = make(chan bool, 1)
		ffd.incomingEmpty <- true
	case syscall.SOCK_DGRAM:
		ffd.queue = newPacketQueue(defaultBuffer)
	default:
		return wrapErr(syscall.EINVAL)
	}

	fd.fakeNetFD = ffd
	if _, dup := sockets.LoadOrStore(ffd.fakeAddr, fd); dup {
		fd.fakeNetFD = nil
		return wrapErr(syscall.EADDRINUSE)
	}

	return nil
}

func fakeConnect(ctx context.Context, fd *netFD, laddr, raddr sockaddr) error {
	wrapErr := func(err error) error {
		if errno, ok := err.(syscall.Errno); ok {
			err = os.NewSyscallError("connect", errno)
		}
		if errors.Is(err, syscall.EADDRINUSE) {
			return err
		}
		if terr, ok := err.(interface{ Timeout() bool }); !ok || !terr.Timeout() {
			// For consistency with the net implementation on other platforms,
			// if we don't need to preserve the Timeout-ness of err we should
			// wrap it in an AddrError. (Unfortunately we can't wrap errors
			// that convey structured information, because AddrError reduces
			// the wrapped Err to a flat string.)
			if _, ok := err.(*AddrError); !ok {
				err = &AddrError{
					Err:  err.Error(),
					Addr: raddr.String(),
				}
			}
		}
		return err
	}

	if fd.isConnected {
		return wrapErr(syscall.EISCONN)
	}
	if ctx.Err() != nil {
		return wrapErr(syscall.ETIMEDOUT)
	}

	fd.raddr = matchIPFamily(fd.family, raddr)
	if err := validateResolvedAddr(fd.net, fd.family, fd.raddr.(sockaddr)); err != nil {
		return wrapErr(err)
	}

	if err := fd.fakeNetFD.assignFakeAddr(laddr); err != nil {
		return wrapErr(err)
	}
	fd.fakeNetFD.queue = newPacketQueue(defaultBuffer)

	switch fd.sotype {
	case syscall.SOCK_DGRAM:
		if ua, ok := fd.laddr.(*UnixAddr); !ok || ua.Name != "" {
			fd.fakeNetFD.fakeAddr = fakeAddr(fd.laddr.(sockaddr))
			if _, dup := sockets.LoadOrStore(fd.fakeNetFD.fakeAddr, fd); dup {
				return wrapErr(syscall.EADDRINUSE)
			}
		}
		fd.isConnected = true
		return nil

	case syscall.SOCK_STREAM, syscall.SOCK_SEQPACKET:
	default:
		return wrapErr(syscall.EINVAL)
	}

	fa := fakeAddr(raddr)
	lni, ok := sockets.Load(fa)
	if !ok {
		return wrapErr(syscall.ECONNREFUSED)
	}
	ln := lni.(*netFD)
	if ln.sotype != fd.sotype {
		return wrapErr(syscall.EPROTOTYPE)
	}
	if ln.incoming == nil {
		return wrapErr(syscall.ECONNREFUSED)
	}

	peer := &netFD{
		family:      ln.family,
		sotype:      ln.sotype,
		net:         ln.net,
		laddr:       ln.laddr,
		raddr:       fd.laddr,
		isConnected: true,
	}
	peer.fakeNetFD = newFakeNetFD(fd)
	peer.fakeNetFD.queue = newPacketQueue(defaultBuffer)
	defer func() {
		if fd.peer != peer {
			// Failed to connect; clean up.
			peer.Close()
		}
	}()

	var incoming []*netFD
	select {
	case <-ctx.Done():
		return wrapErr(syscall.ETIMEDOUT)
	case ok = <-ln.incomingEmpty:
	case incoming, ok = <-ln.incoming:
	}
	if !ok {
		return wrapErr(syscall.ECONNREFUSED)
	}

	fd.isConnected = true
	fd.peer = peer
	peer.peer = fd

	incoming = append(incoming, peer)
	if len(incoming) >= listenerBacklog() {
		ln.incomingFull <- incoming
	} else {
		ln.incoming <- incoming
	}
	return nil
}

func (ffd *fakeNetFD) assignFakeAddr(addr sockaddr) error {
	validate := func(sa sockaddr) error {
		if err := validateResolvedAddr(ffd.fd.net, ffd.fd.family, sa); err != nil {
			return err
		}
		ffd.fd.laddr = sa
		return nil
	}

	assignIP := func(addr sockaddr) error {
		var (
			ip   IP
			port int
			zone string
		)
		switch addr := addr.(type) {
		case *TCPAddr:
			if addr != nil {
				ip = addr.IP
				port = addr.Port
				zone = addr.Zone
			}
		case *UDPAddr:
			if addr != nil {
				ip = addr.IP
				port = addr.Port
				zone = addr.Zone
			}
		default:
			return validate(addr)
		}

		if ip == nil {
			ip = IPv4(127, 0, 0, 1)
		}
		switch ffd.fd.family {
		case syscall.AF_INET:
			if ip4 := ip.To4(); ip4 != nil {
				ip = ip4
			}
		case syscall.AF_INET6:
			if ip16 := ip.To16(); ip16 != nil {
				ip = ip16
			}
		}
		if ip == nil {
			return syscall.EINVAL
		}

		if port == 0 {
			var prevPort int32
			portWrapped := false
			nextPort := func() (int, bool) {
				for {
					port := nextPortCounter.Add(1)
					if port <= 0 || port >= 1<<16 {
						// nextPortCounter ran off the end of the port space.
						// Bump it back into range.
						for {
							if nextPortCounter.CompareAndSwap(port, 0) {
								break
							}
							if port = nextPortCounter.Load(); port >= 0 && port < 1<<16 {
								break
							}
						}
						if portWrapped {
							// This is the second wraparound, so we've scanned the whole port space
							// at least once already and it's time to give up.
							return 0, false
						}
						portWrapped = true
						prevPort = 0
						continue
					}

					if port <= prevPort {
						// nextPortCounter has wrapped around since the last time we read it.
						if portWrapped {
							// This is the second wraparound, so we've scanned the whole port space
							// at least once already and it's time to give up.
							return 0, false
						} else {
							portWrapped = true
						}
					}

					prevPort = port
					return int(port), true
				}
			}

			for {
				var ok bool
				port, ok = nextPort()
				if !ok {
					ffd.assignedPort = 0
					return syscall.EADDRINUSE
				}

				ffd.assignedPort = int(port)
				if _, dup := fakePorts.LoadOrStore(ffd.assignedPort, ffd.fd); !dup {
					break
				}
			}
		}

		switch addr.(type) {
		case *TCPAddr:
			return validate(&TCPAddr{IP: ip, Port: port, Zone: zone})
		case *UDPAddr:
			return validate(&UDPAddr{IP: ip, Port: port, Zone: zone})
		default:
			panic("unreachable")
		}
	}

	switch ffd.fd.net {
	case "tcp", "tcp4", "tcp6":
		if addr == nil {
			return assignIP(new(TCPAddr))
		}
		return assignIP(addr)

	case "udp", "udp4", "udp6":
		if addr == nil {
			return assignIP(new(UDPAddr))
		}
		return assignIP(addr)

	case "unix", "unixgram", "unixpacket":
		uaddr, ok := addr.(*UnixAddr)
		if !ok && addr != nil {
			return &AddrError{
				Err:  "non-Unix address for " + ffd.fd.net + " network",
				Addr: addr.String(),
			}
		}
		if uaddr == nil {
			return validate(&UnixAddr{Net: ffd.fd.net})
		}
		return validate(&UnixAddr{Net: ffd.fd.net, Name: uaddr.Name})

	default:
		return &AddrError{
			Err:  syscall.EAFNOSUPPORT.Error(),
			Addr: addr.String(),
		}
	}
}

func (ffd *fakeNetFD) readFrom(p []byte) (n int, sa syscall.Sockaddr, err error) {
	if ffd.queue == nil {
		return 0, nil, os.NewSyscallError("readFrom", syscall.EINVAL)
	}

	n, from, err := ffd.queue.recvfrom(ffd.readDeadline.Load(), p, true, nil)

	if from != nil {
		// Convert the net.sockaddr to a syscall.Sockaddr type.
		var saErr error
		sa, saErr = from.sockaddr(ffd.fd.family)
		if err == nil {
			err = saErr
		}
	}

	return n, sa, err
}

func (ffd *fakeNetFD) readFromInet4(p []byte, sa *syscall.SockaddrInet4) (n int, err error) {
	n, _, err = ffd.queue.recvfrom(ffd.readDeadline.Load(), p, true, func(from sockaddr) error {
		fromSA, err := from.sockaddr(syscall.AF_INET)
		if err != nil {
			return err
		}
		if fromSA == nil {
			return os.NewSyscallError("readFromInet4", syscall.EINVAL)
		}
		*sa = *(fromSA.(*syscall.SockaddrInet4))
		return nil
	})
	return n, err
}

func (ffd *fakeNetFD) readFromInet6(p []byte, sa *syscall.SockaddrInet6) (n int, err error) {
	n, _, err = ffd.queue.recvfrom(ffd.readDeadline.Load(), p, true, func(from sockaddr) error {
		fromSA, err := from.sockaddr(syscall.AF_INET6)
		if err != nil {
			return err
		}
		if fromSA == nil {
			return os.NewSyscallError("readFromInet6", syscall.EINVAL)
		}
		*sa = *(fromSA.(*syscall.SockaddrInet6))
		return nil
	})
	return n, err
}

func (ffd *fakeNetFD) readMsg(p []byte, oob []byte, flags int) (n, oobn, retflags int, sa syscall.Sockaddr, err error) {
	if flags != 0 {
		return 0, 0, 0, nil, os.NewSyscallError("readMsg", syscall.ENOTSUP)
	}
	n, sa, err = ffd.readFrom(p)
	return n, 0, 0, sa, err
}

func (ffd *fakeNetFD) readMsgInet4(p []byte, oob []byte, flags int, sa *syscall.SockaddrInet4) (n, oobn, retflags int, err error) {
	if flags != 0 {
		return 0, 0, 0, os.NewSyscallError("readMsgInet4", syscall.ENOTSUP)
	}
	n, err = ffd.readFromInet4(p, sa)
	return n, 0, 0, err
}

func (ffd *fakeNetFD) readMsgInet6(p []byte, oob []byte, flags int, sa *syscall.SockaddrInet6) (n, oobn, retflags int, err error) {
	if flags != 0 {
		return 0, 0, 0, os.NewSyscallError("readMsgInet6", syscall.ENOTSUP)
	}
	n, err = ffd.readFromInet6(p, sa)
	return n, 0, 0, err
}

func (ffd *fakeNetFD) writeMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	if len(oob) > 0 {
		return 0, 0, os.NewSyscallError("writeMsg", syscall.ENOTSUP)
	}
	n, err = ffd.writeTo(p, sa)
	return n, 0, err
}

func (ffd *fakeNetFD) writeMsgInet4(p []byte, oob []byte, sa *syscall.SockaddrInet4) (n int, oobn int, err error) {
	return ffd.writeMsg(p, oob, sa)
}

func (ffd *fakeNetFD) writeMsgInet6(p []byte, oob []byte, sa *syscall.SockaddrInet6) (n int, oobn int, err error) {
	return ffd.writeMsg(p, oob, sa)
}

func (ffd *fakeNetFD) writeTo(p []byte, sa syscall.Sockaddr) (n int, err error) {
	raddr := ffd.fd.raddr
	if sa != nil {
		if ffd.fd.isConnected {
			return 0, os.NewSyscallError("writeTo", syscall.EISCONN)
		}
		raddr = ffd.fd.addrFunc()(sa)
	}
	if raddr == nil {
		return 0, os.NewSyscallError("writeTo", syscall.EINVAL)
	}

	peeri, _ := sockets.Load(fakeAddr(raddr.(sockaddr)))
	if peeri == nil {
		if len(ffd.fd.net) >= 3 && ffd.fd.net[:3] == "udp" {
			return len(p), nil
		}
		return 0, os.NewSyscallError("writeTo", syscall.ECONNRESET)
	}
	peer := peeri.(*netFD)
	if peer.queue == nil {
		if len(ffd.fd.net) >= 3 && ffd.fd.net[:3] == "udp" {
			return len(p), nil
		}
		return 0, os.NewSyscallError("writeTo", syscall.ECONNRESET)
	}

	block := true
	if len(ffd.fd.net) >= 3 && ffd.fd.net[:3] == "udp" {
		block = false
	}
	return peer.queue.send(ffd.writeDeadline.Load(), p, ffd.fd.laddr.(sockaddr), block)
}

func (ffd *fakeNetFD) writeToInet4(p []byte, sa *syscall.SockaddrInet4) (n int, err error) {
	return ffd.writeTo(p, sa)
}

func (ffd *fakeNetFD) writeToInet6(p []byte, sa *syscall.SockaddrInet6) (n int, err error) {
	return ffd.writeTo(p, sa)
}

func (ffd *fakeNetFD) dup() (f *os.File, err error) {
	return nil, os.NewSyscallError("dup", syscall.ENOSYS)
}

func (ffd *fakeNetFD) setReadBuffer(bytes int) error {
	if ffd.queue == nil {
		return os.NewSyscallError("setReadBuffer", syscall.EINVAL)
	}
	ffd.queue.setReadBuffer(bytes)
	return nil
}

func (ffd *fakeNetFD) setWriteBuffer(bytes int) error {
	return os.NewSyscallError("setWriteBuffer", syscall.ENOTSUP)
}

func (ffd *fakeNetFD) setLinger(sec int) error {
	if sec < 0 || ffd.peer == nil {
		return os.NewSyscallError("setLinger", syscall.EINVAL)
	}
	ffd.peer.queue.setLinger(sec > 0)
	return nil
}
