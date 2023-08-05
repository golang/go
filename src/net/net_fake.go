// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fake networking for js/wasm and wasip1/wasm. It is intended to allow tests of other package to pass.

//go:build (js && wasm) || wasip1

package net

import (
	"context"
	"io"
	"os"
	"sync"
	"syscall"
	"time"
)

var listenersMu sync.Mutex
var listeners = make(map[fakeNetAddr]*netFD)

var portCounterMu sync.Mutex
var portCounter = 0

func nextPort() int {
	portCounterMu.Lock()
	defer portCounterMu.Unlock()
	portCounter++
	return portCounter
}

type fakeNetAddr struct {
	network string
	address string
}

type fakeNetFD struct {
	listener fakeNetAddr
	r        *bufferedPipe
	w        *bufferedPipe
	incoming chan *netFD
	closedMu sync.Mutex
	closed   bool
}

// socket returns a network file descriptor that is ready for
// asynchronous I/O using the network poller.
func socket(ctx context.Context, net string, family, sotype, proto int, ipv6only bool, laddr, raddr sockaddr, ctrlCtxFn func(context.Context, string, string, syscall.RawConn) error) (*netFD, error) {
	fd := &netFD{family: family, sotype: sotype, net: net}
	if laddr != nil && raddr == nil {
		return fakelistener(fd, laddr)
	}
	fd2 := &netFD{family: family, sotype: sotype, net: net}
	return fakeconn(fd, fd2, laddr, raddr)
}

func fakeIPAndPort(ip IP, port int) (IP, int) {
	if ip == nil {
		ip = IPv4(127, 0, 0, 1)
	}
	if port == 0 {
		port = nextPort()
	}
	return ip, port
}

func fakeTCPAddr(addr *TCPAddr) *TCPAddr {
	var ip IP
	var port int
	var zone string
	if addr != nil {
		ip, port, zone = addr.IP, addr.Port, addr.Zone
	}
	ip, port = fakeIPAndPort(ip, port)
	return &TCPAddr{IP: ip, Port: port, Zone: zone}
}

func fakeUDPAddr(addr *UDPAddr) *UDPAddr {
	var ip IP
	var port int
	var zone string
	if addr != nil {
		ip, port, zone = addr.IP, addr.Port, addr.Zone
	}
	ip, port = fakeIPAndPort(ip, port)
	return &UDPAddr{IP: ip, Port: port, Zone: zone}
}

func fakeUnixAddr(sotype int, addr *UnixAddr) *UnixAddr {
	var net, name string
	if addr != nil {
		name = addr.Name
	}
	switch sotype {
	case syscall.SOCK_DGRAM:
		net = "unixgram"
	case syscall.SOCK_SEQPACKET:
		net = "unixpacket"
	default:
		net = "unix"
	}
	return &UnixAddr{Net: net, Name: name}
}

func fakelistener(fd *netFD, laddr sockaddr) (*netFD, error) {
	switch l := laddr.(type) {
	case *TCPAddr:
		laddr = fakeTCPAddr(l)
	case *UDPAddr:
		laddr = fakeUDPAddr(l)
	case *UnixAddr:
		if l.Name == "" {
			return nil, syscall.ENOENT
		}
		laddr = fakeUnixAddr(fd.sotype, l)
	default:
		return nil, syscall.EOPNOTSUPP
	}

	listener := fakeNetAddr{
		network: laddr.Network(),
		address: laddr.String(),
	}

	fd.fakeNetFD = &fakeNetFD{
		listener: listener,
		incoming: make(chan *netFD, 1024),
	}

	fd.laddr = laddr
	listenersMu.Lock()
	defer listenersMu.Unlock()
	if _, exists := listeners[listener]; exists {
		return nil, syscall.EADDRINUSE
	}
	listeners[listener] = fd
	return fd, nil
}

func fakeconn(fd *netFD, fd2 *netFD, laddr, raddr sockaddr) (*netFD, error) {
	switch r := raddr.(type) {
	case *TCPAddr:
		r = fakeTCPAddr(r)
		raddr = r
		laddr = fakeTCPAddr(laddr.(*TCPAddr))
	case *UDPAddr:
		r = fakeUDPAddr(r)
		raddr = r
		laddr = fakeUDPAddr(laddr.(*UDPAddr))
	case *UnixAddr:
		r = fakeUnixAddr(fd.sotype, r)
		raddr = r
		laddr = &UnixAddr{Net: r.Net, Name: r.Name}
	default:
		return nil, syscall.EAFNOSUPPORT
	}
	fd.laddr = laddr
	fd.raddr = raddr

	fd.fakeNetFD = &fakeNetFD{
		r: newBufferedPipe(65536),
		w: newBufferedPipe(65536),
	}
	fd2.fakeNetFD = &fakeNetFD{
		r: fd.fakeNetFD.w,
		w: fd.fakeNetFD.r,
	}

	fd2.laddr = fd.raddr
	fd2.raddr = fd.laddr

	listener := fakeNetAddr{
		network: fd.raddr.Network(),
		address: fd.raddr.String(),
	}
	listenersMu.Lock()
	defer listenersMu.Unlock()
	l, ok := listeners[listener]
	if !ok {
		return nil, syscall.ECONNREFUSED
	}
	l.incoming <- fd2
	return fd, nil
}

func (fd *fakeNetFD) Read(p []byte) (n int, err error) {
	return fd.r.Read(p)
}

func (fd *fakeNetFD) Write(p []byte) (nn int, err error) {
	return fd.w.Write(p)
}

func (fd *fakeNetFD) Close() error {
	fd.closedMu.Lock()
	if fd.closed {
		fd.closedMu.Unlock()
		return nil
	}
	fd.closed = true
	fd.closedMu.Unlock()

	if fd.listener != (fakeNetAddr{}) {
		listenersMu.Lock()
		delete(listeners, fd.listener)
		close(fd.incoming)
		fd.listener = fakeNetAddr{}
		listenersMu.Unlock()
		return nil
	}

	fd.r.Close()
	fd.w.Close()
	return nil
}

func (fd *fakeNetFD) closeRead() error {
	fd.r.Close()
	return nil
}

func (fd *fakeNetFD) closeWrite() error {
	fd.w.Close()
	return nil
}

func (fd *fakeNetFD) accept() (*netFD, error) {
	c, ok := <-fd.incoming
	if !ok {
		return nil, syscall.EINVAL
	}
	return c, nil
}

func (fd *fakeNetFD) SetDeadline(t time.Time) error {
	fd.r.SetReadDeadline(t)
	fd.w.SetWriteDeadline(t)
	return nil
}

func (fd *fakeNetFD) SetReadDeadline(t time.Time) error {
	fd.r.SetReadDeadline(t)
	return nil
}

func (fd *fakeNetFD) SetWriteDeadline(t time.Time) error {
	fd.w.SetWriteDeadline(t)
	return nil
}

func newBufferedPipe(softLimit int) *bufferedPipe {
	p := &bufferedPipe{softLimit: softLimit}
	p.rCond.L = &p.mu
	p.wCond.L = &p.mu
	return p
}

type bufferedPipe struct {
	softLimit int
	mu        sync.Mutex
	buf       []byte
	closed    bool
	rCond     sync.Cond
	wCond     sync.Cond
	rDeadline time.Time
	wDeadline time.Time
}

func (p *bufferedPipe) Read(b []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for {
		if p.closed && len(p.buf) == 0 {
			return 0, io.EOF
		}
		if !p.rDeadline.IsZero() {
			d := time.Until(p.rDeadline)
			if d <= 0 {
				return 0, os.ErrDeadlineExceeded
			}
			time.AfterFunc(d, p.rCond.Broadcast)
		}
		if len(p.buf) > 0 {
			break
		}
		p.rCond.Wait()
	}

	n := copy(b, p.buf)
	p.buf = p.buf[n:]
	p.wCond.Broadcast()
	return n, nil
}

func (p *bufferedPipe) Write(b []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for {
		if p.closed {
			return 0, syscall.ENOTCONN
		}
		if !p.wDeadline.IsZero() {
			d := time.Until(p.wDeadline)
			if d <= 0 {
				return 0, os.ErrDeadlineExceeded
			}
			time.AfterFunc(d, p.wCond.Broadcast)
		}
		if len(p.buf) <= p.softLimit {
			break
		}
		p.wCond.Wait()
	}

	p.buf = append(p.buf, b...)
	p.rCond.Broadcast()
	return len(b), nil
}

func (p *bufferedPipe) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.closed = true
	p.rCond.Broadcast()
	p.wCond.Broadcast()
}

func (p *bufferedPipe) SetReadDeadline(t time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.rDeadline = t
	p.rCond.Broadcast()
}

func (p *bufferedPipe) SetWriteDeadline(t time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.wDeadline = t
	p.wCond.Broadcast()
}

func sysSocket(family, sotype, proto int) (int, error) {
	return 0, syscall.ENOSYS
}

func (fd *fakeNetFD) connect(ctx context.Context, la, ra syscall.Sockaddr) (syscall.Sockaddr, error) {
	return nil, syscall.ENOSYS
}

func (fd *fakeNetFD) readFrom(p []byte) (n int, sa syscall.Sockaddr, err error) {
	return 0, nil, syscall.ENOSYS

}
func (fd *fakeNetFD) readFromInet4(p []byte, sa *syscall.SockaddrInet4) (n int, err error) {
	return 0, syscall.ENOSYS
}

func (fd *fakeNetFD) readFromInet6(p []byte, sa *syscall.SockaddrInet6) (n int, err error) {
	return 0, syscall.ENOSYS
}

func (fd *fakeNetFD) readMsg(p []byte, oob []byte, flags int) (n, oobn, retflags int, sa syscall.Sockaddr, err error) {
	return 0, 0, 0, nil, syscall.ENOSYS
}

func (fd *fakeNetFD) readMsgInet4(p []byte, oob []byte, flags int, sa *syscall.SockaddrInet4) (n, oobn, retflags int, err error) {
	return 0, 0, 0, syscall.ENOSYS
}

func (fd *fakeNetFD) readMsgInet6(p []byte, oob []byte, flags int, sa *syscall.SockaddrInet6) (n, oobn, retflags int, err error) {
	return 0, 0, 0, syscall.ENOSYS
}

func (fd *fakeNetFD) writeMsgInet4(p []byte, oob []byte, sa *syscall.SockaddrInet4) (n int, oobn int, err error) {
	return 0, 0, syscall.ENOSYS
}

func (fd *fakeNetFD) writeMsgInet6(p []byte, oob []byte, sa *syscall.SockaddrInet6) (n int, oobn int, err error) {
	return 0, 0, syscall.ENOSYS
}

func (fd *fakeNetFD) writeTo(p []byte, sa syscall.Sockaddr) (n int, err error) {
	return 0, syscall.ENOSYS
}

func (fd *fakeNetFD) writeToInet4(p []byte, sa *syscall.SockaddrInet4) (n int, err error) {
	return 0, syscall.ENOSYS
}

func (fd *fakeNetFD) writeToInet6(p []byte, sa *syscall.SockaddrInet6) (n int, err error) {
	return 0, syscall.ENOSYS
}

func (fd *fakeNetFD) writeMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	return 0, 0, syscall.ENOSYS
}

func (fd *fakeNetFD) dup() (f *os.File, err error) {
	return nil, syscall.ENOSYS
}
