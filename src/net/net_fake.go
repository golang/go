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

	"golang.org/x/net/dns/dnsmessage"
)

var listenersMu sync.Mutex
var listeners = make(map[string]*netFD)

var portCounterMu sync.Mutex
var portCounter = 0

func nextPort() int {
	portCounterMu.Lock()
	defer portCounterMu.Unlock()
	portCounter++
	return portCounter
}

type fakeNetFD struct {
	listener bool
	laddr    Addr
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
	return fakeconn(fd, fd2, raddr)
}

func fakelistener(fd *netFD, laddr sockaddr) (*netFD, error) {
	l := laddr.(*TCPAddr)
	fd.laddr = &TCPAddr{
		IP:   l.IP,
		Port: nextPort(),
		Zone: l.Zone,
	}
	fd.fakeNetFD = &fakeNetFD{
		listener: true,
		laddr:    fd.laddr,
		incoming: make(chan *netFD, 1024),
	}
	listenersMu.Lock()
	listeners[fd.laddr.(*TCPAddr).String()] = fd
	listenersMu.Unlock()
	return fd, nil
}

func fakeconn(fd *netFD, fd2 *netFD, raddr sockaddr) (*netFD, error) {
	fd.laddr = &TCPAddr{
		IP:   IPv4(127, 0, 0, 1),
		Port: nextPort(),
	}
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
	listenersMu.Lock()
	l, ok := listeners[fd.raddr.(*TCPAddr).String()]
	if !ok {
		listenersMu.Unlock()
		return nil, syscall.ECONNREFUSED
	}
	l.incoming <- fd2
	listenersMu.Unlock()

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

	if fd.listener {
		listenersMu.Lock()
		delete(listeners, fd.laddr.String())
		close(fd.incoming)
		fd.listener = false
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

func (r *Resolver) lookup(ctx context.Context, name string, qtype dnsmessage.Type, conf *dnsConfig) (dnsmessage.Parser, string, error) {
	panic("unreachable")
}
