// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fake networking for js/wasm. It is intended to allow tests of other package to pass.

//go:build js && wasm
// +build js,wasm

package net

import (
	"context"
	"internal/poll"
	"io"
	"os"
	"sync"
	"syscall"
	"time"
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

// Network file descriptor.
type netFD struct {
	r        *bufferedPipe
	w        *bufferedPipe
	incoming chan *netFD

	closedMu sync.Mutex
	closed   bool

	// immutable until Close
	listener bool
	family   int
	sotype   int
	net      string
	laddr    Addr
	raddr    Addr

	// unused
	pfd         poll.FD
	isConnected bool // handshake completed or use of association with peer
}

// socket returns a network file descriptor that is ready for
// asynchronous I/O using the network poller.
func socket(ctx context.Context, net string, family, sotype, proto int, ipv6only bool, laddr, raddr sockaddr, ctrlFn func(string, string, syscall.RawConn) error) (*netFD, error) {
	fd := &netFD{family: family, sotype: sotype, net: net}

	if laddr != nil && raddr == nil { // listener
		l := laddr.(*TCPAddr)
		fd.laddr = &TCPAddr{
			IP:   l.IP,
			Port: nextPort(),
			Zone: l.Zone,
		}
		fd.listener = true
		fd.incoming = make(chan *netFD, 1024)
		listenersMu.Lock()
		listeners[fd.laddr.(*TCPAddr).String()] = fd
		listenersMu.Unlock()
		return fd, nil
	}

	fd.laddr = &TCPAddr{
		IP:   IPv4(127, 0, 0, 1),
		Port: nextPort(),
	}
	fd.raddr = raddr
	fd.r = newBufferedPipe(65536)
	fd.w = newBufferedPipe(65536)

	fd2 := &netFD{family: fd.family, sotype: sotype, net: net}
	fd2.laddr = fd.raddr
	fd2.raddr = fd.laddr
	fd2.r = fd.w
	fd2.w = fd.r
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

func (fd *netFD) Read(p []byte) (n int, err error) {
	return fd.r.Read(p)
}

func (fd *netFD) Write(p []byte) (nn int, err error) {
	return fd.w.Write(p)
}

func (fd *netFD) Close() error {
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

func (fd *netFD) closeRead() error {
	fd.r.Close()
	return nil
}

func (fd *netFD) closeWrite() error {
	fd.w.Close()
	return nil
}

func (fd *netFD) accept() (*netFD, error) {
	c, ok := <-fd.incoming
	if !ok {
		return nil, syscall.EINVAL
	}
	return c, nil
}

func (fd *netFD) SetDeadline(t time.Time) error {
	fd.r.SetReadDeadline(t)
	fd.w.SetWriteDeadline(t)
	return nil
}

func (fd *netFD) SetReadDeadline(t time.Time) error {
	fd.r.SetReadDeadline(t)
	return nil
}

func (fd *netFD) SetWriteDeadline(t time.Time) error {
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
				return 0, syscall.EAGAIN
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
				return 0, syscall.EAGAIN
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

func (fd *netFD) readFrom(p []byte) (n int, sa syscall.Sockaddr, err error) {
	return 0, nil, syscall.ENOSYS
}

func (fd *netFD) readMsg(p []byte, oob []byte, flags int) (n, oobn, retflags int, sa syscall.Sockaddr, err error) {
	return 0, 0, 0, nil, syscall.ENOSYS
}

func (fd *netFD) writeTo(p []byte, sa syscall.Sockaddr) (n int, err error) {
	return 0, syscall.ENOSYS
}

func (fd *netFD) writeMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	return 0, 0, syscall.ENOSYS
}

func (fd *netFD) dup() (f *os.File, err error) {
	return nil, syscall.ENOSYS
}
