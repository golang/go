// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows

package net

import (
	"internal/poll"
	"runtime"
	"syscall"
	"time"
)

// Network file descriptor.
type netFD struct {
	pfd poll.FD

	// immutable until Close
	family      int
	sotype      int
	isConnected bool // handshake completed or use of association with peer
	net         string
	laddr       Addr
	raddr       Addr
}

func (fd *netFD) setAddr(laddr, raddr Addr) {
	fd.laddr = laddr
	fd.raddr = raddr
	runtime.SetFinalizer(fd, (*netFD).Close)
}

func (fd *netFD) Close() error {
	runtime.SetFinalizer(fd, nil)
	return fd.pfd.Close()
}

func (fd *netFD) shutdown(how int) error {
	err := fd.pfd.Shutdown(how)
	runtime.KeepAlive(fd)
	return wrapSyscallError("shutdown", err)
}

func (fd *netFD) closeRead() error {
	return fd.shutdown(syscall.SHUT_RD)
}

func (fd *netFD) closeWrite() error {
	return fd.shutdown(syscall.SHUT_WR)
}

func (fd *netFD) Read(p []byte) (n int, err error) {
	n, err = fd.pfd.Read(p)
	runtime.KeepAlive(fd)
	return n, wrapSyscallError(readSyscallName, err)
}

func (fd *netFD) readFrom(p []byte) (n int, sa syscall.Sockaddr, err error) {
	n, sa, err = fd.pfd.ReadFrom(p)
	runtime.KeepAlive(fd)
	return n, sa, wrapSyscallError(readFromSyscallName, err)
}
func (fd *netFD) readFromInet4(p []byte, from *syscall.SockaddrInet4) (n int, err error) {
	n, err = fd.pfd.ReadFromInet4(p, from)
	runtime.KeepAlive(fd)
	return n, wrapSyscallError(readFromSyscallName, err)
}

func (fd *netFD) readFromInet6(p []byte, from *syscall.SockaddrInet6) (n int, err error) {
	n, err = fd.pfd.ReadFromInet6(p, from)
	runtime.KeepAlive(fd)
	return n, wrapSyscallError(readFromSyscallName, err)
}

func (fd *netFD) readMsg(p []byte, oob []byte, flags int) (n, oobn, retflags int, sa syscall.Sockaddr, err error) {
	n, oobn, retflags, sa, err = fd.pfd.ReadMsg(p, oob, flags)
	runtime.KeepAlive(fd)
	return n, oobn, retflags, sa, wrapSyscallError(readMsgSyscallName, err)
}

func (fd *netFD) readMsgInet4(p []byte, oob []byte, flags int, sa *syscall.SockaddrInet4) (n, oobn, retflags int, err error) {
	n, oobn, retflags, err = fd.pfd.ReadMsgInet4(p, oob, flags, sa)
	runtime.KeepAlive(fd)
	return n, oobn, retflags, wrapSyscallError(readMsgSyscallName, err)
}

func (fd *netFD) readMsgInet6(p []byte, oob []byte, flags int, sa *syscall.SockaddrInet6) (n, oobn, retflags int, err error) {
	n, oobn, retflags, err = fd.pfd.ReadMsgInet6(p, oob, flags, sa)
	runtime.KeepAlive(fd)
	return n, oobn, retflags, wrapSyscallError(readMsgSyscallName, err)
}

func (fd *netFD) Write(p []byte) (nn int, err error) {
	nn, err = fd.pfd.Write(p)
	runtime.KeepAlive(fd)
	return nn, wrapSyscallError(writeSyscallName, err)
}

func (fd *netFD) writeTo(p []byte, sa syscall.Sockaddr) (n int, err error) {
	n, err = fd.pfd.WriteTo(p, sa)
	runtime.KeepAlive(fd)
	return n, wrapSyscallError(writeToSyscallName, err)
}

func (fd *netFD) writeToInet4(p []byte, sa *syscall.SockaddrInet4) (n int, err error) {
	n, err = fd.pfd.WriteToInet4(p, sa)
	runtime.KeepAlive(fd)
	return n, wrapSyscallError(writeToSyscallName, err)
}

func (fd *netFD) writeToInet6(p []byte, sa *syscall.SockaddrInet6) (n int, err error) {
	n, err = fd.pfd.WriteToInet6(p, sa)
	runtime.KeepAlive(fd)
	return n, wrapSyscallError(writeToSyscallName, err)
}

func (fd *netFD) writeMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	n, oobn, err = fd.pfd.WriteMsg(p, oob, sa)
	runtime.KeepAlive(fd)
	return n, oobn, wrapSyscallError(writeMsgSyscallName, err)
}

func (fd *netFD) writeMsgInet4(p []byte, oob []byte, sa *syscall.SockaddrInet4) (n int, oobn int, err error) {
	n, oobn, err = fd.pfd.WriteMsgInet4(p, oob, sa)
	runtime.KeepAlive(fd)
	return n, oobn, wrapSyscallError(writeMsgSyscallName, err)
}

func (fd *netFD) writeMsgInet6(p []byte, oob []byte, sa *syscall.SockaddrInet6) (n int, oobn int, err error) {
	n, oobn, err = fd.pfd.WriteMsgInet6(p, oob, sa)
	runtime.KeepAlive(fd)
	return n, oobn, wrapSyscallError(writeMsgSyscallName, err)
}

func (fd *netFD) SetDeadline(t time.Time) error {
	return fd.pfd.SetDeadline(t)
}

func (fd *netFD) SetReadDeadline(t time.Time) error {
	return fd.pfd.SetReadDeadline(t)
}

func (fd *netFD) SetWriteDeadline(t time.Time) error {
	return fd.pfd.SetWriteDeadline(t)
}
