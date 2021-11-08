// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"internal/poll"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

const (
	readSyscallName     = "wsarecv"
	readFromSyscallName = "wsarecvfrom"
	readMsgSyscallName  = "wsarecvmsg"
	writeSyscallName    = "wsasend"
	writeToSyscallName  = "wsasendto"
	writeMsgSyscallName = "wsasendmsg"
)

// canUseConnectEx reports whether we can use the ConnectEx Windows API call
// for the given network type.
func canUseConnectEx(net string) bool {
	switch net {
	case "tcp", "tcp4", "tcp6":
		return true
	}
	// ConnectEx windows API does not support connectionless sockets.
	return false
}

func newFD(sysfd syscall.Handle, family, sotype int, net string) (*netFD, error) {
	ret := &netFD{
		pfd: poll.FD{
			Sysfd:         sysfd,
			IsStream:      sotype == syscall.SOCK_STREAM,
			ZeroReadIsEOF: sotype != syscall.SOCK_DGRAM && sotype != syscall.SOCK_RAW,
		},
		family: family,
		sotype: sotype,
		net:    net,
	}
	return ret, nil
}

func (fd *netFD) init() error {
	errcall, err := fd.pfd.Init(fd.net, true)
	if errcall != "" {
		err = wrapSyscallError(errcall, err)
	}
	return err
}

// Always returns nil for connected peer address result.
func (fd *netFD) connect(ctx context.Context, la, ra syscall.Sockaddr) (syscall.Sockaddr, error) {
	// Do not need to call fd.writeLock here,
	// because fd is not yet accessible to user,
	// so no concurrent operations are possible.
	if err := fd.init(); err != nil {
		return nil, err
	}
	if deadline, ok := ctx.Deadline(); ok && !deadline.IsZero() {
		fd.pfd.SetWriteDeadline(deadline)
		defer fd.pfd.SetWriteDeadline(noDeadline)
	}
	if !canUseConnectEx(fd.net) {
		err := connectFunc(fd.pfd.Sysfd, ra)
		return nil, os.NewSyscallError("connect", err)
	}
	// ConnectEx windows API requires an unconnected, previously bound socket.
	if la == nil {
		switch ra.(type) {
		case *syscall.SockaddrInet4:
			la = &syscall.SockaddrInet4{}
		case *syscall.SockaddrInet6:
			la = &syscall.SockaddrInet6{}
		default:
			panic("unexpected type in connect")
		}
		if err := syscall.Bind(fd.pfd.Sysfd, la); err != nil {
			return nil, os.NewSyscallError("bind", err)
		}
	}

	// Wait for the goroutine converting context.Done into a write timeout
	// to exist, otherwise our caller might cancel the context and
	// cause fd.setWriteDeadline(aLongTimeAgo) to cancel a successful dial.
	done := make(chan bool) // must be unbuffered
	defer func() { done <- true }()
	go func() {
		select {
		case <-ctx.Done():
			// Force the runtime's poller to immediately give
			// up waiting for writability.
			fd.pfd.SetWriteDeadline(aLongTimeAgo)
			<-done
		case <-done:
		}
	}()

	// Call ConnectEx API.
	if err := fd.pfd.ConnectEx(ra); err != nil {
		select {
		case <-ctx.Done():
			return nil, mapErr(ctx.Err())
		default:
			if _, ok := err.(syscall.Errno); ok {
				err = os.NewSyscallError("connectex", err)
			}
			return nil, err
		}
	}
	// Refresh socket properties.
	return nil, os.NewSyscallError("setsockopt", syscall.Setsockopt(fd.pfd.Sysfd, syscall.SOL_SOCKET, syscall.SO_UPDATE_CONNECT_CONTEXT, (*byte)(unsafe.Pointer(&fd.pfd.Sysfd)), int32(unsafe.Sizeof(fd.pfd.Sysfd))))
}

func (c *conn) writeBuffers(v *Buffers) (int64, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.fd.writeBuffers(v)
	if err != nil {
		return n, &OpError{Op: "wsasend", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, nil
}

func (fd *netFD) writeBuffers(buf *Buffers) (int64, error) {
	n, err := fd.pfd.Writev((*[][]byte)(buf))
	runtime.KeepAlive(fd)
	return n, wrapSyscallError("wsasend", err)
}

func (fd *netFD) accept() (*netFD, error) {
	s, rawsa, rsan, errcall, err := fd.pfd.Accept(func() (syscall.Handle, error) {
		return sysSocket(fd.family, fd.sotype, 0)
	})

	if err != nil {
		if errcall != "" {
			err = wrapSyscallError(errcall, err)
		}
		return nil, err
	}

	// Associate our new socket with IOCP.
	netfd, err := newFD(s, fd.family, fd.sotype, fd.net)
	if err != nil {
		poll.CloseFunc(s)
		return nil, err
	}
	if err := netfd.init(); err != nil {
		fd.Close()
		return nil, err
	}

	// Get local and peer addr out of AcceptEx buffer.
	var lrsa, rrsa *syscall.RawSockaddrAny
	var llen, rlen int32
	syscall.GetAcceptExSockaddrs((*byte)(unsafe.Pointer(&rawsa[0])),
		0, rsan, rsan, &lrsa, &llen, &rrsa, &rlen)
	lsa, _ := lrsa.Sockaddr()
	rsa, _ := rrsa.Sockaddr()

	netfd.setAddr(netfd.addrFunc()(lsa), netfd.addrFunc()(rsa))
	return netfd, nil
}

// Unimplemented functions.

func (fd *netFD) dup() (*os.File, error) {
	// TODO: Implement this
	return nil, syscall.EWINDOWS
}
