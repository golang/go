// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"internal/poll"
	"internal/syscall/windows"
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

func init() {
	poll.InitWSA()
}

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

	if ctx.Done() != nil {
		// Propagate the Context's deadline and cancellation.
		// If the context is already done, or if it has a nonzero deadline,
		// ensure that that is applied before the call to ConnectEx begins
		// so that we don't return spurious connections.
		defer fd.pfd.SetWriteDeadline(noDeadline)

		if ctx.Err() != nil {
			fd.pfd.SetWriteDeadline(aLongTimeAgo)
		} else {
			if deadline, ok := ctx.Deadline(); ok && !deadline.IsZero() {
				fd.pfd.SetWriteDeadline(deadline)
			}

			done := make(chan struct{})
			stop := context.AfterFunc(ctx, func() {
				// Force the runtime's poller to immediately give
				// up waiting for writability.
				fd.pfd.SetWriteDeadline(aLongTimeAgo)
				close(done)
			})
			defer func() {
				if !stop() {
					// Wait for the call to SetWriteDeadline to complete so that we can
					// reset the deadline if everything else succeeded.
					<-done
				}
			}()
		}
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

	var isloopback bool
	switch ra := ra.(type) {
	case *syscall.SockaddrInet4:
		isloopback = ra.Addr[0] == 127
	case *syscall.SockaddrInet6:
		isloopback = ra.Addr == [16]byte(IPv6loopback)
	default:
		panic("unexpected type in connect")
	}
	if isloopback {
		// This makes ConnectEx() fails faster if the target port on the localhost
		// is not reachable, instead of waiting for 2s.
		params := windows.TCP_INITIAL_RTO_PARAMETERS{
			Rtt:                   windows.TCP_INITIAL_RTO_UNSPECIFIED_RTT, // use the default or overridden by the Administrator
			MaxSynRetransmissions: 1,                                       // minimum possible value before Windows 10.0.16299
		}
		if windows.SupportTCPInitialRTONoSYNRetransmissions() {
			// In Windows 10.0.16299 TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS makes ConnectEx() fails instantly.
			params.MaxSynRetransmissions = windows.TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS
		}
		var out uint32
		// Don't abort the connection if WSAIoctl fails, as it is only an optimization.
		// If it fails reliably, we expect TestDialClosedPortFailFast to detect it.
		_ = fd.pfd.WSAIoctl(windows.SIO_TCP_INITIAL_RTO, (*byte)(unsafe.Pointer(&params)), uint32(unsafe.Sizeof(params)), nil, 0, &out, nil, 0)
	}

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
	s, rawsa, rsan, errcall, err := fd.pfd.Accept(func { sysSocket(fd.family, fd.sotype, 0) })

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
	// TODO: Implement this, perhaps using internal/poll.DupCloseOnExec.
	return nil, syscall.EWINDOWS
}
