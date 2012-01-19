// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"errors"
	"fmt"
	"io"
	"net"
	"time"
)

// Dial initiates a connection to the addr from the remote host.
// addr is resolved using net.ResolveTCPAddr before connection. 
// This could allow an observer to observe the DNS name of the 
// remote host. Consider using ssh.DialTCP to avoid this.
func (c *ClientConn) Dial(n, addr string) (net.Conn, error) {
	raddr, err := net.ResolveTCPAddr(n, addr)
	if err != nil {
		return nil, err
	}
	return c.DialTCP(n, nil, raddr)
}

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is used
// as the local address for the connection.
func (c *ClientConn) DialTCP(n string, laddr, raddr *net.TCPAddr) (net.Conn, error) {
	if laddr == nil {
		laddr = &net.TCPAddr{
			IP:   net.IPv4zero,
			Port: 0,
		}
	}
	ch, err := c.dial(laddr.IP.String(), laddr.Port, raddr.IP.String(), raddr.Port)
	if err != nil {
		return nil, err
	}
	return &tcpchanconn{
		tcpchan: ch,
		laddr:   laddr,
		raddr:   raddr,
	}, nil
}

// RFC 4254 7.2
type channelOpenDirectMsg struct {
	ChanType      string
	PeersId       uint32
	PeersWindow   uint32
	MaxPacketSize uint32
	raddr         string
	rport         uint32
	laddr         string
	lport         uint32
}

// dial opens a direct-tcpip connection to the remote server. laddr and raddr are passed as
// strings and are expected to be resolveable at the remote end.
func (c *ClientConn) dial(laddr string, lport int, raddr string, rport int) (*tcpchan, error) {
	ch := c.newChan(c.transport)
	if err := c.writePacket(marshal(msgChannelOpen, channelOpenDirectMsg{
		ChanType:      "direct-tcpip",
		PeersId:       ch.id,
		PeersWindow:   1 << 14,
		MaxPacketSize: 1 << 15, // RFC 4253 6.1
		raddr:         raddr,
		rport:         uint32(rport),
		laddr:         laddr,
		lport:         uint32(lport),
	})); err != nil {
		c.chanlist.remove(ch.id)
		return nil, err
	}
	if err := ch.waitForChannelOpenResponse(); err != nil {
		c.chanlist.remove(ch.id)
		return nil, fmt.Errorf("ssh: unable to open direct tcpip connection: %v", err)
	}
	return &tcpchan{
		clientChan: ch,
		Reader:     ch.stdout,
		Writer:     ch.stdin,
	}, nil
}

type tcpchan struct {
	*clientChan // the backing channel
	io.Reader
	io.Writer
}

// tcpchanconn fulfills the net.Conn interface without 
// the tcpchan having to hold laddr or raddr directly.
type tcpchanconn struct {
	*tcpchan
	laddr, raddr net.Addr
}

// LocalAddr returns the local network address.
func (t *tcpchanconn) LocalAddr() net.Addr {
	return t.laddr
}

// RemoteAddr returns the remote network address.
func (t *tcpchanconn) RemoteAddr() net.Addr {
	return t.raddr
}

// SetDeadline sets the read and write deadlines associated
// with the connection.
func (t *tcpchanconn) SetDeadline(deadline time.Time) error {
	if err := t.SetReadDeadline(deadline); err != nil {
		return err
	}
	return t.SetWriteDeadline(deadline)
}

// SetReadDeadline sets the read deadline.
// A zero value for t means Read will not time out.
// After the deadline, the error from Read will implement net.Error
// with Timeout() == true.
func (t *tcpchanconn) SetReadDeadline(deadline time.Time) error {
	return errors.New("ssh: tcpchan: deadline not supported")
}

// SetWriteDeadline exists to satisfy the net.Conn interface
// but is not implemented by this type.  It always returns an error.
func (t *tcpchanconn) SetWriteDeadline(deadline time.Time) error {
	return errors.New("ssh: tcpchan: deadline not supported")
}
