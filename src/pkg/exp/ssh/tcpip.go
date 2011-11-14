// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"errors"
	"io"
	"net"
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

// dial opens a direct-tcpip connection to the remote server. laddr and raddr are passed as
// strings and are expected to be resolveable at the remote end.
func (c *ClientConn) dial(laddr string, lport int, raddr string, rport int) (*tcpchan, error) {
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
	// wait for response
	switch msg := (<-ch.msg).(type) {
	case *channelOpenConfirmMsg:
		ch.peersId = msg.MyId
		ch.win <- int(msg.MyWindow)
	case *channelOpenFailureMsg:
		c.chanlist.remove(ch.id)
		return nil, errors.New("ssh: error opening remote TCP connection: " + msg.Message)
	default:
		c.chanlist.remove(ch.id)
		return nil, errors.New("ssh: unexpected packet")
	}
	return &tcpchan{
		clientChan: ch,
		Reader: &chanReader{
			packetWriter: ch,
			id:           ch.id,
			data:         ch.data,
		},
		Writer: &chanWriter{
			packetWriter: ch,
			id:           ch.id,
			win:          ch.win,
		},
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

// SetTimeout sets the read and write deadlines associated
// with the connection.
func (t *tcpchanconn) SetTimeout(nsec int64) error {
	if err := t.SetReadTimeout(nsec); err != nil {
		return err
	}
	return t.SetWriteTimeout(nsec)
}

// SetReadTimeout sets the time (in nanoseconds) that
// Read will wait for data before returning an error with Timeout() == true.
// Setting nsec == 0 (the default) disables the deadline.
func (t *tcpchanconn) SetReadTimeout(nsec int64) error {
	return errors.New("ssh: tcpchan: timeout not supported")
}

// SetWriteTimeout sets the time (in nanoseconds) that
// Write will wait to send its data before returning an error with Timeout() == true.
// Setting nsec == 0 (the default) disables the deadline.
// Even if write times out, it may return n > 0, indicating that
// some of the data was successfully written.
func (t *tcpchanconn) SetWriteTimeout(nsec int64) error {
	return errors.New("ssh: tcpchan: timeout not supported")
}
