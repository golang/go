// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The websocket package implements a client and server for the Web Socket protocol.
// The protocol is defined at http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol
package websocket

// TODO(ukai):
//   better logging.

import (
	"bufio"
	"io"
	"net"
	"os"
)

// WebSocketAddr is an implementation of net.Addr for Web Sockets.
type WebSocketAddr string

// Network returns the network type for a Web Socket, "websocket".
func (addr WebSocketAddr) Network() string { return "websocket" }

// String returns the network address for a Web Socket.
func (addr WebSocketAddr) String() string { return string(addr) }

// Conn is a channel to communicate to a Web Socket.
// It implements the net.Conn interface.
type Conn struct {
	// The origin URI for the Web Socket.
	Origin string
	// The location URI for the Web Socket.
	Location string
	// The subprotocol for the Web Socket.
	Protocol string

	buf *bufio.ReadWriter
	rwc io.ReadWriteCloser
}

// newConn creates a new Web Socket.
func newConn(origin, location, protocol string, buf *bufio.ReadWriter, rwc io.ReadWriteCloser) *Conn {
	if buf == nil {
		br := bufio.NewReader(rwc)
		bw := bufio.NewWriter(rwc)
		buf = bufio.NewReadWriter(br, bw)
	}
	ws := &Conn{origin, location, protocol, buf, rwc}
	return ws
}

// Read implements the io.Reader interface for a Conn.
func (ws *Conn) Read(msg []byte) (n int, err os.Error) {
	for {
		frameByte, err := ws.buf.ReadByte()
		if err != nil {
			return n, err
		}
		if (frameByte & 0x80) == 0x80 {
			length := 0
			for {
				c, err := ws.buf.ReadByte()
				if err != nil {
					return n, err
				}
				length = length*128 + int(c&0x7f)
				if (c & 0x80) == 0 {
					break
				}
			}
			for length > 0 {
				_, err := ws.buf.ReadByte()
				if err != nil {
					return n, err
				}
				length--
			}
		} else {
			for {
				c, err := ws.buf.ReadByte()
				if err != nil {
					return n, err
				}
				if c == '\xff' {
					return n, err
				}
				if frameByte == 0 {
					if n+1 <= cap(msg) {
						msg = msg[0 : n+1]
					}
					msg[n] = c
					n++
				}
				if n >= cap(msg) {
					return n, os.E2BIG
				}
			}
		}
	}

	panic("unreachable")
}

// Write implements the io.Writer interface for a Conn.
func (ws *Conn) Write(msg []byte) (n int, err os.Error) {
	ws.buf.WriteByte(0)
	ws.buf.Write(msg)
	ws.buf.WriteByte(0xff)
	err = ws.buf.Flush()
	return len(msg), err
}

// Close implements the io.Closer interface for a Conn.
func (ws *Conn) Close() os.Error { return ws.rwc.Close() }

// LocalAddr returns the WebSocket Origin for the connection.
func (ws *Conn) LocalAddr() net.Addr { return WebSocketAddr(ws.Origin) }

// RemoteAddr returns the WebSocket locations for the connection.
func (ws *Conn) RemoteAddr() net.Addr { return WebSocketAddr(ws.Location) }

// SetTimeout sets the connection's network timeout in nanoseconds.
func (ws *Conn) SetTimeout(nsec int64) os.Error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetTimeout(nsec)
	}
	return os.EINVAL
}

// SetReadTimeout sets the connection's network read timeout in nanoseconds.
func (ws *Conn) SetReadTimeout(nsec int64) os.Error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetReadTimeout(nsec)
	}
	return os.EINVAL
}

// SeWritetTimeout sets the connection's network write timeout in nanoseconds.
func (ws *Conn) SetWriteTimeout(nsec int64) os.Error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetWriteTimeout(nsec)
	}
	return os.EINVAL
}

var _ net.Conn = (*Conn)(nil) // compile-time check that *Conn implements net.Conn.
