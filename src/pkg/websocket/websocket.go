// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The websocket package implements Web Socket protocol server.
package websocket

// References:
//   The Web Socket protocol: http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol

// TODO(ukai):
//   better logging.

import (
	"bufio";
	"io";
	"net";
	"os";
)

type WebSocketAddr string

func (addr WebSocketAddr) Network() string	{ return "websocket" }

func (addr WebSocketAddr) String() string	{ return string(addr) }

// Conn is an channels to communicate over Web Socket.
type Conn struct {
	// An origin URI of the Web Socket.
	Origin	string;
	// A location URI of the Web Socket.
	Location	string;
	// A subprotocol of the Web Socket.
	Protocol	string;

	buf	*bufio.ReadWriter;
	rwc	io.ReadWriteCloser;
}

// newConn creates a new Web Socket.
func newConn(origin, location, protocol string, buf *bufio.ReadWriter, rwc io.ReadWriteCloser) *Conn {
	if buf == nil {
		br := bufio.NewReader(rwc);
		bw := bufio.NewWriter(rwc);
		buf = bufio.NewReadWriter(br, bw);
	}
	ws := &Conn{origin, location, protocol, buf, rwc};
	return ws;
}

func (ws *Conn) Read(msg []byte) (n int, err os.Error) {
	for {
		frameByte, err := ws.buf.ReadByte();
		if err != nil {
			return n, err
		}
		if (frameByte & 0x80) == 0x80 {
			length := 0;
			for {
				c, err := ws.buf.ReadByte();
				if err != nil {
					return n, err
				}
				if (c & 0x80) == 0x80 {
					length = length*128 + int(c&0x7f)
				} else {
					break
				}
			}
			for length > 0 {
				_, err := ws.buf.ReadByte();
				if err != nil {
					return n, err
				}
				length--;
			}
		} else {
			for {
				c, err := ws.buf.ReadByte();
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
					msg[n] = c;
					n++;
				}
				if n >= cap(msg) {
					return n, os.E2BIG
				}
			}
		}
	}

	panic("unreachable");
}

func (ws *Conn) Write(msg []byte) (n int, err os.Error) {
	ws.buf.WriteByte(0);
	ws.buf.Write(msg);
	ws.buf.WriteByte(0xff);
	err = ws.buf.Flush();
	return len(msg), err;
}

func (ws *Conn) Close() os.Error	{ return ws.rwc.Close() }

func (ws *Conn) LocalAddr() net.Addr	{ return WebSocketAddr(ws.Origin) }

func (ws *Conn) RemoteAddr() net.Addr	{ return WebSocketAddr(ws.Location) }

func (ws *Conn) SetTimeout(nsec int64) os.Error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetTimeout(nsec)
	}
	return os.EINVAL;
}

func (ws *Conn) SetReadTimeout(nsec int64) os.Error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetReadTimeout(nsec)
	}
	return os.EINVAL;
}

func (ws *Conn) SetWriteTimeout(nsec int64) os.Error {
	if conn, ok := ws.rwc.(net.Conn); ok {
		return conn.SetWriteTimeout(nsec)
	}
	return os.EINVAL;
}

var _ net.Conn = (*Conn)(nil)	// compile-time check that *Conn implements net.Conn.
