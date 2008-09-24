// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"io";
	"bufio";
	"http";
	"os"
)

// Read/write/close interface.
type RWC interface {
	Read(p *[]byte) (n int, err *os.Error);
	Write(p *[]byte) (n int, err *os.Error);
	Close() *os.Error
}

// Active HTTP connection (server side).
export type Conn struct {
	rwc RWC;
	br *bufio.BufRead;
	bw *bufio.BufWrite;
	close bool;
	chunking bool;
}

// Create new connection from rwc.
export func NewConn(rwc RWC) (c *Conn, err *os.Error) {
	c = new(Conn);
	c.rwc = rwc;
	if c.br, err = bufio.NewBufRead(rwc); err != nil {
		return nil, err
	}
	if c.bw, err = bufio.NewBufWrite(rwc); err != nil {
		return nil, err
	}
	return c, nil
}

// Read next request from connection.
func (c *Conn) ReadRequest() (req *Request, err *os.Error) {
	if req, err = ReadRequest(c.br); err != nil {
		return nil, err
	}

	// TODO: Proper handling of (lack of) Connection: close,
	// and chunked transfer encoding on output.
	c.close = true
	return req, nil
}

// Close the connection.
func (c *Conn) Close() {
	c.bw.Flush();
	c.rwc.Close();
}

