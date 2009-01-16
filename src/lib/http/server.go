// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trivial HTTP server

// TODO: Routines for writing responses.

package http

import (
	"io";
	"os";
	"net";
	"http";
	"strconv";
)

// Serve a new connection.
func ServeConnection(fd net.Conn, raddr string, f *(*Conn, *Request)) {
	c, err := NewConn(fd);
	if err != nil {
		return
	}
	for {
		req, err := c.ReadRequest();
		if err != nil {
			break
		}
		f(c, req);
		if c.close {
			break
		}
	}
	c.Close();
}

// Web server: already listening on l, call f for each request.
export func Serve(l net.Listener, f *(*Conn, *Request)) *os.Error {
	// TODO: Make this unnecessary
	s, e := os.Getenv("GOMAXPROCS");
	if n, ok := strconv.Atoi(s); n < 3 {
		print("Warning: $GOMAXPROCS needs to be at least 3.\n");
	}

	for {
		rw, raddr, e := l.Accept();
		if e != nil {
			return e
		}
		go ServeConnection(rw, raddr, f)
	}
	panic("not reached")
}

// Web server: listen on address, call f for each request.
export func ListenAndServe(addr string, f *(*Conn, *Request)) *os.Error {
	l, e := net.Listen("tcp", addr);
	if e != nil {
		return e
	}
	e = Serve(l, f);
	l.Close();
	return e
}
