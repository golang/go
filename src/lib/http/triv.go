// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io";
	"bufio";
	"os";
	"net";
	"http"
)

func Echo(conn *http.Conn, req *http.Request) {
	fd := conn.bw;
	conn.close = true;
	io.WriteString(fd, "HTTP/1.1 200 OK\r\n"
		"Content-Type: text/plain\r\n"
		"\r\n");
	io.WriteString(fd, req.method+" "+req.rawurl+" "+req.proto+"\r\n")
}

func main() {
	err := http.ListenAndServe("0.0.0.0:12345", &Echo)
	if err != nil {
		panic("ListenAndServe: ", err.String())
	}
}

