// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"fmt"
	"http"
	"io"
)

func newServerConn(rwc io.ReadWriteCloser, buf *bufio.ReadWriter, req *http.Request) (conn *Conn, err error) {
	config := new(Config)
	var hs serverHandshaker = &hybiServerHandshaker{Config: config}
	code, err := hs.ReadHandshake(buf.Reader, req)
	if err == ErrBadWebSocketVersion {
		fmt.Fprintf(buf, "HTTP/1.1 %03d %s\r\n", code, http.StatusText(code))
		fmt.Fprintf(buf, "Sec-WebSocket-Version: %s\r\n", SupportedProtocolVersion)
		buf.WriteString("\r\n")
		buf.WriteString(err.Error())
		return
	}
	if err != nil {
		hs = &hixie76ServerHandshaker{Config: config}
		code, err = hs.ReadHandshake(buf.Reader, req)
	}
	if err != nil {
		hs = &hixie75ServerHandshaker{Config: config}
		code, err = hs.ReadHandshake(buf.Reader, req)
	}
	if err != nil {
		fmt.Fprintf(buf, "HTTP/1.1 %03d %s\r\n", code, http.StatusText(code))
		buf.WriteString("\r\n")
		buf.WriteString(err.Error())
		return
	}
	config.Protocol = nil

	err = hs.AcceptHandshake(buf.Writer)
	if err != nil {
		return
	}
	conn = hs.NewServerConn(buf, rwc, req)
	return
}

/*
Handler is an interface to a WebSocket.

A trivial example server:

	package main

	import (
		"http"
		"io"
		"websocket"
	)

	// Echo the data received on the WebSocket.
	func EchoServer(ws *websocket.Conn) {
		io.Copy(ws, ws);
	}

	func main() {
		http.Handle("/echo", websocket.Handler(EchoServer));
		err := http.ListenAndServe(":12345", nil);
		if err != nil {
			panic("ListenAndServe: " + err.String())
		}
	}
*/
type Handler func(*Conn)

// ServeHTTP implements the http.Handler interface for a Web Socket
func (h Handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	rwc, buf, err := w.(http.Hijacker).Hijack()
	if err != nil {
		panic("Hijack failed: " + err.Error())
		return
	}
	// The server should abort the WebSocket connection if it finds
	// the client did not send a handshake that matches with protocol
	// specification.
	defer rwc.Close()
	conn, err := newServerConn(rwc, buf, req)
	if err != nil {
		return
	}
	if conn == nil {
		panic("unepxected nil conn")
	}
	h(conn)
}
