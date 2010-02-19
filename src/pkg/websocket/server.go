// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"http"
	"io"
)

/*
	Handler is an interface to a WebSocket.
	A trivial example server is:

	package main

	import (
		"http"
		"io"
		"websocket"
	)

	// Echo the data received on the Web Socket.
	func EchoServer(ws *websocket.Conn) {
		io.Copy(ws, ws);
	}

	func main() {
		http.Handle("/echo", websocket.Handler(EchoServer));
		err := http.ListenAndServe(":12345", nil);
		if err != nil {
			panic("ListenAndServe: ", err.String())
		}
	}
*/
type Handler func(*Conn)

// ServeHTTP implements the http.Handler interface for a Web Socket.
func (f Handler) ServeHTTP(c *http.Conn, req *http.Request) {
	if req.Method != "GET" || req.Proto != "HTTP/1.1" {
		c.WriteHeader(http.StatusBadRequest)
		io.WriteString(c, "Unexpected request")
		return
	}
	if v, present := req.Header["Upgrade"]; !present || v != "WebSocket" {
		c.WriteHeader(http.StatusBadRequest)
		io.WriteString(c, "missing Upgrade: WebSocket header")
		return
	}
	if v, present := req.Header["Connection"]; !present || v != "Upgrade" {
		c.WriteHeader(http.StatusBadRequest)
		io.WriteString(c, "missing Connection: Upgrade header")
		return
	}
	origin, present := req.Header["Origin"]
	if !present {
		c.WriteHeader(http.StatusBadRequest)
		io.WriteString(c, "missing Origin header")
		return
	}

	rwc, buf, err := c.Hijack()
	if err != nil {
		panic("Hijack failed: ", err.String())
		return
	}
	defer rwc.Close()
	location := "ws://" + req.Host + req.URL.Path

	// TODO(ukai): verify origin,location,protocol.

	buf.WriteString("HTTP/1.1 101 Web Socket Protocol Handshake\r\n")
	buf.WriteString("Upgrade: WebSocket\r\n")
	buf.WriteString("Connection: Upgrade\r\n")
	buf.WriteString("WebSocket-Origin: " + origin + "\r\n")
	buf.WriteString("WebSocket-Location: " + location + "\r\n")
	protocol, present := req.Header["Websocket-Protocol"]
	// canonical header key of WebSocket-Protocol.
	if present {
		buf.WriteString("WebSocket-Protocol: " + protocol + "\r\n")
	}
	buf.WriteString("\r\n")
	if err := buf.Flush(); err != nil {
		return
	}
	ws := newConn(origin, location, protocol, buf, rwc)
	f(ws)
}
