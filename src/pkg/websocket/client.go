// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"http"
	"io"
	"net"
	"os"
)

type ProtocolError struct {
	os.ErrorString
}

var (
	ErrBadStatus            = &ProtocolError{"bad status"}
	ErrNoUpgrade            = &ProtocolError{"no upgrade"}
	ErrBadUpgrade           = &ProtocolError{"bad upgrade"}
	ErrNoWebSocketOrigin    = &ProtocolError{"no WebSocket-Origin"}
	ErrBadWebSocketOrigin   = &ProtocolError{"bad WebSocket-Origin"}
	ErrNoWebSocketLocation  = &ProtocolError{"no WebSocket-Location"}
	ErrBadWebSocketLocation = &ProtocolError{"bad WebSocket-Location"}
	ErrNoWebSocketProtocol  = &ProtocolError{"no WebSocket-Protocol"}
	ErrBadWebSocketProtocol = &ProtocolError{"bad WebSocket-Protocol"}
)

// newClient creates a new Web Socket client connection.
func newClient(resourceName, host, origin, location, protocol string, rwc io.ReadWriteCloser) (ws *Conn, err os.Error) {
	br := bufio.NewReader(rwc)
	bw := bufio.NewWriter(rwc)
	err = handshake(resourceName, host, origin, location, protocol, br, bw)
	if err != nil {
		return
	}
	buf := bufio.NewReadWriter(br, bw)
	ws = newConn(origin, location, protocol, buf, rwc)
	return
}

/*
	Dial opens a new client connection to a Web Socket.
	A trivial example client is:

	package main

	import (
		"websocket"
		"strings"
	)

	func main() {
	 	ws, err := websocket.Dial("ws://localhost/ws", "", "http://localhost/");
	 	if err != nil {
			panic("Dial: ", err.String())
		}
		if _, err := ws.Write(strings.Bytes("hello, world!\n")); err != nil {
			panic("Write: ", err.String())
		}
		var msg = make([]byte, 512);
		if n, err := ws.Read(msg); err != nil {
			panic("Read: ", err.String())
		}
		// use msg[0:n]
	}
*/
func Dial(url, protocol, origin string) (ws *Conn, err os.Error) {
	parsedUrl, err := http.ParseURL(url)
	if err != nil {
		return
	}
	client, err := net.Dial("tcp", "", parsedUrl.Host)
	if err != nil {
		return
	}
	return newClient(parsedUrl.Path, parsedUrl.Host, origin, url, protocol, client)
}

func handshake(resourceName, host, origin, location, protocol string, br *bufio.Reader, bw *bufio.Writer) (err os.Error) {
	bw.WriteString("GET " + resourceName + " HTTP/1.1\r\n")
	bw.WriteString("Upgrade: WebSocket\r\n")
	bw.WriteString("Connection: Upgrade\r\n")
	bw.WriteString("Host: " + host + "\r\n")
	bw.WriteString("Origin: " + origin + "\r\n")
	if protocol != "" {
		bw.WriteString("WebSocket-Protocol: " + protocol + "\r\n")
	}
	bw.WriteString("\r\n")
	bw.Flush()
	resp, err := http.ReadResponse(br, "GET")
	if err != nil {
		return
	}
	if resp.Status != "101 Web Socket Protocol Handshake" {
		return ErrBadStatus
	}
	upgrade, found := resp.Header["Upgrade"]
	if !found {
		return ErrNoUpgrade
	}
	if upgrade != "WebSocket" {
		return ErrBadUpgrade
	}
	connection, found := resp.Header["Connection"]
	if !found || connection != "Upgrade" {
		return ErrBadUpgrade
	}

	ws_origin, found := resp.Header["Websocket-Origin"]
	if !found {
		return ErrNoWebSocketOrigin
	}
	if ws_origin != origin {
		return ErrBadWebSocketOrigin
	}
	ws_location, found := resp.Header["Websocket-Location"]
	if !found {
		return ErrNoWebSocketLocation
	}
	if ws_location != location {
		return ErrBadWebSocketLocation
	}
	if protocol != "" {
		ws_protocol, found := resp.Header["Websocket-Protocol"]
		if !found {
			return ErrNoWebSocketProtocol
		}
		if ws_protocol != protocol {
			return ErrBadWebSocketProtocol
		}
	}
	return
}
