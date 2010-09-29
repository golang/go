// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"http"
	"io"
	"strings"
)

/*
Handler is an interface to a WebSocket.

A trivial example server:

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
			panic("ListenAndServe: " + err.String())
		}
	}
*/
type Handler func(*Conn)

/*
Gets key number from Sec-WebSocket-Key<n>: field as described
in 5.2 Sending the server's opening handshake, 4.
*/
func getKeyNumber(s string) (r uint32) {
	// 4. Let /key-number_n/ be the digits (characters in the range
	// U+0030 DIGIT ZERO (0) to U+0039 DIGIT NINE (9)) in /key_1/,
	// interpreted as a base ten integer, ignoring all other characters
	// in /key_n/.
	r = 0
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			r = r*10 + uint32(s[i]) - '0'
		}
	}
	return
}

// ServeHTTP implements the http.Handler interface for a Web Socket
func (f Handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	rwc, buf, err := w.Hijack()
	if err != nil {
		panic("Hijack failed: " + err.String())
		return
	}
	// The server should abort the WebSocket connection if it finds
	// the client did not send a handshake that matches with protocol
	// specification.
	defer rwc.Close()

	if req.Method != "GET" {
		return
	}
	// HTTP version can be safely ignored.

	if strings.ToLower(req.Header["Upgrade"]) != "websocket" ||
		strings.ToLower(req.Header["Connection"]) != "upgrade" {
		return
	}

	// TODO(ukai): check Host
	origin, found := req.Header["Origin"]
	if !found {
		return
	}

	key1, found := req.Header["Sec-Websocket-Key1"]
	if !found {
		return
	}
	key2, found := req.Header["Sec-Websocket-Key2"]
	if !found {
		return
	}
	key3 := make([]byte, 8)
	if _, err := io.ReadFull(buf, key3); err != nil {
		return
	}

	var location string
	if w.UsingTLS() {
		location = "wss://" + req.Host + req.URL.RawPath
	} else {
		location = "ws://" + req.Host + req.URL.RawPath
	}

	// Step 4. get key number in Sec-WebSocket-Key<n> fields.
	keyNumber1 := getKeyNumber(key1)
	keyNumber2 := getKeyNumber(key2)

	// Step 5. get number of spaces in Sec-WebSocket-Key<n> fields.
	space1 := uint32(strings.Count(key1, " "))
	space2 := uint32(strings.Count(key2, " "))
	if space1 == 0 || space2 == 0 {
		return
	}

	// Step 6. key number must be an integral multiple of spaces.
	if keyNumber1%space1 != 0 || keyNumber2%space2 != 0 {
		return
	}

	// Step 7. let part be key number divided by spaces.
	part1 := keyNumber1 / space1
	part2 := keyNumber2 / space2

	// Step 8. let challenge to be concatination of part1, part2 and key3.
	// Step 9. get MD5 fingerprint of challenge.
	response, err := getChallengeResponse(part1, part2, key3)
	if err != nil {
		return
	}

	// Step 10. send response status line.
	buf.WriteString("HTTP/1.1 101 WebSocket Protocol Handshake\r\n")
	// Step 11. send response headers.
	buf.WriteString("Upgrade: WebSocket\r\n")
	buf.WriteString("Connection: Upgrade\r\n")
	buf.WriteString("Sec-WebSocket-Location: " + location + "\r\n")
	buf.WriteString("Sec-WebSocket-Origin: " + origin + "\r\n")
	protocol, found := req.Header["Sec-Websocket-Protocol"]
	if found {
		buf.WriteString("Sec-WebSocket-Protocol: " + protocol + "\r\n")
	}
	// Step 12. send CRLF.
	buf.WriteString("\r\n")
	// Step 13. send response data.
	buf.Write(response)
	if err := buf.Flush(); err != nil {
		return
	}
	ws := newConn(origin, location, protocol, buf, rwc)
	f(ws)
}


/*
Draft75Handler is an interface to a WebSocket based on the
(soon obsolete) draft-hixie-thewebsocketprotocol-75.
*/
type Draft75Handler func(*Conn)

// ServeHTTP implements the http.Handler interface for a Web Socket.
func (f Draft75Handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != "GET" || req.Proto != "HTTP/1.1" {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "Unexpected request")
		return
	}
	if req.Header["Upgrade"] != "WebSocket" {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "missing Upgrade: WebSocket header")
		return
	}
	if req.Header["Connection"] != "Upgrade" {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "missing Connection: Upgrade header")
		return
	}
	origin, found := req.Header["Origin"]
	if !found {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "missing Origin header")
		return
	}

	rwc, buf, err := w.Hijack()
	if err != nil {
		panic("Hijack failed: " + err.String())
		return
	}
	defer rwc.Close()

	var location string
	if w.UsingTLS() {
		location = "wss://" + req.Host + req.URL.RawPath
	} else {
		location = "ws://" + req.Host + req.URL.RawPath
	}

	// TODO(ukai): verify origin,location,protocol.

	buf.WriteString("HTTP/1.1 101 Web Socket Protocol Handshake\r\n")
	buf.WriteString("Upgrade: WebSocket\r\n")
	buf.WriteString("Connection: Upgrade\r\n")
	buf.WriteString("WebSocket-Origin: " + origin + "\r\n")
	buf.WriteString("WebSocket-Location: " + location + "\r\n")
	protocol, found := req.Header["Websocket-Protocol"]
	// canonical header key of WebSocket-Protocol.
	if found {
		buf.WriteString("WebSocket-Protocol: " + protocol + "\r\n")
	}
	buf.WriteString("\r\n")
	if err := buf.Flush(); err != nil {
		return
	}
	ws := newConn(origin, location, protocol, buf, rwc)
	f(ws)
}
