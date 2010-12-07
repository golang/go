// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"bytes"
	"container/vector"
	"crypto/tls"
	"fmt"
	"http"
	"io"
	"net"
	"os"
	"rand"
	"strings"
)

type ProtocolError struct {
	os.ErrorString
}

var (
	ErrBadScheme            = os.ErrorString("bad scheme")
	ErrBadStatus            = &ProtocolError{"bad status"}
	ErrBadUpgrade           = &ProtocolError{"missing or bad upgrade"}
	ErrBadWebSocketOrigin   = &ProtocolError{"missing or bad WebSocket-Origin"}
	ErrBadWebSocketLocation = &ProtocolError{"missing or bad WebSocket-Location"}
	ErrBadWebSocketProtocol = &ProtocolError{"missing or bad WebSocket-Protocol"}
	ErrChallengeResponse    = &ProtocolError{"mismatch challange/response"}
	secKeyRandomChars       [0x30 - 0x21 + 0x7F - 0x3A]byte
)

type DialError struct {
	URL      string
	Protocol string
	Origin   string
	Error    os.Error
}

func (e *DialError) String() string {
	return "websocket.Dial " + e.URL + ": " + e.Error.String()
}

func init() {
	i := 0
	for ch := byte(0x21); ch < 0x30; ch++ {
		secKeyRandomChars[i] = ch
		i++
	}
	for ch := byte(0x3a); ch < 0x7F; ch++ {
		secKeyRandomChars[i] = ch
		i++
	}
}

type handshaker func(resourceName, host, origin, location, protocol string, br *bufio.Reader, bw *bufio.Writer) os.Error

// newClient creates a new Web Socket client connection.
func newClient(resourceName, host, origin, location, protocol string, rwc io.ReadWriteCloser, handshake handshaker) (ws *Conn, err os.Error) {
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

A trivial example client:

	package main

	import (
		"websocket"
		"strings"
	)

	func main() {
	 	ws, err := websocket.Dial("ws://localhost/ws", "", "http://localhost/");
	 	if err != nil {
			panic("Dial: " + err.String())
		}
		if _, err := ws.Write([]byte("hello, world!\n")); err != nil {
			panic("Write: " + err.String())
		}
		var msg = make([]byte, 512);
		if n, err := ws.Read(msg); err != nil {
			panic("Read: " + err.String())
		}
		// use msg[0:n]
	}
*/
func Dial(url, protocol, origin string) (ws *Conn, err os.Error) {
	var client net.Conn

	parsedUrl, err := http.ParseURL(url)
	if err != nil {
		goto Error
	}

	switch parsedUrl.Scheme {
	case "ws":
		client, err = net.Dial("tcp", "", parsedUrl.Host)

	case "wss":
		client, err = tls.Dial("tcp", "", parsedUrl.Host, nil)

	default:
		err = ErrBadScheme
	}
	if err != nil {
		goto Error
	}

	ws, err = newClient(parsedUrl.RawPath, parsedUrl.Host, origin, url, protocol, client, handshake)
	if err != nil {
		goto Error
	}
	return

Error:
	return nil, &DialError{url, protocol, origin, err}
}

/*
Generates handshake key as described in 4.1 Opening handshake step 16 to 22.
cf. http://www.whatwg.org/specs/web-socket-protocol/
*/
func generateKeyNumber() (key string, number uint32) {
	// 16.  Let /spaces_n/ be a random integer from 1 to 12 inclusive.
	spaces := rand.Intn(12) + 1

	// 17. Let /max_n/ be the largest integer not greater than
	//     4,294,967,295 divided by /spaces_n/
	max := int(4294967295 / uint32(spaces))

	// 18. Let /number_n/ be a random integer from 0 to /max_n/ inclusive.
	number = uint32(rand.Intn(max + 1))

	// 19. Let /product_n/ be the result of multiplying /number_n/ and
	//     /spaces_n/ together.
	product := number * uint32(spaces)

	// 20. Let /key_n/ be a string consisting of /product_n/, expressed
	// in base ten using the numerals in the range U+0030 DIGIT ZERO (0)
	// to U+0039 DIGIT NINE (9).
	key = fmt.Sprintf("%d", product)

	// 21. Insert between one and twelve random characters from the ranges
	//     U+0021 to U+002F and U+003A to U+007E into /key_n/ at random
	//     positions.
	n := rand.Intn(12) + 1
	for i := 0; i < n; i++ {
		pos := rand.Intn(len(key)) + 1
		ch := secKeyRandomChars[rand.Intn(len(secKeyRandomChars))]
		key = key[0:pos] + string(ch) + key[pos:]
	}

	// 22. Insert /spaces_n/ U+0020 SPACE characters into /key_n/ at random
	//     positions other than the start or end of the string.
	for i := 0; i < spaces; i++ {
		pos := rand.Intn(len(key)-1) + 1
		key = key[0:pos] + " " + key[pos:]
	}

	return
}

/*
Generates handshake key_3 as described in 4.1 Opening handshake step 26.
cf. http://www.whatwg.org/specs/web-socket-protocol/
*/
func generateKey3() (key []byte) {
	// 26. Let /key3/ be a string consisting of eight random bytes (or
	//  equivalently, a random 64 bit integer encoded in big-endian order).
	key = make([]byte, 8)
	for i := 0; i < 8; i++ {
		key[i] = byte(rand.Intn(256))
	}
	return
}

/*
Web Socket protocol handshake based on
http://www.whatwg.org/specs/web-socket-protocol/
(draft of http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol)
*/
func handshake(resourceName, host, origin, location, protocol string, br *bufio.Reader, bw *bufio.Writer) (err os.Error) {
	// 4.1. Opening handshake.
	// Step 5.  send a request line.
	bw.WriteString("GET " + resourceName + " HTTP/1.1\r\n")

	// Step 6-14. push request headers in fields.
	var fields vector.StringVector
	fields.Push("Upgrade: WebSocket\r\n")
	fields.Push("Connection: Upgrade\r\n")
	fields.Push("Host: " + host + "\r\n")
	fields.Push("Origin: " + origin + "\r\n")
	if protocol != "" {
		fields.Push("Sec-WebSocket-Protocol: " + protocol + "\r\n")
	}
	// TODO(ukai): Step 15. send cookie if any.

	// Step 16-23. generate keys and push Sec-WebSocket-Key<n> in fields.
	key1, number1 := generateKeyNumber()
	key2, number2 := generateKeyNumber()
	fields.Push("Sec-WebSocket-Key1: " + key1 + "\r\n")
	fields.Push("Sec-WebSocket-Key2: " + key2 + "\r\n")

	// Step 24. shuffle fields and send them out.
	for i := 1; i < len(fields); i++ {
		j := rand.Intn(i)
		fields[i], fields[j] = fields[j], fields[i]
	}
	for i := 0; i < len(fields); i++ {
		bw.WriteString(fields[i])
	}
	// Step 25. send CRLF.
	bw.WriteString("\r\n")

	// Step 26. genearte 8 bytes random key.
	key3 := generateKey3()
	// Step 27. send it out.
	bw.Write(key3)
	if err = bw.Flush(); err != nil {
		return
	}

	// Step 28-29, 32-40. read response from server.
	resp, err := http.ReadResponse(br, "GET")
	if err != nil {
		return err
	}
	// Step 30. check response code is 101.
	if resp.StatusCode != 101 {
		return ErrBadStatus
	}

	// Step 41. check websocket headers.
	if resp.Header["Upgrade"] != "WebSocket" ||
		strings.ToLower(resp.Header["Connection"]) != "upgrade" {
		return ErrBadUpgrade
	}

	if resp.Header["Sec-Websocket-Origin"] != origin {
		return ErrBadWebSocketOrigin
	}

	if resp.Header["Sec-Websocket-Location"] != location {
		return ErrBadWebSocketLocation
	}

	if protocol != "" && resp.Header["Sec-Websocket-Protocol"] != protocol {
		return ErrBadWebSocketProtocol
	}

	// Step 42-43. get expected data from challange data.
	expected, err := getChallengeResponse(number1, number2, key3)
	if err != nil {
		return err
	}

	// Step 44. read 16 bytes from server.
	reply := make([]byte, 16)
	if _, err = io.ReadFull(br, reply); err != nil {
		return err
	}

	// Step 45. check the reply equals to expected data.
	if !bytes.Equal(expected, reply) {
		return ErrChallengeResponse
	}
	// WebSocket connection is established.
	return
}

/*
Handhake described in (soon obsolete)
draft-hixie-thewebsocket-protocol-75.
*/
func draft75handshake(resourceName, host, origin, location, protocol string, br *bufio.Reader, bw *bufio.Writer) (err os.Error) {
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
	if resp.Header["Upgrade"] != "WebSocket" ||
		resp.Header["Connection"] != "Upgrade" {
		return ErrBadUpgrade
	}
	if resp.Header["Websocket-Origin"] != origin {
		return ErrBadWebSocketOrigin
	}
	if resp.Header["Websocket-Location"] != location {
		return ErrBadWebSocketLocation
	}
	if protocol != "" && resp.Header["Websocket-Protocol"] != protocol {
		return ErrBadWebSocketProtocol
	}
	return
}
