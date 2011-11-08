// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

// Test the getNonceAccept function with values in
// http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-17
func TestSecWebSocketAccept(t *testing.T) {
	nonce := []byte("dGhlIHNhbXBsZSBub25jZQ==")
	expected := []byte("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")
	accept, err := getNonceAccept(nonce)
	if err != nil {
		t.Errorf("getNonceAccept: returned error %v", err)
		return
	}
	if !bytes.Equal(expected, accept) {
		t.Errorf("getNonceAccept: expected %q got %q", expected, accept)
	}
}

func TestHybiClientHandshake(t *testing.T) {
	b := bytes.NewBuffer([]byte{})
	bw := bufio.NewWriter(b)
	br := bufio.NewReader(strings.NewReader(`HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
Sec-WebSocket-Protocol: chat

`))
	var err error
	config := new(Config)
	config.Location, err = url.ParseRequest("ws://server.example.com/chat")
	if err != nil {
		t.Fatal("location url", err)
	}
	config.Origin, err = url.ParseRequest("http://example.com")
	if err != nil {
		t.Fatal("origin url", err)
	}
	config.Protocol = append(config.Protocol, "chat")
	config.Protocol = append(config.Protocol, "superchat")
	config.Version = ProtocolVersionHybi13

	config.handshakeData = map[string]string{
		"key": "dGhlIHNhbXBsZSBub25jZQ==",
	}
	err = hybiClientHandshake(config, br, bw)
	if err != nil {
		t.Errorf("handshake failed: %v", err)
	}
	req, err := http.ReadRequest(bufio.NewReader(b))
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	if req.Method != "GET" {
		t.Errorf("request method expected GET, but got %q", req.Method)
	}
	if req.URL.Path != "/chat" {
		t.Errorf("request path expected /chat, but got %q", req.URL.Path)
	}
	if req.Proto != "HTTP/1.1" {
		t.Errorf("request proto expected HTTP/1.1, but got %q", req.Proto)
	}
	if req.Host != "server.example.com" {
		t.Errorf("request Host expected server.example.com, but got %v", req.Host)
	}
	var expectedHeader = map[string]string{
		"Connection":             "Upgrade",
		"Upgrade":                "websocket",
		"Sec-Websocket-Key":      config.handshakeData["key"],
		"Origin":                 config.Origin.String(),
		"Sec-Websocket-Protocol": "chat, superchat",
		"Sec-Websocket-Version":  fmt.Sprintf("%d", ProtocolVersionHybi13),
	}
	for k, v := range expectedHeader {
		if req.Header.Get(k) != v {
			t.Errorf(fmt.Sprintf("%s expected %q but got %q", k, v, req.Header.Get(k)))
		}
	}
}

func TestHybiClientHandshakeHybi08(t *testing.T) {
	b := bytes.NewBuffer([]byte{})
	bw := bufio.NewWriter(b)
	br := bufio.NewReader(strings.NewReader(`HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
Sec-WebSocket-Protocol: chat

`))
	var err error
	config := new(Config)
	config.Location, err = url.ParseRequest("ws://server.example.com/chat")
	if err != nil {
		t.Fatal("location url", err)
	}
	config.Origin, err = url.ParseRequest("http://example.com")
	if err != nil {
		t.Fatal("origin url", err)
	}
	config.Protocol = append(config.Protocol, "chat")
	config.Protocol = append(config.Protocol, "superchat")
	config.Version = ProtocolVersionHybi08

	config.handshakeData = map[string]string{
		"key": "dGhlIHNhbXBsZSBub25jZQ==",
	}
	err = hybiClientHandshake(config, br, bw)
	if err != nil {
		t.Errorf("handshake failed: %v", err)
	}
	req, err := http.ReadRequest(bufio.NewReader(b))
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	if req.Method != "GET" {
		t.Errorf("request method expected GET, but got %q", req.Method)
	}
	if req.URL.Path != "/chat" {
		t.Errorf("request path expected /demo, but got %q", req.URL.Path)
	}
	if req.Proto != "HTTP/1.1" {
		t.Errorf("request proto expected HTTP/1.1, but got %q", req.Proto)
	}
	if req.Host != "server.example.com" {
		t.Errorf("request Host expected example.com, but got %v", req.Host)
	}
	var expectedHeader = map[string]string{
		"Connection":             "Upgrade",
		"Upgrade":                "websocket",
		"Sec-Websocket-Key":      config.handshakeData["key"],
		"Sec-Websocket-Origin":   config.Origin.String(),
		"Sec-Websocket-Protocol": "chat, superchat",
		"Sec-Websocket-Version":  fmt.Sprintf("%d", ProtocolVersionHybi08),
	}
	for k, v := range expectedHeader {
		if req.Header.Get(k) != v {
			t.Errorf(fmt.Sprintf("%s expected %q but got %q", k, v, req.Header.Get(k)))
		}
	}
}

func TestHybiServerHandshake(t *testing.T) {
	config := new(Config)
	handshaker := &hybiServerHandshaker{Config: config}
	br := bufio.NewReader(strings.NewReader(`GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Origin: http://example.com
Sec-WebSocket-Protocol: chat, superchat
Sec-WebSocket-Version: 13

`))
	req, err := http.ReadRequest(br)
	if err != nil {
		t.Fatal("request", err)
	}
	code, err := handshaker.ReadHandshake(br, req)
	if err != nil {
		t.Errorf("handshake failed: %v", err)
	}
	if code != http.StatusSwitchingProtocols {
		t.Errorf("status expected %q but got %q", http.StatusSwitchingProtocols, code)
	}
	b := bytes.NewBuffer([]byte{})
	bw := bufio.NewWriter(b)

	config.Protocol = []string{"chat"}

	err = handshaker.AcceptHandshake(bw)
	if err != nil {
		t.Errorf("handshake response failed: %v", err)
	}
	expectedResponse := strings.Join([]string{
		"HTTP/1.1 101 Switching Protocols",
		"Upgrade: websocket",
		"Connection: Upgrade",
		"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
		"Sec-WebSocket-Protocol: chat",
		"", ""}, "\r\n")

	if b.String() != expectedResponse {
		t.Errorf("handshake expected %q but got %q", expectedResponse, b.String())
	}
}

func TestHybiServerHandshakeHybi08(t *testing.T) {
	config := new(Config)
	handshaker := &hybiServerHandshaker{Config: config}
	br := bufio.NewReader(strings.NewReader(`GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Origin: http://example.com
Sec-WebSocket-Protocol: chat, superchat
Sec-WebSocket-Version: 8

`))
	req, err := http.ReadRequest(br)
	if err != nil {
		t.Fatal("request", err)
	}
	code, err := handshaker.ReadHandshake(br, req)
	if err != nil {
		t.Errorf("handshake failed: %v", err)
	}
	if code != http.StatusSwitchingProtocols {
		t.Errorf("status expected %q but got %q", http.StatusSwitchingProtocols, code)
	}
	b := bytes.NewBuffer([]byte{})
	bw := bufio.NewWriter(b)

	config.Protocol = []string{"chat"}

	err = handshaker.AcceptHandshake(bw)
	if err != nil {
		t.Errorf("handshake response failed: %v", err)
	}
	expectedResponse := strings.Join([]string{
		"HTTP/1.1 101 Switching Protocols",
		"Upgrade: websocket",
		"Connection: Upgrade",
		"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
		"Sec-WebSocket-Protocol: chat",
		"", ""}, "\r\n")

	if b.String() != expectedResponse {
		t.Errorf("handshake expected %q but got %q", expectedResponse, b.String())
	}
}

func TestHybiServerHandshakeHybiBadVersion(t *testing.T) {
	config := new(Config)
	handshaker := &hybiServerHandshaker{Config: config}
	br := bufio.NewReader(strings.NewReader(`GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Origin: http://example.com
Sec-WebSocket-Protocol: chat, superchat
Sec-WebSocket-Version: 9

`))
	req, err := http.ReadRequest(br)
	if err != nil {
		t.Fatal("request", err)
	}
	code, err := handshaker.ReadHandshake(br, req)
	if err != ErrBadWebSocketVersion {
		t.Errorf("handshake expected err %q but got %q", ErrBadWebSocketVersion, err)
	}
	if code != http.StatusBadRequest {
		t.Errorf("status expected %q but got %q", http.StatusBadRequest, code)
	}
}

func testHybiFrame(t *testing.T, testHeader, testPayload, testMaskedPayload []byte, frameHeader *hybiFrameHeader) {
	b := bytes.NewBuffer([]byte{})
	frameWriterFactory := &hybiFrameWriterFactory{bufio.NewWriter(b), false}
	w, _ := frameWriterFactory.NewFrameWriter(TextFrame)
	w.(*hybiFrameWriter).header = frameHeader
	_, err := w.Write(testPayload)
	w.Close()
	if err != nil {
		t.Errorf("Write error %q", err)
	}
	var expectedFrame []byte
	expectedFrame = append(expectedFrame, testHeader...)
	expectedFrame = append(expectedFrame, testMaskedPayload...)
	if !bytes.Equal(expectedFrame, b.Bytes()) {
		t.Errorf("frame expected %q got %q", expectedFrame, b.Bytes())
	}
	frameReaderFactory := &hybiFrameReaderFactory{bufio.NewReader(b)}
	r, err := frameReaderFactory.NewFrameReader()
	if err != nil {
		t.Errorf("Read error %q", err)
	}
	if header := r.HeaderReader(); header == nil {
		t.Errorf("no header")
	} else {
		actualHeader := make([]byte, r.Len())
		n, err := header.Read(actualHeader)
		if err != nil {
			t.Errorf("Read header error %q", err)
		} else {
			if n < len(testHeader) {
				t.Errorf("header too short %q got %q", testHeader, actualHeader[:n])
			}
			if !bytes.Equal(testHeader, actualHeader[:n]) {
				t.Errorf("header expected %q got %q", testHeader, actualHeader[:n])
			}
		}
	}
	if trailer := r.TrailerReader(); trailer != nil {
		t.Errorf("unexpected trailer %q", trailer)
	}
	frame := r.(*hybiFrameReader)
	if frameHeader.Fin != frame.header.Fin ||
		frameHeader.OpCode != frame.header.OpCode ||
		len(testPayload) != int(frame.header.Length) {
		t.Errorf("mismatch %v (%d) vs %v", frameHeader, len(testPayload), frame)
	}
	payload := make([]byte, len(testPayload))
	_, err = r.Read(payload)
	if err != nil {
		t.Errorf("read %v", err)
	}
	if !bytes.Equal(testPayload, payload) {
		t.Errorf("payload %q vs %q", testPayload, payload)
	}
}

func TestHybiShortTextFrame(t *testing.T) {
	frameHeader := &hybiFrameHeader{Fin: true, OpCode: TextFrame}
	payload := []byte("hello")
	testHybiFrame(t, []byte{0x81, 0x05}, payload, payload, frameHeader)

	payload = make([]byte, 125)
	testHybiFrame(t, []byte{0x81, 125}, payload, payload, frameHeader)
}

func TestHybiShortMaskedTextFrame(t *testing.T) {
	frameHeader := &hybiFrameHeader{Fin: true, OpCode: TextFrame,
		MaskingKey: []byte{0xcc, 0x55, 0x80, 0x20}}
	payload := []byte("hello")
	maskedPayload := []byte{0xa4, 0x30, 0xec, 0x4c, 0xa3}
	header := []byte{0x81, 0x85}
	header = append(header, frameHeader.MaskingKey...)
	testHybiFrame(t, header, payload, maskedPayload, frameHeader)
}

func TestHybiShortBinaryFrame(t *testing.T) {
	frameHeader := &hybiFrameHeader{Fin: true, OpCode: BinaryFrame}
	payload := []byte("hello")
	testHybiFrame(t, []byte{0x82, 0x05}, payload, payload, frameHeader)

	payload = make([]byte, 125)
	testHybiFrame(t, []byte{0x82, 125}, payload, payload, frameHeader)
}

func TestHybiControlFrame(t *testing.T) {
	frameHeader := &hybiFrameHeader{Fin: true, OpCode: PingFrame}
	payload := []byte("hello")
	testHybiFrame(t, []byte{0x89, 0x05}, payload, payload, frameHeader)

	frameHeader = &hybiFrameHeader{Fin: true, OpCode: PongFrame}
	testHybiFrame(t, []byte{0x8A, 0x05}, payload, payload, frameHeader)

	frameHeader = &hybiFrameHeader{Fin: true, OpCode: CloseFrame}
	payload = []byte{0x03, 0xe8} // 1000
	testHybiFrame(t, []byte{0x88, 0x02}, payload, payload, frameHeader)
}

func TestHybiLongFrame(t *testing.T) {
	frameHeader := &hybiFrameHeader{Fin: true, OpCode: TextFrame}
	payload := make([]byte, 126)
	testHybiFrame(t, []byte{0x81, 126, 0x00, 126}, payload, payload, frameHeader)

	payload = make([]byte, 65535)
	testHybiFrame(t, []byte{0x81, 126, 0xff, 0xff}, payload, payload, frameHeader)

	payload = make([]byte, 65536)
	testHybiFrame(t, []byte{0x81, 127, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00}, payload, payload, frameHeader)
}

func TestHybiClientRead(t *testing.T) {
	wireData := []byte{0x81, 0x05, 'h', 'e', 'l', 'l', 'o',
		0x89, 0x05, 'h', 'e', 'l', 'l', 'o', // ping
		0x81, 0x05, 'w', 'o', 'r', 'l', 'd'}
	br := bufio.NewReader(bytes.NewBuffer(wireData))
	bw := bufio.NewWriter(bytes.NewBuffer([]byte{}))
	conn := newHybiConn(newConfig(t, "/"), bufio.NewReadWriter(br, bw), nil, nil)

	msg := make([]byte, 512)
	n, err := conn.Read(msg)
	if err != nil {
		t.Errorf("read 1st frame, error %q", err)
	}
	if n != 5 {
		t.Errorf("read 1st frame, expect 5, got %d", n)
	}
	if !bytes.Equal(wireData[2:7], msg[:n]) {
		t.Errorf("read 1st frame %v, got %v", wireData[2:7], msg[:n])
	}
	n, err = conn.Read(msg)
	if err != nil {
		t.Errorf("read 2nd frame, error %q", err)
	}
	if n != 5 {
		t.Errorf("read 2nd frame, expect 5, got %d", n)
	}
	if !bytes.Equal(wireData[16:21], msg[:n]) {
		t.Errorf("read 2nd frame %v, got %v", wireData[16:21], msg[:n])
	}
	n, err = conn.Read(msg)
	if err == nil {
		t.Errorf("read not EOF")
	}
	if n != 0 {
		t.Errorf("expect read 0, got %d", n)
	}
}

func TestHybiShortRead(t *testing.T) {
	wireData := []byte{0x81, 0x05, 'h', 'e', 'l', 'l', 'o',
		0x89, 0x05, 'h', 'e', 'l', 'l', 'o', // ping
		0x81, 0x05, 'w', 'o', 'r', 'l', 'd'}
	br := bufio.NewReader(bytes.NewBuffer(wireData))
	bw := bufio.NewWriter(bytes.NewBuffer([]byte{}))
	conn := newHybiConn(newConfig(t, "/"), bufio.NewReadWriter(br, bw), nil, nil)

	step := 0
	pos := 0
	expectedPos := []int{2, 5, 16, 19}
	expectedLen := []int{3, 2, 3, 2}
	for {
		msg := make([]byte, 3)
		n, err := conn.Read(msg)
		if step >= len(expectedPos) {
			if err == nil {
				t.Errorf("read not EOF")
			}
			if n != 0 {
				t.Errorf("expect read 0, got %d", n)
			}
			return
		}
		pos = expectedPos[step]
		endPos := pos + expectedLen[step]
		if err != nil {
			t.Errorf("read from %d, got error %q", pos, err)
			return
		}
		if n != endPos-pos {
			t.Errorf("read from %d, expect %d, got %d", pos, endPos-pos, n)
		}
		if !bytes.Equal(wireData[pos:endPos], msg[:n]) {
			t.Errorf("read from %d, frame %v, got %v", pos, wireData[pos:endPos], msg[:n])
		}
		step++
	}
}

func TestHybiServerRead(t *testing.T) {
	wireData := []byte{0x81, 0x85, 0xcc, 0x55, 0x80, 0x20,
		0xa4, 0x30, 0xec, 0x4c, 0xa3, // hello
		0x89, 0x85, 0xcc, 0x55, 0x80, 0x20,
		0xa4, 0x30, 0xec, 0x4c, 0xa3, // ping: hello
		0x81, 0x85, 0xed, 0x83, 0xb4, 0x24,
		0x9a, 0xec, 0xc6, 0x48, 0x89, // world
	}
	br := bufio.NewReader(bytes.NewBuffer(wireData))
	bw := bufio.NewWriter(bytes.NewBuffer([]byte{}))
	conn := newHybiConn(newConfig(t, "/"), bufio.NewReadWriter(br, bw), nil, new(http.Request))

	expected := [][]byte{[]byte("hello"), []byte("world")}

	msg := make([]byte, 512)
	n, err := conn.Read(msg)
	if err != nil {
		t.Errorf("read 1st frame, error %q", err)
	}
	if n != 5 {
		t.Errorf("read 1st frame, expect 5, got %d", n)
	}
	if !bytes.Equal(expected[0], msg[:n]) {
		t.Errorf("read 1st frame %q, got %q", expected[0], msg[:n])
	}

	n, err = conn.Read(msg)
	if err != nil {
		t.Errorf("read 2nd frame, error %q", err)
	}
	if n != 5 {
		t.Errorf("read 2nd frame, expect 5, got %d", n)
	}
	if !bytes.Equal(expected[1], msg[:n]) {
		t.Errorf("read 2nd frame %q, got %q", expected[1], msg[:n])
	}

	n, err = conn.Read(msg)
	if err == nil {
		t.Errorf("read not EOF")
	}
	if n != 0 {
		t.Errorf("expect read 0, got %d", n)
	}
}

func TestHybiServerReadWithoutMasking(t *testing.T) {
	wireData := []byte{0x81, 0x05, 'h', 'e', 'l', 'l', 'o'}
	br := bufio.NewReader(bytes.NewBuffer(wireData))
	bw := bufio.NewWriter(bytes.NewBuffer([]byte{}))
	conn := newHybiConn(newConfig(t, "/"), bufio.NewReadWriter(br, bw), nil, new(http.Request))
	// server MUST close the connection upon receiving a non-masked frame.
	msg := make([]byte, 512)
	_, err := conn.Read(msg)
	if err != io.EOF {
		t.Errorf("read 1st frame, expect %q, but got %q", io.EOF, err)
	}
}

func TestHybiClientReadWithMasking(t *testing.T) {
	wireData := []byte{0x81, 0x85, 0xcc, 0x55, 0x80, 0x20,
		0xa4, 0x30, 0xec, 0x4c, 0xa3, // hello
	}
	br := bufio.NewReader(bytes.NewBuffer(wireData))
	bw := bufio.NewWriter(bytes.NewBuffer([]byte{}))
	conn := newHybiConn(newConfig(t, "/"), bufio.NewReadWriter(br, bw), nil, nil)

	// client MUST close the connection upon receiving a masked frame.
	msg := make([]byte, 512)
	_, err := conn.Read(msg)
	if err != io.EOF {
		t.Errorf("read 1st frame, expect %q, but got %q", io.EOF, err)
	}
}

// Test the hybiServerHandshaker supports firefox implementation and
// checks Connection request header include (but it's not necessary 
// equal to) "upgrade"   
func TestHybiServerFirefoxHandshake(t *testing.T) {
	config := new(Config)
	handshaker := &hybiServerHandshaker{Config: config}
	br := bufio.NewReader(strings.NewReader(`GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: keep-alive, upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Origin: http://example.com
Sec-WebSocket-Protocol: chat, superchat
Sec-WebSocket-Version: 13

`))
	req, err := http.ReadRequest(br)
	if err != nil {
		t.Fatal("request", err)
	}
	code, err := handshaker.ReadHandshake(br, req)
	if err != nil {
		t.Errorf("handshake failed: %v", err)
	}
	if code != http.StatusSwitchingProtocols {
		t.Errorf("status expected %q but got %q", http.StatusSwitchingProtocols, code)
	}
	b := bytes.NewBuffer([]byte{})
	bw := bufio.NewWriter(b)

	config.Protocol = []string{"chat"}

	err = handshaker.AcceptHandshake(bw)
	if err != nil {
		t.Errorf("handshake response failed: %v", err)
	}
	expectedResponse := strings.Join([]string{
		"HTTP/1.1 101 Switching Protocols",
		"Upgrade: websocket",
		"Connection: Upgrade",
		"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
		"Sec-WebSocket-Protocol: chat",
		"", ""}, "\r\n")

	if b.String() != expectedResponse {
		t.Errorf("handshake expected %q but got %q", expectedResponse, b.String())
	}
}
