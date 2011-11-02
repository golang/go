// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"bytes"
	"fmt"
	"http"
	"io"
	"strings"
	"testing"
	"url"
)

// Test the getChallengeResponse function with values from section
// 5.1 of the specification steps 18, 26, and 43 from
// http://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-00
func TestHixie76Challenge(t *testing.T) {
	var part1 uint32 = 777007543
	var part2 uint32 = 114997259
	key3 := []byte{0x47, 0x30, 0x22, 0x2D, 0x5A, 0x3F, 0x47, 0x58}
	expected := []byte("0st3Rl&q-2ZU^weu")

	response, err := getChallengeResponse(part1, part2, key3)
	if err != nil {
		t.Errorf("getChallengeResponse: returned error %v", err)
		return
	}
	if !bytes.Equal(expected, response) {
		t.Errorf("getChallengeResponse: expected %q got %q", expected, response)
	}
}

func TestHixie76ClientHandshake(t *testing.T) {
	b := bytes.NewBuffer([]byte{})
	bw := bufio.NewWriter(b)
	br := bufio.NewReader(strings.NewReader(`HTTP/1.1 101 WebSocket Protocol Handshake
Upgrade: WebSocket
Connection: Upgrade
Sec-WebSocket-Origin: http://example.com
Sec-WebSocket-Location: ws://example.com/demo
Sec-WebSocket-Protocol: sample

8jKS'y:G*Co,Wxa-`))

	var err error
	config := new(Config)
	config.Location, err = url.ParseRequest("ws://example.com/demo")
	if err != nil {
		t.Fatal("location url", err)
	}
	config.Origin, err = url.ParseRequest("http://example.com")
	if err != nil {
		t.Fatal("origin url", err)
	}
	config.Protocol = append(config.Protocol, "sample")
	config.Version = ProtocolVersionHixie76

	config.handshakeData = map[string]string{
		"key1":    "4 @1  46546xW%0l 1 5",
		"number1": "829309203",
		"key2":    "12998 5 Y3 1  .P00",
		"number2": "259970620",
		"key3":    "^n:ds[4U",
	}
	err = hixie76ClientHandshake(config, br, bw)
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
	if req.URL.Path != "/demo" {
		t.Errorf("request path expected /demo, but got %q", req.URL.Path)
	}
	if req.Proto != "HTTP/1.1" {
		t.Errorf("request proto expected HTTP/1.1, but got %q", req.Proto)
	}
	if req.Host != "example.com" {
		t.Errorf("request Host expected example.com, but got %v", req.Host)
	}
	var expectedHeader = map[string]string{
		"Connection":             "Upgrade",
		"Upgrade":                "WebSocket",
		"Origin":                 "http://example.com",
		"Sec-Websocket-Key1":     config.handshakeData["key1"],
		"Sec-Websocket-Key2":     config.handshakeData["key2"],
		"Sec-WebSocket-Protocol": config.Protocol[0],
	}
	for k, v := range expectedHeader {
		if req.Header.Get(k) != v {
			t.Errorf(fmt.Sprintf("%s expected %q but got %q", k, v, req.Header.Get(k)))
		}
	}
}

func TestHixie76ServerHandshake(t *testing.T) {
	config := new(Config)
	handshaker := &hixie76ServerHandshaker{Config: config}
	br := bufio.NewReader(strings.NewReader(`GET /demo HTTP/1.1
Host: example.com
Connection: Upgrade
Sec-WebSocket-Key2: 12998 5 Y3 1  .P00
Sec-WebSocket-Protocol: sample
Upgrade: WebSocket
Sec-WebSocket-Key1: 4 @1  46546xW%0l 1 5
Origin: http://example.com

^n:ds[4U`))
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

	err = handshaker.AcceptHandshake(bw)
	if err != nil {
		t.Errorf("handshake response failed: %v", err)
	}
	expectedResponse := strings.Join([]string{
		"HTTP/1.1 101 WebSocket Protocol Handshake",
		"Upgrade: WebSocket",
		"Connection: Upgrade",
		"Sec-WebSocket-Origin: http://example.com",
		"Sec-WebSocket-Location: ws://example.com/demo",
		"Sec-WebSocket-Protocol: sample",
		"", ""}, "\r\n") + "8jKS'y:G*Co,Wxa-"
	if b.String() != expectedResponse {
		t.Errorf("handshake expected %q but got %q", expectedResponse, b.String())
	}
}

func TestHixie76SkipLengthFrame(t *testing.T) {
	b := []byte{'\x80', '\x01', 'x', 0, 'h', 'e', 'l', 'l', 'o', '\xff'}
	buf := bytes.NewBuffer(b)
	br := bufio.NewReader(buf)
	bw := bufio.NewWriter(buf)
	config := newConfig(t, "/")
	ws := newHixieConn(config, bufio.NewReadWriter(br, bw), nil, nil)
	msg := make([]byte, 5)
	n, err := ws.Read(msg)
	if err != nil {
		t.Errorf("Read: %v", err)
	}
	if !bytes.Equal(b[4:9], msg[0:n]) {
		t.Errorf("Read: expected %q got %q", b[4:9], msg[0:n])
	}
}

func TestHixie76SkipNoUTF8Frame(t *testing.T) {
	b := []byte{'\x01', 'n', '\xff', 0, 'h', 'e', 'l', 'l', 'o', '\xff'}
	buf := bytes.NewBuffer(b)
	br := bufio.NewReader(buf)
	bw := bufio.NewWriter(buf)
	config := newConfig(t, "/")
	ws := newHixieConn(config, bufio.NewReadWriter(br, bw), nil, nil)
	msg := make([]byte, 5)
	n, err := ws.Read(msg)
	if err != nil {
		t.Errorf("Read: %v", err)
	}
	if !bytes.Equal(b[4:9], msg[0:n]) {
		t.Errorf("Read: expected %q got %q", b[4:9], msg[0:n])
	}
}

func TestHixie76ClosingFrame(t *testing.T) {
	b := []byte{0, 'h', 'e', 'l', 'l', 'o', '\xff'}
	buf := bytes.NewBuffer(b)
	br := bufio.NewReader(buf)
	bw := bufio.NewWriter(buf)
	config := newConfig(t, "/")
	ws := newHixieConn(config, bufio.NewReadWriter(br, bw), nil, nil)
	msg := make([]byte, 5)
	n, err := ws.Read(msg)
	if err != nil {
		t.Errorf("read: %v", err)
	}
	if !bytes.Equal(b[1:6], msg[0:n]) {
		t.Errorf("Read: expected %q got %q", b[1:6], msg[0:n])
	}
	n, err = ws.Read(msg)
	if err != io.EOF {
		t.Errorf("read: %v", err)
	}
}
