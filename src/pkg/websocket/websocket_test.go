// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bytes"
	"http"
	"io"
	"log"
	"net"
	"once"
	"strings"
	"testing"
)

var serverAddr string

func echoServer(ws *Conn) { io.Copy(ws, ws) }

func startServer() {
	l, e := net.Listen("tcp", ":0") // any available address
	if e != nil {
		log.Exitf("net.Listen tcp :0 %v", e)
	}
	serverAddr = l.Addr().String()
	log.Stderr("Test WebSocket server listening on ", serverAddr)
	http.Handle("/echo", Handler(echoServer))
	go http.Serve(l, nil)
}

func TestEcho(t *testing.T) {
	once.Do(startServer)

	client, err := net.Dial("tcp", "", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}

	ws, err := newClient("/echo", "localhost", "http://localhost",
		"ws://localhost/echo", "", client)
	if err != nil {
		t.Errorf("WebSocket handshake error", err)
		return
	}
	msg := strings.Bytes("hello, world\n")
	if _, err := ws.Write(msg); err != nil {
		t.Errorf("Write: error %v", err)
	}
	var actual_msg = make([]byte, 512)
	n, err := ws.Read(actual_msg)
	if err != nil {
		t.Errorf("Read: error %v", err)
	}
	actual_msg = actual_msg[0:n]
	if !bytes.Equal(msg, actual_msg) {
		t.Errorf("Echo: expected %q got %q", msg, actual_msg)
	}
	ws.Close()
}
