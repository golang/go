// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"crypto/tls"
	"io"
	"net"
	"os"
	"url"
)

// DialError is an error that occurs while dialling a websocket server.
type DialError struct {
	*Config
	Error os.Error
}

func (e *DialError) String() string {
	return "websocket.Dial " + e.Config.Location.String() + ": " + e.Error.String()
}

// NewConfig creates a new WebSocket config for client connection.
func NewConfig(server, origin string) (config *Config, err os.Error) {
	config = new(Config)
	config.Version = ProtocolVersionHybi13
	config.Location, err = url.ParseRequest(server)
	if err != nil {
		return
	}
	config.Origin, err = url.ParseRequest(origin)
	if err != nil {
		return
	}
	return
}

// NewClient creates a new WebSocket client connection over rwc.
func NewClient(config *Config, rwc io.ReadWriteCloser) (ws *Conn, err os.Error) {
	br := bufio.NewReader(rwc)
	bw := bufio.NewWriter(rwc)
	switch config.Version {
	case ProtocolVersionHixie75:
		err = hixie75ClientHandshake(config, br, bw)
	case ProtocolVersionHixie76, ProtocolVersionHybi00:
		err = hixie76ClientHandshake(config, br, bw)
	case ProtocolVersionHybi08, ProtocolVersionHybi13:
		err = hybiClientHandshake(config, br, bw)
	default:
		err = ErrBadProtocolVersion
	}
	if err != nil {
		return
	}
	buf := bufio.NewReadWriter(br, bw)
	switch config.Version {
	case ProtocolVersionHixie75, ProtocolVersionHixie76, ProtocolVersionHybi00:
		ws = newHixieClientConn(config, buf, rwc)
	case ProtocolVersionHybi08, ProtocolVersionHybi13:
		ws = newHybiClientConn(config, buf, rwc)
	}
	return
}

/*
Dial opens a new client connection to a WebSocket.

A trivial example client:

	package main

	import (
		"http"
		"log"
		"strings"
		"websocket"
	)

	func main() {
		origin := "http://localhost/"
		url := "ws://localhost/ws" 
		ws, err := websocket.Dial(url, "", origin)
		if err != nil {
			log.Fatal(err)
		}
		if _, err := ws.Write([]byte("hello, world!\n")); err != nil {
			log.Fatal(err)
		}
		var msg = make([]byte, 512);
		if n, err := ws.Read(msg); err != nil {
			log.Fatal(err)
		}
		// use msg[0:n]
	}
*/
func Dial(url_, protocol, origin string) (ws *Conn, err os.Error) {
	config, err := NewConfig(url_, origin)
	if err != nil {
		return nil, err
	}
	return DialConfig(config)
}

// DialConfig opens a new client connection to a WebSocket with a config.
func DialConfig(config *Config) (ws *Conn, err os.Error) {
	var client net.Conn
	if config.Location == nil {
		return nil, &DialError{config, ErrBadWebSocketLocation}
	}
	if config.Origin == nil {
		return nil, &DialError{config, ErrBadWebSocketOrigin}
	}
	switch config.Location.Scheme {
	case "ws":
		client, err = net.Dial("tcp", config.Location.Host)

	case "wss":
		client, err = tls.Dial("tcp", config.Location.Host, config.TlsConfig)

	default:
		err = ErrBadScheme
	}
	if err != nil {
		goto Error
	}

	ws, err = NewClient(config, client)
	if err != nil {
		goto Error
	}
	return

Error:
	return nil, &DialError{config, err}
}
