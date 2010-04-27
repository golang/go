// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package partially implements the TLS 1.1 protocol, as specified in RFC 4346.
package tls

import (
	"os"
	"net"
)

func Server(conn net.Conn, config *Config) *Conn {
	return &Conn{conn: conn, config: config}
}

func Client(conn net.Conn, config *Config) *Conn {
	return &Conn{conn: conn, config: config, isClient: true}
}

type Listener struct {
	listener net.Listener
	config   *Config
}

func (l *Listener) Accept() (c net.Conn, err os.Error) {
	c, err = l.listener.Accept()
	if err != nil {
		return
	}
	c = Server(c, l.config)
	return
}

func (l *Listener) Close() os.Error { return l.listener.Close() }

func (l *Listener) Addr() net.Addr { return l.listener.Addr() }

// NewListener creates a Listener which accepts connections from an inner
// Listener and wraps each connection with Server.
// The configuration config must be non-nil and must have
// at least one certificate.
func NewListener(listener net.Listener, config *Config) (l *Listener) {
	l = new(Listener)
	l.listener = listener
	l.config = config
	return
}

func Listen(network, laddr string, config *Config) (net.Listener, os.Error) {
	if config == nil || len(config.Certificates) == 0 {
		return nil, os.NewError("tls.Listen: no certificates in configuration")
	}
	l, err := net.Listen(network, laddr)
	if err != nil {
		return nil, err
	}
	return NewListener(l, config), nil
}

func Dial(network, laddr, raddr string) (net.Conn, os.Error) {
	c, err := net.Dial(network, laddr, raddr)
	if err != nil {
		return nil, err
	}
	return Client(c, nil), nil
}
