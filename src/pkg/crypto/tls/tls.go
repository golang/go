// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package partially implements the TLS 1.1 protocol, as specified in RFC 4346.
package tls

import (
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"io/ioutil"
	"net"
	"os"
	"strings"
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

	colonPos := strings.LastIndex(raddr, ":")
	if colonPos == -1 {
		colonPos = len(raddr)
	}
	hostname := raddr[:colonPos]

	config := defaultConfig()
	config.ServerName = hostname
	conn := Client(c, config)
	err = conn.Handshake()
	if err == nil {
		return conn, nil
	}
	c.Close()
	return nil, err
}

// LoadX509KeyPair reads and parses a public/private key pair from a pair of
// files. The files must contain PEM encoded data.
func LoadX509KeyPair(certFile string, keyFile string) (cert Certificate, err os.Error) {
	certPEMBlock, err := ioutil.ReadFile(certFile)
	if err != nil {
		return
	}

	certDERBlock, _ := pem.Decode(certPEMBlock)
	if certDERBlock == nil {
		err = os.ErrorString("crypto/tls: failed to parse certificate PEM data")
		return
	}

	cert.Certificate = [][]byte{certDERBlock.Bytes}

	keyPEMBlock, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return
	}

	keyDERBlock, _ := pem.Decode(keyPEMBlock)
	if keyDERBlock == nil {
		err = os.ErrorString("crypto/tls: failed to parse key PEM data")
		return
	}

	key, err := x509.ParsePKCS1PrivateKey(keyDERBlock.Bytes)
	if err != nil {
		err = os.ErrorString("crypto/tls: failed to parse key")
		return
	}

	cert.PrivateKey = key

	// We don't need to parse the public key for TLS, but we so do anyway
	// to check that it looks sane and matches the private key.
	x509Cert, err := x509.ParseCertificate(certDERBlock.Bytes)
	if err != nil {
		return
	}

	if x509Cert.PublicKeyAlgorithm != x509.RSA || x509Cert.PublicKey.(*rsa.PublicKey).N.Cmp(key.PublicKey.N) != 0 {
		err = os.ErrorString("crypto/tls: private key does not match public key")
		return
	}

	return
}
