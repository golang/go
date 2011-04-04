// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of Server

package httptest

import (
	"crypto/rand"
	"crypto/tls"
	"fmt"
	"http"
	"net"
	"os"
	"time"
)

// A Server is an HTTP server listening on a system-chosen port on the
// local loopback interface, for use in end-to-end HTTP tests.
type Server struct {
	URL      string // base URL of form http://ipaddr:port with no trailing slash
	Listener net.Listener
	TLS      *tls.Config // nil if not using using TLS
}

// historyListener keeps track of all connections that it's ever
// accepted.
type historyListener struct {
	net.Listener
	history []net.Conn
}

func (hs *historyListener) Accept() (c net.Conn, err os.Error) {
	c, err = hs.Listener.Accept()
	if err == nil {
		hs.history = append(hs.history, c)
	}
	return
}

func newLocalListener() net.Listener {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		if l, err = net.Listen("tcp6", "[::1]:0"); err != nil {
			panic(fmt.Sprintf("httptest: failed to listen on a port: %v", err))
		}
	}
	return l
}

// NewServer starts and returns a new Server.
// The caller should call Close when finished, to shut it down.
func NewServer(handler http.Handler) *Server {
	ts := new(Server)
	l := newLocalListener()
	ts.Listener = &historyListener{l, make([]net.Conn, 0)}
	ts.URL = "http://" + l.Addr().String()
	server := &http.Server{Handler: handler}
	go server.Serve(ts.Listener)
	return ts
}

// NewTLSServer starts and returns a new Server using TLS.
// The caller should call Close when finished, to shut it down.
func NewTLSServer(handler http.Handler) *Server {
	l := newLocalListener()
	ts := new(Server)

	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		panic(fmt.Sprintf("httptest: NewTLSServer: %v", err))
	}

	ts.TLS = &tls.Config{
		Rand:         rand.Reader,
		Time:         time.Seconds,
		NextProtos:   []string{"http/1.1"},
		Certificates: []tls.Certificate{cert},
	}
	tlsListener := tls.NewListener(l, ts.TLS)

	ts.Listener = &historyListener{tlsListener, make([]net.Conn, 0)}
	ts.URL = "https://" + l.Addr().String()
	server := &http.Server{Handler: handler}
	go server.Serve(ts.Listener)
	return ts
}

// Close shuts down the server.
func (s *Server) Close() {
	s.Listener.Close()
}

// CloseClientConnections closes any currently open HTTP connections
// to the test Server.
func (s *Server) CloseClientConnections() {
	hl, ok := s.Listener.(*historyListener)
	if !ok {
		return
	}
	for _, conn := range hl.history {
		conn.Close()
	}
}

// localhostCert is a PEM-encoded TLS cert with SAN DNS names
// "127.0.0.1" and "[::1]", expiring at the last second of 2049 (the end
// of ASN.1 time).
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBwTCCASugAwIBAgIBADALBgkqhkiG9w0BAQUwADAeFw0xMTAzMzEyMDI1MDda
Fw00OTEyMzEyMzU5NTlaMAAwggCdMAsGCSqGSIb3DQEBAQOCAIwAMIIAhwKCAIB6
oy4iT42G6qk+GGn5VL5JlnJT6ZG5cqaMNFaNGlIxNb6CPUZLKq2sM3gRaimsktIw
nNAcNwQGHpe1tZo+J/Pl04JTt71Y/TTAxy7OX27aZf1Rpt0SjdZ7vTPnFDPNsHGe
KBKvPt55l2+YKjkZmV7eRevsVbpkNvNGB+T5d4Ge/wIBA6NPME0wDgYDVR0PAQH/
BAQDAgCgMA0GA1UdDgQGBAQBAgMEMA8GA1UdIwQIMAaABAECAwQwGwYDVR0RBBQw
EoIJMTI3LjAuMC4xggVbOjoxXTALBgkqhkiG9w0BAQUDggCBAHC3gbdvc44vs+wD
g2kONiENnx8WKc0UTGg/TOXS3gaRb+CUIQtHWja65l8rAfclEovjHgZ7gx8brO0W
JuC6p3MUAKsgOssIrrRIx2rpnfcmFVMzguCmrMNVmKUAalw18Yp0F72xYAIitVQl
kJrLdIhBajcJRYu/YGltHQRaXuVt
-----END CERTIFICATE-----
`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBkgIBAQKCAIB6oy4iT42G6qk+GGn5VL5JlnJT6ZG5cqaMNFaNGlIxNb6CPUZL
Kq2sM3gRaimsktIwnNAcNwQGHpe1tZo+J/Pl04JTt71Y/TTAxy7OX27aZf1Rpt0S
jdZ7vTPnFDPNsHGeKBKvPt55l2+YKjkZmV7eRevsVbpkNvNGB+T5d4Ge/wIBAwKC
AIBRwh7Bil5Z8cYpZZv7jdQxDvbim7Z7ocRdeDmzZuF2I9RW04QyHHPIIlALnBvI
YeF1veASz1gEFGUjzmbUGqKYSbCoTzXoev+F4bmbRxcX9sOmtslqvhMSHRSzA5NH
aDVI3Hn4wvBVD8gePu8ACWqvPGbCiql11OKCMfjlPn2uuwJAx/24/F5DjXZ6hQQ7
HxScOxKrpx5WnA9r1wZTltOTZkhRRzuLc21WJeE3M15QUdWi3zZxCKRFoth65HEs
jy9YHQJAnPueRI44tz79b5QqVbeaOMUr7ZCb1Kp0uo6G+ANPLdlfliAupwij2eIz
mHRJOWk0jBtXfRft1McH2H51CpXAyw==
-----END RSA PRIVATE KEY-----
`)
