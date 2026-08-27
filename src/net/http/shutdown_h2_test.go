// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"context"
	"crypto/tls"
	"internal/nettest"
	"io"
	. "net/http"
	"net/http/internal/testcert"
	"testing"
	"testing/synctest"
	"time"
)

// Issue 81153: Shutdown polls until its context expires if an HTTP/2 connection
// finishes its TLS handshake after Shutdown has run the http2 server's one-shot
// GOAWAY sweep
func TestServerShutdownH2ConnRegisteredDuringShutdown(t *testing.T) {
	synctest.Test(t, testServerShutdownH2ConnRegisteredDuringShutdown)
}
func testServerShutdownH2ConnRegisteredDuringShutdown(t *testing.T) {
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}

	inHandshake := make(chan struct{})      // closed when the server enters the TLS handshake
	releaseHandshake := make(chan struct{}) // closed to let the handshake complete

	srv := &Server{
		Handler: HandlerFunc(func(w ResponseWriter, r *Request) {}),
		TLSConfig: &tls.Config{
			GetCertificate: func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
				close(inHandshake)
				<-releaseHandshake
				return &cert, nil
			},
		},
	}
	defer srv.Close()
	listener := nettest.NewListener()
	defer listener.Close()
	go srv.ServeTLS(listener, "", "")

	// Start a TLS handshake (that will be stalled in GetCertificate).
	conn := listener.NewConn()
	defer conn.Close()
	tlsConn := tls.Client(conn, &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:         []string{"h2"},
	})
	handshakeDone := make(chan error, 1)
	go func() { handshakeDone <- tlsConn.Handshake() }()

	<-inHandshake // conn is accepted and mid-handshake but not yet registered with the http2 server

	shutdownDone := make(chan error, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	go func() { shutdownDone <- srv.Shutdown(ctx) }()

	// Wait for Shutdown's OnShutdown hook (the http2 server's one-shot
	// GOAWAY sweep, which sees no registered conns right now) to run.
	synctest.Wait()

	// Let the handshake finish. The connection negotiates h2 and registers
	// with the http2 server, which has already done its GOAWAY sweep.
	close(releaseHandshake)
	if err := <-handshakeDone; err != nil {
		t.Fatal(err)
	}

	// Speak enough HTTP/2 to be a healthy idle connection: client preface
	// plus an empty SETTINGS frame. Never open a stream.
	io.WriteString(tlsConn, "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
	tlsConn.Write([]byte{0, 0, 0, 0x4, 0, 0, 0, 0, 0}) // SETTINGS, len 0, stream 0
	go io.Copy(io.Discard, tlsConn)

	if err := <-shutdownDone; err != nil {
		t.Errorf("Shutdown did not drain the HTTP/2 connection: %v", err)
	}
}
