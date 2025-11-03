// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

package http

import (
	"context"
	"errors"
	"net"
	"syscall"
	"testing"
	"time"
)

// Receive a single HTTP request on the given server socket, then close recvCh.
// Wait for the context to close while handling this request, then record the
// context Cause and return it over the causeCh.
func getClosingCause(ln net.Listener, recvCh chan<-struct{}, causeCh chan<- error) {
	mux := NewServeMux()
	server := Server{
		Handler: mux,
	}

	mux.HandleFunc("/", func(w ResponseWriter, req *Request) {
		ctx := req.Context()
		close(recvCh)

		w.WriteHeader(200)

		select {
		case <-ctx.Done():
			causeCh <- context.Cause(ctx)
			close(causeCh)
			server.Close()
		}
	})

	server.Serve(ln)
}

func TestClientDisconnectedError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	recvCh := make(chan struct{})
	causeCh := make(chan error)
	go getClosingCause(ln, recvCh, causeCh)

	// Send a HTTP/1.1 request, then gracefully close the connection while it's being handled.
	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	conn.Write([]byte("GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"))
	<-recvCh

	if err := conn.(*net.TCPConn).Close(); err != nil {
		t.Fatal(err)
	}

	select {
	case cause := <-causeCh:
		if !errors.Is(cause, errClientDisconnected) {
			t.Fatalf("after graceful client disconnect, context cancellation cause wasn't 'client disconnected': %v", cause)
		}

	case <-ctx.Done():
		t.Fatal("never received a cause")
	}
}

func TestClientConnectionResetError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	
	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}

	recvCh := make(chan struct{})
	causeCh := make(chan error)
	go getClosingCause(ln, recvCh, causeCh)

	// Send a HTTP/1.1 request, then hard-reset connection while it's being handled.
	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	conn.Write([]byte("GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"))
	<-recvCh

	tcp := conn.(*net.TCPConn)
	if err := tcp.SetLinger(0); err != nil {
		t.Fatal(err)
	}
	if err := tcp.Close(); err != nil {
		t.Fatal(err)
	}

	select {
	case cause := <-causeCh:
		if !errors.Is(cause, errClientDisconnected) {
			t.Fatalf("after client connection reset, context cancellation cause wasn't 'client disconnected': %v", cause)
		}
		if !errors.Is(cause, syscall.ECONNRESET) {
			t.Fatalf("after client connection reset, context cancellation cause wasn't ECONNRESET: %v", cause)
		}

	case <-ctx.Done():
		t.Fatal("never received a cause")
	}
}
