// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"net"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/stack/stacktest"
	"golang.org/x/tools/internal/testenv"
)

func TestIdleTimeout(t *testing.T) {
	testenv.NeedsLocalhostNet(t)

	stacktest.NoLeak(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ln, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	connect := func() net.Conn {
		conn, err := net.DialTimeout("tcp", ln.Addr().String(), 5*time.Second)
		if err != nil {
			panic(err)
		}
		return conn
	}

	server := HandlerServer(MethodNotFound)
	// connTimer := &fakeTimer{c: make(chan time.Time, 1)}
	var (
		runErr error
		wg     sync.WaitGroup
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		runErr = Serve(ctx, ln, server, 100*time.Millisecond)
	}()

	// Exercise some connection/disconnection patterns, and then assert that when
	// our timer fires, the server exits.
	conn1 := connect()
	conn2 := connect()
	conn1.Close()
	conn2.Close()
	conn3 := connect()
	conn3.Close()

	wg.Wait()

	if runErr != ErrIdleTimeout {
		t.Errorf("run() returned error %v, want %v", runErr, ErrIdleTimeout)
	}
}
