// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2_test

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"testing"
	"time"

	jsonrpc2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/internal/stack/stacktest"
)

func TestIdleTimeout(t *testing.T) {
	stacktest.NoLeak(t)

	// Use a panicking time.AfterFunc instead of context.WithTimeout so that we
	// get a goroutine dump on failure. We expect the test to take on the order of
	// a few tens of milliseconds at most, so 10s should be several orders of
	// magnitude of headroom.
	timer := time.AfterFunc(10*time.Second, func() {
		debug.SetTraceback("all")
		panic("TestIdleTimeout deadlocked")
	})
	defer timer.Stop()

	ctx := context.Background()

	try := func(d time.Duration) (longEnough bool) {
		listener, err := jsonrpc2.NetListener(ctx, "tcp", "localhost:0", jsonrpc2.NetListenOptions{})
		if err != nil {
			t.Fatal(err)
		}

		idleStart := time.Now()
		listener = jsonrpc2.NewIdleListener(d, listener)
		defer listener.Close()

		server, err := jsonrpc2.Serve(ctx, listener, jsonrpc2.ConnectionOptions{})
		if err != nil {
			t.Fatal(err)
		}

		// Exercise some connection/disconnection patterns, and then assert that when
		// our timer fires, the server exits.
		conn1, err := jsonrpc2.Dial(ctx, listener.Dialer(), jsonrpc2.ConnectionOptions{})
		if err != nil {
			if since := time.Since(idleStart); since < d {
				t.Fatalf("conn1 failed to connect after %v: %v", since, err)
			}
			t.Log("jsonrpc2.Dial:", err)
			return false // Took to long to dial, so the failure could have been due to the idle timeout.
		}
		// On the server side, Accept can race with the connection timing out.
		// Send a call and wait for the response to ensure that the connection was
		// actually fully accepted.
		ac := conn1.Call(ctx, "ping", nil)
		if err := ac.Await(ctx, nil); !errors.Is(err, jsonrpc2.ErrMethodNotFound) {
			if since := time.Since(idleStart); since < d {
				t.Fatalf("conn1 broken after %v: %v", since, err)
			}
			t.Log(`conn1.Call(ctx, "ping", nil):`, err)
			conn1.Close()
			return false
		}

		conn2, err := jsonrpc2.Dial(ctx, listener.Dialer(), jsonrpc2.ConnectionOptions{})
		if err != nil {
			conn1.Close()
			if since := time.Since(idleStart); since < d {
				t.Fatalf("conn2 failed to connect while non-idle: %v", err)
			}
			t.Log("jsonrpc2.Dial:", err)
			return false
		}

		if err := conn1.Close(); err != nil {
			t.Fatalf("conn1.Close failed with error: %v", err)
		}
		idleStart = time.Now()
		if err := conn2.Close(); err != nil {
			t.Fatalf("conn2.Close failed with error: %v", err)
		}

		conn3, err := jsonrpc2.Dial(ctx, listener.Dialer(), jsonrpc2.ConnectionOptions{})
		if err != nil {
			if since := time.Since(idleStart); since < d {
				t.Fatalf("conn3 failed to connect after %v: %v", since, err)
			}
			t.Log("jsonrpc2.Dial:", err)
			return false // Took to long to dial, so the failure could have been due to the idle timeout.
		}

		ac = conn3.Call(ctx, "ping", nil)
		if err := ac.Await(ctx, nil); !errors.Is(err, jsonrpc2.ErrMethodNotFound) {
			if since := time.Since(idleStart); since < d {
				t.Fatalf("conn3 broken after %v: %v", since, err)
			}
			t.Log(`conn3.Call(ctx, "ping", nil):`, err)
			conn3.Close()
			return false
		}

		idleStart = time.Now()
		if err := conn3.Close(); err != nil {
			t.Fatalf("conn3.Close failed with error: %v", err)
		}

		serverError := server.Wait()

		if !errors.Is(serverError, jsonrpc2.ErrIdleTimeout) {
			t.Errorf("run() returned error %v, want %v", serverError, jsonrpc2.ErrIdleTimeout)
		}
		if since := time.Since(idleStart); since < d {
			t.Errorf("server shut down after %v idle; want at least %v", since, d)
		}
		return true
	}

	d := 1 * time.Millisecond
	for {
		t.Logf("testing with idle timout %v", d)
		if !try(d) {
			d *= 2
			continue
		}
		break
	}
}

type msg struct {
	Msg string
}

type fakeHandler struct{}

func (fakeHandler) Handle(ctx context.Context, req *jsonrpc2.Request) (interface{}, error) {
	switch req.Method {
	case "ping":
		return &msg{"pong"}, nil
	default:
		return nil, jsonrpc2.ErrNotHandled
	}
}

func TestServe(t *testing.T) {
	stacktest.NoLeak(t)
	ctx := context.Background()

	tests := []struct {
		name    string
		factory func(context.Context) (jsonrpc2.Listener, error)
	}{
		{"tcp", func(ctx context.Context) (jsonrpc2.Listener, error) {
			return jsonrpc2.NetListener(ctx, "tcp", "localhost:0", jsonrpc2.NetListenOptions{})
		}},
		{"pipe", func(ctx context.Context) (jsonrpc2.Listener, error) {
			return jsonrpc2.NetPipeListener(ctx)
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fake, err := test.factory(ctx)
			if err != nil {
				t.Fatal(err)
			}
			conn, shutdown, err := newFake(t, ctx, fake)
			if err != nil {
				t.Fatal(err)
			}
			defer shutdown()
			var got msg
			if err := conn.Call(ctx, "ping", &msg{"ting"}).Await(ctx, &got); err != nil {
				t.Fatal(err)
			}
			if want := "pong"; got.Msg != want {
				t.Errorf("conn.Call(...): returned %q, want %q", got, want)
			}
		})
	}
}

func newFake(t *testing.T, ctx context.Context, l jsonrpc2.Listener) (*jsonrpc2.Connection, func(), error) {
	server, err := jsonrpc2.Serve(ctx, l, jsonrpc2.ConnectionOptions{
		Handler: fakeHandler{},
	})
	if err != nil {
		return nil, nil, err
	}

	client, err := jsonrpc2.Dial(ctx,
		l.Dialer(),
		jsonrpc2.ConnectionOptions{
			Handler: fakeHandler{},
		})
	if err != nil {
		return nil, nil, err
	}
	return client, func() {
		if err := l.Close(); err != nil {
			t.Fatal(err)
		}
		if err := client.Close(); err != nil {
			t.Fatal(err)
		}
		server.Wait()
	}, nil
}

// TestIdleListenerAcceptCloseRace checks for the Accept/Close race fixed in CL 388597.
//
// (A bug in the idleListener implementation caused a successful Accept to block
// on sending to a background goroutine that could have already exited.)
func TestIdleListenerAcceptCloseRace(t *testing.T) {
	ctx := context.Background()

	n := 10

	// Each iteration of the loop appears to take around a millisecond, so to
	// avoid spurious failures we'll set the watchdog for three orders of
	// magnitude longer. When the bug was present, this reproduced the deadlock
	// reliably on a Linux workstation when run with -count=100, which should be
	// frequent enough to show up on the Go build dashboard if it regresses.
	watchdog := time.Duration(n) * 1000 * time.Millisecond
	timer := time.AfterFunc(watchdog, func() {
		debug.SetTraceback("all")
		panic(fmt.Sprintf("TestAcceptCloseRace deadlocked after %v", watchdog))
	})
	defer timer.Stop()

	for ; n > 0; n-- {
		listener, err := jsonrpc2.NetPipeListener(ctx)
		if err != nil {
			t.Fatal(err)
		}
		listener = jsonrpc2.NewIdleListener(24*time.Hour, listener)

		done := make(chan struct{})
		go func() {
			conn, err := jsonrpc2.Dial(ctx, listener.Dialer(), jsonrpc2.ConnectionOptions{})
			listener.Close()
			if err == nil {
				conn.Close()
			}
			close(done)
		}()

		// Accept may return a non-nil error if Close closes the underlying network
		// connection before the wrapped Accept call unblocks. However, it must not
		// deadlock!
		c, err := listener.Accept(ctx)
		if err == nil {
			c.Close()
		}
		<-done
	}
}
