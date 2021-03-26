// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2_test

import (
	"context"
	"errors"
	"testing"
	"time"

	jsonrpc2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/internal/stack/stacktest"
)

func TestIdleTimeout(t *testing.T) {
	stacktest.NoLeak(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	listener, err := jsonrpc2.NetListener(ctx, "tcp", "localhost:0", jsonrpc2.NetListenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	listener = jsonrpc2.NewIdleListener(100*time.Millisecond, listener)
	defer listener.Close()
	server, err := jsonrpc2.Serve(ctx, listener, jsonrpc2.ConnectionOptions{})
	if err != nil {
		t.Fatal(err)
	}

	connect := func() *jsonrpc2.Connection {
		client, err := jsonrpc2.Dial(ctx,
			listener.Dialer(),
			jsonrpc2.ConnectionOptions{})
		if err != nil {
			t.Fatal(err)
		}
		return client
	}
	// Exercise some connection/disconnection patterns, and then assert that when
	// our timer fires, the server exits.
	conn1 := connect()
	conn2 := connect()
	if err := conn1.Close(); err != nil {
		t.Fatalf("conn1.Close failed with error: %v", err)
	}
	if err := conn2.Close(); err != nil {
		t.Fatalf("conn2.Close failed with error: %v", err)
	}
	conn3 := connect()
	if err := conn3.Close(); err != nil {
		t.Fatalf("conn3.Close failed with error: %v", err)
	}

	serverError := server.Wait()

	if !errors.Is(serverError, jsonrpc2.ErrIdleTimeout) {
		t.Errorf("run() returned error %v, want %v", serverError, jsonrpc2.ErrIdleTimeout)
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
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tests := []struct {
		name    string
		factory func(context.Context) (jsonrpc2.Listener, error)
	}{
		{"tcp", func(ctx context.Context) (jsonrpc2.Listener, error) {
			return jsonrpc2.NetListener(ctx, "tcp", "localhost:0", jsonrpc2.NetListenOptions{})
		}},
		{"pipe", func(ctx context.Context) (jsonrpc2.Listener, error) {
			return jsonrpc2.NetPipe(ctx)
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fake, err := test.factory(ctx)
			if err != nil {
				t.Fatal(err)
			}
			conn, shutdown, err := newFake(ctx, fake)
			if err != nil {
				t.Fatal(err)
			}
			defer shutdown(ctx)
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

func newFake(ctx context.Context, l jsonrpc2.Listener) (*jsonrpc2.Connection, func(context.Context), error) {
	l = jsonrpc2.NewIdleListener(100*time.Millisecond, l)
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
	return client, func(ctx context.Context) {
		l.Close()
		client.Close()
		server.Wait()
	}, nil
}
