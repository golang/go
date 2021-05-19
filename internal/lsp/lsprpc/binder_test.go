// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rFindley): move this to lsprpc_test once it no longer shares with
//                 lsprpc_test.go.

package lsprpc

import (
	"context"
	"regexp"
	"strings"
	"testing"
	"time"

	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/internal/lsp/protocol"
)

type testEnv struct {
	listener  jsonrpc2_v2.Listener
	conn      *jsonrpc2_v2.Connection
	rpcServer *jsonrpc2_v2.Server
}

func (e testEnv) Shutdown(t *testing.T) {
	if err := e.listener.Close(); err != nil {
		t.Error(err)
	}
	if err := e.conn.Close(); err != nil {
		t.Error(err)
	}
	if err := e.rpcServer.Wait(); err != nil {
		t.Error(err)
	}
}

func startServing(ctx context.Context, t *testing.T, server protocol.Server, client protocol.Client) testEnv {
	listener, err := jsonrpc2_v2.NetPipe(ctx)
	if err != nil {
		t.Fatal(err)
	}
	newServer := func(ctx context.Context, client protocol.ClientCloser) protocol.Server {
		return server
	}
	serverBinder := NewServerBinder(newServer)
	rpcServer, err := jsonrpc2_v2.Serve(ctx, listener, serverBinder)
	if err != nil {
		t.Fatal(err)
	}
	clientBinder := NewClientBinder(func(context.Context, protocol.Server) protocol.Client { return client })
	conn, err := jsonrpc2_v2.Dial(ctx, listener.Dialer(), clientBinder)
	if err != nil {
		t.Fatal(err)
	}
	return testEnv{
		listener:  listener,
		rpcServer: rpcServer,
		conn:      conn,
	}
}

func TestClientLoggingV2(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	client := fakeClient{logs: make(chan string, 10)}
	env := startServing(ctx, t, pingServer{}, client)
	defer env.Shutdown(t)
	if err := protocol.ServerDispatcherV2(env.conn).DidOpen(ctx, &protocol.DidOpenTextDocumentParams{}); err != nil {
		t.Errorf("DidOpen: %v", err)
	}
	select {
	case got := <-client.logs:
		want := "ping"
		matched, err := regexp.MatchString(want, got)
		if err != nil {
			t.Fatal(err)
		}
		if !matched {
			t.Errorf("got log %q, want a log containing %q", got, want)
		}
	case <-time.After(1 * time.Second):
		t.Error("timeout waiting for client log")
	}
}

func TestRequestCancellationV2(t *testing.T) {
	ctx := context.Background()

	server := waitableServer{
		started:   make(chan struct{}),
		completed: make(chan error),
	}
	client := fakeClient{logs: make(chan string, 10)}
	env := startServing(ctx, t, server, client)
	defer env.Shutdown(t)

	sd := protocol.ServerDispatcherV2(env.conn)
	ctx, cancel := context.WithCancel(ctx)

	result := make(chan error)
	go func() {
		_, err := sd.Hover(ctx, &protocol.HoverParams{})
		result <- err
	}()
	// Wait for the Hover request to start.
	<-server.started
	cancel()
	if err := <-result; err == nil {
		t.Error("nil error for cancelled Hover(), want non-nil")
	}
	if err := <-server.completed; err == nil || !strings.Contains(err.Error(), "cancelled hover") {
		t.Errorf("Hover(): unexpected server-side error %v", err)
	}
}
