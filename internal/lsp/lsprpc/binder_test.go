// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rFindley): move this to the lsprpc_test package once it no longer
//                 shares with lsprpc_test.go.

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
	listener jsonrpc2_v2.Listener
	server   *jsonrpc2_v2.Server

	// non-nil if constructed with forwarded=true
	fwdListener jsonrpc2_v2.Listener
	fwdServer   *jsonrpc2_v2.Server

	// the ingoing connection, either to the forwarder or server
	conn *jsonrpc2_v2.Connection
}

func (e testEnv) Shutdown(t *testing.T) {
	if err := e.listener.Close(); err != nil {
		t.Error(err)
	}
	if e.fwdListener != nil {
		if err := e.fwdListener.Close(); err != nil {
			t.Error(err)
		}
	}
	if err := e.conn.Close(); err != nil {
		t.Error(err)
	}
	if e.fwdServer != nil {
		if err := e.fwdServer.Wait(); err != nil {
			t.Error(err)
		}
	}
	if err := e.server.Wait(); err != nil {
		t.Error(err)
	}
}

func startServing(ctx context.Context, t *testing.T, server protocol.Server, client protocol.Client, forwarded bool) testEnv {
	listener, err := jsonrpc2_v2.NetPipeListener(ctx)
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
	env := testEnv{
		listener: listener,
		server:   rpcServer,
	}
	clientBinder := NewClientBinder(func(context.Context, protocol.Server) protocol.Client { return client })
	if forwarded {
		fwdListener, err := jsonrpc2_v2.NetPipeListener(ctx)
		if err != nil {
			t.Fatal(err)
		}
		fwdBinder := NewForwardBinder(listener.Dialer())
		fwdServer, err := jsonrpc2_v2.Serve(ctx, fwdListener, fwdBinder)
		if err != nil {
			t.Fatal(err)
		}
		conn, err := jsonrpc2_v2.Dial(ctx, fwdListener.Dialer(), clientBinder)
		if err != nil {
			t.Fatal(err)
		}
		env.fwdListener = fwdListener
		env.fwdServer = fwdServer
		env.conn = conn
	} else {
		conn, err := jsonrpc2_v2.Dial(ctx, listener.Dialer(), clientBinder)
		if err != nil {
			t.Fatal(err)
		}
		env.conn = conn
	}
	return env
}

func TestClientLoggingV2(t *testing.T) {
	ctx := context.Background()

	for name, forwarded := range map[string]bool{
		"forwarded":  true,
		"standalone": false,
	} {
		t.Run(name, func(t *testing.T) {
			client := fakeClient{logs: make(chan string, 10)}
			env := startServing(ctx, t, pingServer{}, client, forwarded)
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
		})
	}
}

func TestRequestCancellationV2(t *testing.T) {
	ctx := context.Background()

	for name, forwarded := range map[string]bool{
		"forwarded":  true,
		"standalone": false,
	} {
		t.Run(name, func(t *testing.T) {
			server := waitableServer{
				started:   make(chan struct{}),
				completed: make(chan error),
			}
			client := fakeClient{logs: make(chan string, 10)}
			env := startServing(ctx, t, server, client, forwarded)
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
		})
	}
}
