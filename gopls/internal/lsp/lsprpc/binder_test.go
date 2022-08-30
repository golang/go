// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc_test

import (
	"context"
	"regexp"
	"strings"
	"testing"
	"time"

	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/gopls/internal/lsp/protocol"

	. "golang.org/x/tools/gopls/internal/lsp/lsprpc"
)

type TestEnv struct {
	Listeners []jsonrpc2_v2.Listener
	Conns     []*jsonrpc2_v2.Connection
	Servers   []*jsonrpc2_v2.Server
}

func (e *TestEnv) Shutdown(t *testing.T) {
	for _, l := range e.Listeners {
		if err := l.Close(); err != nil {
			t.Error(err)
		}
	}
	for _, c := range e.Conns {
		if err := c.Close(); err != nil {
			t.Error(err)
		}
	}
	for _, s := range e.Servers {
		if err := s.Wait(); err != nil {
			t.Error(err)
		}
	}
}

func (e *TestEnv) serve(ctx context.Context, t *testing.T, server jsonrpc2_v2.Binder) (jsonrpc2_v2.Listener, *jsonrpc2_v2.Server) {
	l, err := jsonrpc2_v2.NetPipeListener(ctx)
	if err != nil {
		t.Fatal(err)
	}
	e.Listeners = append(e.Listeners, l)
	s, err := jsonrpc2_v2.Serve(ctx, l, server)
	if err != nil {
		t.Fatal(err)
	}
	e.Servers = append(e.Servers, s)
	return l, s
}

func (e *TestEnv) dial(ctx context.Context, t *testing.T, dialer jsonrpc2_v2.Dialer, client jsonrpc2_v2.Binder, forwarded bool) *jsonrpc2_v2.Connection {
	if forwarded {
		l, _ := e.serve(ctx, t, NewForwardBinder(dialer))
		dialer = l.Dialer()
	}
	conn, err := jsonrpc2_v2.Dial(ctx, dialer, client)
	if err != nil {
		t.Fatal(err)
	}
	e.Conns = append(e.Conns, conn)
	return conn
}

func staticClientBinder(client protocol.Client) jsonrpc2_v2.Binder {
	f := func(context.Context, protocol.Server) protocol.Client { return client }
	return NewClientBinder(f)
}

func staticServerBinder(server protocol.Server) jsonrpc2_v2.Binder {
	f := func(ctx context.Context, client protocol.ClientCloser) protocol.Server {
		return server
	}
	return NewServerBinder(f)
}

func TestClientLoggingV2(t *testing.T) {
	ctx := context.Background()

	for name, forwarded := range map[string]bool{
		"forwarded":  true,
		"standalone": false,
	} {
		t.Run(name, func(t *testing.T) {
			client := FakeClient{Logs: make(chan string, 10)}
			env := new(TestEnv)
			defer env.Shutdown(t)
			l, _ := env.serve(ctx, t, staticServerBinder(PingServer{}))
			conn := env.dial(ctx, t, l.Dialer(), staticClientBinder(client), forwarded)

			if err := protocol.ServerDispatcherV2(conn).DidOpen(ctx, &protocol.DidOpenTextDocumentParams{}); err != nil {
				t.Errorf("DidOpen: %v", err)
			}
			select {
			case got := <-client.Logs:
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
			server := WaitableServer{
				Started:   make(chan struct{}),
				Completed: make(chan error),
			}
			env := new(TestEnv)
			defer env.Shutdown(t)
			l, _ := env.serve(ctx, t, staticServerBinder(server))
			client := FakeClient{Logs: make(chan string, 10)}
			conn := env.dial(ctx, t, l.Dialer(), staticClientBinder(client), forwarded)

			sd := protocol.ServerDispatcherV2(conn)
			ctx, cancel := context.WithCancel(ctx)

			result := make(chan error)
			go func() {
				_, err := sd.Hover(ctx, &protocol.HoverParams{})
				result <- err
			}()
			// Wait for the Hover request to start.
			<-server.Started
			cancel()
			if err := <-result; err == nil {
				t.Error("nil error for cancelled Hover(), want non-nil")
			}
			if err := <-server.Completed; err == nil || !strings.Contains(err.Error(), "cancelled hover") {
				t.Errorf("Hover(): unexpected server-side error %v", err)
			}
		})
	}
}
