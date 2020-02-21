// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"regexp"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/log"
)

type fakeClient struct {
	protocol.Client

	logs chan string
}

func (c fakeClient) LogMessage(ctx context.Context, params *protocol.LogMessageParams) error {
	c.logs <- params.Message
	return nil
}

type pingServer struct{ protocol.Server }

func (s pingServer) DidOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	log.Print(ctx, "ping")
	return nil
}

func (s pingServer) Shutdown(ctx context.Context) error {
	return nil
}

func TestClientLogging(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := pingServer{}
	client := fakeClient{logs: make(chan string, 10)}

	di := debug.NewInstance("", "")
	ss := NewStreamServer(cache.New(nil, di.State), false, di)
	ss.serverForTest = server
	ts := servertest.NewPipeServer(ctx, ss)
	defer ts.Close()
	cc := ts.Connect(ctx)
	cc.AddHandler(protocol.ClientHandler(client))

	protocol.ServerDispatcher(cc).DidOpen(ctx, &protocol.DidOpenTextDocumentParams{})

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

// waitableServer instruments LSP request so that we can control their timing.
// The requests chosen are arbitrary: we simply needed one that blocks, and
// another that doesn't.
type waitableServer struct {
	protocol.Server

	started chan struct{}
}

func (s waitableServer) Hover(ctx context.Context, _ *protocol.HoverParams) (*protocol.Hover, error) {
	s.started <- struct{}{}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(200 * time.Millisecond):
	}
	return &protocol.Hover{}, nil
}

func (s waitableServer) Resolve(_ context.Context, item *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return item, nil
}

func (s waitableServer) Shutdown(ctx context.Context) error {
	return nil
}

func TestRequestCancellation(t *testing.T) {
	server := waitableServer{
		started: make(chan struct{}),
	}
	diserve := debug.NewInstance("", "")
	ss := NewStreamServer(cache.New(nil, diserve.State), false, diserve)
	ss.serverForTest = server
	ctx := context.Background()
	tsDirect := servertest.NewTCPServer(ctx, ss)
	defer tsDirect.Close()

	forwarder := NewForwarder("tcp", tsDirect.Addr, false, debug.NewInstance("", ""))
	tsForwarded := servertest.NewPipeServer(ctx, forwarder)
	defer tsForwarded.Close()

	tests := []struct {
		serverType string
		ts         servertest.Connector
	}{
		{"direct", tsDirect},
		{"forwarder", tsForwarded},
	}

	for _, test := range tests {
		t.Run(test.serverType, func(t *testing.T) {
			cc := test.ts.Connect(ctx)
			cc.AddHandler(protocol.Canceller{})
			ctx := context.Background()
			ctx1, cancel1 := context.WithCancel(ctx)
			var (
				err1, err2 error
				wg         sync.WaitGroup
			)
			wg.Add(2)
			go func() {
				defer wg.Done()
				_, err1 = protocol.ServerDispatcher(cc).Hover(ctx1, &protocol.HoverParams{})
			}()
			go func() {
				defer wg.Done()
				_, err2 = protocol.ServerDispatcher(cc).Resolve(ctx, &protocol.CompletionItem{})
			}()
			// Wait for the Hover request to start.
			<-server.started
			cancel1()
			wg.Wait()
			if err1 == nil {
				t.Errorf("cancelled Hover(): got nil err")
			}
			if err2 != nil {
				t.Errorf("uncancelled Hover(): err: %v", err2)
			}
			if _, err := protocol.ServerDispatcher(cc).Resolve(ctx, &protocol.CompletionItem{}); err != nil {
				t.Errorf("subsequent Hover(): %v", err)
			}
		})
	}
}

const exampleProgram = `
-- go.mod --
module mod

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}`

func TestDebugInfoLifecycle(t *testing.T) {
	resetExitFuncs := OverrideExitFuncsForTest()
	defer resetExitFuncs()

	clientDebug := debug.NewInstance("", "")
	serverDebug := debug.NewInstance("", "")

	cache := cache.New(nil, serverDebug.State)
	ss := NewStreamServer(cache, false, serverDebug)
	ctx := context.Background()
	tsBackend := servertest.NewTCPServer(ctx, ss)

	forwarder := NewForwarder("tcp", tsBackend.Addr, false, clientDebug)
	tsForwarder := servertest.NewPipeServer(ctx, forwarder)

	ws, err := fake.NewWorkspace("gopls-lsprpc-test", []byte(exampleProgram))
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()

	conn1 := tsForwarder.Connect(ctx)
	ed1, err := fake.NewConnectedEditor(ctx, ws, conn1)
	if err != nil {
		t.Fatal(err)
	}
	defer ed1.Shutdown(ctx)
	conn2 := tsBackend.Connect(ctx)
	ed2, err := fake.NewConnectedEditor(ctx, ws, conn2)
	if err != nil {
		t.Fatal(err)
	}
	defer ed2.Shutdown(ctx)

	if got, want := len(serverDebug.State.Clients()), 2; got != want {
		t.Errorf("len(server:Clients) = %d, want %d", got, want)
	}
	if got, want := len(serverDebug.State.Sessions()), 2; got != want {
		t.Errorf("len(server:Sessions) = %d, want %d", got, want)
	}
	if got, want := len(clientDebug.State.Servers()), 1; got != want {
		t.Errorf("len(client:Servers) = %d, want %d", got, want)
	}
	// Close one of the connections to verify that the client and session were
	// dropped.
	if err := ed1.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if got, want := len(serverDebug.State.Sessions()), 1; got != want {
		t.Errorf("len(server:Sessions()) = %d, want %d", got, want)
	}
	// TODO(rfindley): once disconnection works, assert that len(Clients) == 1
	// (as of writing, it is still 2)
}

// TODO: add a test for telemetry.
