// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/testenv"
)

type FakeClient struct {
	protocol.Client

	Logs chan string
}

func (c FakeClient) LogMessage(ctx context.Context, params *protocol.LogMessageParams) error {
	c.Logs <- params.Message
	return nil
}

// fakeServer is intended to be embedded in the test fakes below, to trivially
// implement Shutdown.
type fakeServer struct {
	protocol.Server
}

func (fakeServer) Shutdown(ctx context.Context) error {
	return nil
}

type PingServer struct{ fakeServer }

func (s PingServer) DidOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	event.Log(ctx, "ping")
	return nil
}

func TestClientLogging(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := PingServer{}
	client := FakeClient{Logs: make(chan string, 10)}

	ctx = debug.WithInstance(ctx, "", "")
	ss := NewStreamServer(cache.New(nil, nil, nil), false)
	ss.serverForTest = server
	ts := servertest.NewPipeServer(ss, nil)
	defer checkClose(t, ts.Close)
	cc := ts.Connect(ctx)
	cc.Go(ctx, protocol.ClientHandler(client, jsonrpc2.MethodNotFound))

	if err := protocol.ServerDispatcher(cc).DidOpen(ctx, &protocol.DidOpenTextDocumentParams{}); err != nil {
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
}

// WaitableServer instruments LSP request so that we can control their timing.
// The requests chosen are arbitrary: we simply needed one that blocks, and
// another that doesn't.
type WaitableServer struct {
	fakeServer

	Started   chan struct{}
	Completed chan error
}

func (s WaitableServer) Hover(ctx context.Context, _ *protocol.HoverParams) (_ *protocol.Hover, err error) {
	s.Started <- struct{}{}
	defer func() {
		s.Completed <- err
	}()
	select {
	case <-ctx.Done():
		return nil, errors.New("cancelled hover")
	case <-time.After(10 * time.Second):
	}
	return &protocol.Hover{}, nil
}

func (s WaitableServer) ResolveCompletionItem(_ context.Context, item *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return item, nil
}

func checkClose(t *testing.T, closer func() error) {
	t.Helper()
	if err := closer(); err != nil {
		t.Errorf("closing: %v", err)
	}
}

func setupForwarding(ctx context.Context, t *testing.T, s protocol.Server) (direct, forwarded servertest.Connector, cleanup func()) {
	t.Helper()
	serveCtx := debug.WithInstance(ctx, "", "")
	ss := NewStreamServer(cache.New(nil, nil, nil), false)
	ss.serverForTest = s
	tsDirect := servertest.NewTCPServer(serveCtx, ss, nil)

	forwarder, err := NewForwarder("tcp;"+tsDirect.Addr, nil)
	if err != nil {
		t.Fatal(err)
	}
	tsForwarded := servertest.NewPipeServer(forwarder, nil)
	return tsDirect, tsForwarded, func() {
		checkClose(t, tsDirect.Close)
		checkClose(t, tsForwarded.Close)
	}
}

func TestRequestCancellation(t *testing.T) {
	ctx := context.Background()
	server := WaitableServer{
		Started:   make(chan struct{}),
		Completed: make(chan error),
	}
	tsDirect, tsForwarded, cleanup := setupForwarding(ctx, t, server)
	defer cleanup()
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
			sd := protocol.ServerDispatcher(cc)
			cc.Go(ctx,
				protocol.Handlers(
					jsonrpc2.MethodNotFound))

			ctx := context.Background()
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
	sb, err := fake.NewSandbox(&fake.SandboxConfig{Files: fake.UnpackTxt(exampleProgram)})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sb.Close(); err != nil {
			// TODO(golang/go#38490): we can't currently make this an error because
			// it fails on Windows: the workspace directory is still locked by a
			// separate Go process.
			// Once we have a reliable way to wait for proper shutdown, make this an
			// error.
			t.Logf("closing workspace failed: %v", err)
		}
	}()

	baseCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	clientCtx := debug.WithInstance(baseCtx, "", "")
	serverCtx := debug.WithInstance(baseCtx, "", "")

	cache := cache.New(nil, nil, nil)
	ss := NewStreamServer(cache, false)
	tsBackend := servertest.NewTCPServer(serverCtx, ss, nil)

	forwarder, err := NewForwarder("tcp;"+tsBackend.Addr, nil)
	if err != nil {
		t.Fatal(err)
	}
	tsForwarder := servertest.NewPipeServer(forwarder, nil)

	ed1, err := fake.NewEditor(sb, fake.EditorConfig{}).Connect(clientCtx, tsForwarder, fake.ClientHooks{})
	if err != nil {
		t.Fatal(err)
	}
	defer ed1.Close(clientCtx)
	ed2, err := fake.NewEditor(sb, fake.EditorConfig{}).Connect(baseCtx, tsBackend, fake.ClientHooks{})
	if err != nil {
		t.Fatal(err)
	}
	defer ed2.Close(baseCtx)

	serverDebug := debug.GetInstance(serverCtx)
	if got, want := len(serverDebug.State.Clients()), 2; got != want {
		t.Errorf("len(server:Clients) = %d, want %d", got, want)
	}
	if got, want := len(serverDebug.State.Sessions()), 2; got != want {
		t.Errorf("len(server:Sessions) = %d, want %d", got, want)
	}
	clientDebug := debug.GetInstance(clientCtx)
	if got, want := len(clientDebug.State.Servers()), 1; got != want {
		t.Errorf("len(client:Servers) = %d, want %d", got, want)
	}
	// Close one of the connections to verify that the client and session were
	// dropped.
	if err := ed1.Close(clientCtx); err != nil {
		t.Fatal(err)
	}
	/*TODO: at this point we have verified the editor is closed
	However there is no way currently to wait for all associated go routines to
	go away, and we need to wait for those to trigger the client drop
	for now we just give it a little bit of time, but we need to fix this
	in a principled way
	*/
	start := time.Now()
	delay := time.Millisecond
	const maxWait = time.Second
	for len(serverDebug.State.Clients()) > 1 {
		if time.Since(start) > maxWait {
			break
		}
		time.Sleep(delay)
		delay *= 2
	}
	if got, want := len(serverDebug.State.Clients()), 1; got != want {
		t.Errorf("len(server:Clients) = %d, want %d", got, want)
	}
	if got, want := len(serverDebug.State.Sessions()), 1; got != want {
		t.Errorf("len(server:Sessions()) = %d, want %d", got, want)
	}
}

type initServer struct {
	fakeServer

	params *protocol.ParamInitialize
}

func (s *initServer) Initialize(ctx context.Context, params *protocol.ParamInitialize) (*protocol.InitializeResult, error) {
	s.params = params
	return &protocol.InitializeResult{}, nil
}

func TestEnvForwarding(t *testing.T) {
	testenv.NeedsGo1Point(t, 13)
	server := &initServer{}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, tsForwarded, cleanup := setupForwarding(ctx, t, server)
	defer cleanup()

	conn := tsForwarded.Connect(ctx)
	conn.Go(ctx, jsonrpc2.MethodNotFound)
	dispatch := protocol.ServerDispatcher(conn)
	initParams := &protocol.ParamInitialize{}
	initParams.InitializationOptions = map[string]interface{}{
		"env": map[string]interface{}{
			"GONOPROXY": "example.com",
		},
	}
	_, err := dispatch.Initialize(ctx, initParams)
	if err != nil {
		t.Fatal(err)
	}
	if server.params == nil {
		t.Fatalf("initialize params are unset")
	}
	env := server.params.InitializationOptions.(map[string]interface{})["env"].(map[string]interface{})

	// Check for an arbitrary Go variable. It should be set.
	if _, ok := env["GOPRIVATE"]; !ok {
		t.Errorf("Go environment variable GOPRIVATE unset in initialization options")
	}
	// Check that the variable present in our user config was not overwritten.
	if v := env["GONOPROXY"]; v != "example.com" {
		t.Errorf("GONOPROXY environment variable was overwritten")
	}
}

func TestListenParsing(t *testing.T) {
	tests := []struct {
		input, wantNetwork, wantAddr string
	}{
		{"127.0.0.1:0", "tcp", "127.0.0.1:0"},
		{"unix;/tmp/sock", "unix", "/tmp/sock"},
		{"auto", "auto", ""},
		{"auto;foo", "auto", "foo"},
	}

	for _, test := range tests {
		gotNetwork, gotAddr := ParseAddr(test.input)
		if gotNetwork != test.wantNetwork {
			t.Errorf("network = %q, want %q", gotNetwork, test.wantNetwork)
		}
		if gotAddr != test.wantAddr {
			t.Errorf("addr = %q, want %q", gotAddr, test.wantAddr)
		}
	}
}
