// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"regexp"
	"testing"
	"time"

	"golang.org/x/tools/internal/jsonrpc2/servertest"
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

func TestClientLogging(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := pingServer{}
	client := fakeClient{logs: make(chan string, 10)}

	ss := &StreamServer{
		accept: func(c protocol.Client) protocol.Server {
			return server
		},
	}
	ts := servertest.NewServer(ctx, ss)
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
	case <-time.After(1000 * time.Second):
		t.Error("timeout waiting for client log")
	}
}

type waitableServer struct {
	protocol.Server

	started chan struct{}
	// finished records whether the request ended with a cancellation or not
	// (true means the request was cancelled).
	finished chan bool
}

func (s waitableServer) CodeLens(ctx context.Context, params *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	s.started <- struct{}{}
	cancelled := false
	defer func() {
		s.finished <- cancelled
	}()
	select {
	case <-ctx.Done():
		cancelled = true
		return nil, ctx.Err()
	case <-time.After(1 * time.Second):
		cancelled = false
	}
	return []protocol.CodeLens{}, nil
}

func TestRequestCancellation(t *testing.T) {
	server := waitableServer{
		started:  make(chan struct{}),
		finished: make(chan bool),
	}
	ss := &StreamServer{
		accept: func(c protocol.Client) protocol.Server {
			return server
		},
	}
	ctx := context.Background()
	ts := servertest.NewServer(ctx, ss)
	cc := ts.Connect(ctx)
	cc.AddHandler(protocol.Canceller{})
	lensCtx, cancelLens := context.WithCancel(context.Background())
	go func() {
		protocol.ServerDispatcher(cc).CodeLens(lensCtx, &protocol.CodeLensParams{})
	}()
	<-server.started
	cancelLens()
	if got, want := <-server.finished, true; got != want {
		t.Errorf("CodeLens was cancelled: %t, want %t", got, want)
	}
}

// TODO: add a test for telemetry.
