// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rFindley): move this to lsprpc_test once it no longer shares with
//                 lsprpc_test.go.

package lsprpc

import (
	"context"
	"log"
	"regexp"
	"testing"
	"time"

	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/internal/lsp/protocol"
)

func TestClientLoggingV2(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	listener, err := jsonrpc2_v2.NetPipe(ctx)
	if err != nil {
		t.Fatal(err)
	}
	newServer := func(ctx context.Context, client protocol.ClientCloser) protocol.Server {
		return pingServer{}
	}
	serverBinder := NewServerBinder(newServer)
	server, err := jsonrpc2_v2.Serve(ctx, listener, serverBinder)
	if err != nil {
		t.Fatal(err)
	}
	client := fakeClient{logs: make(chan string, 10)}
	clientBinder := NewClientBinder(func(context.Context, protocol.Server) protocol.Client { return client })
	conn, err := jsonrpc2_v2.Dial(ctx, listener.Dialer(), clientBinder)
	if err != nil {
		t.Fatal(err)
	}
	if err := protocol.ServerDispatcherV2(conn).DidOpen(ctx, &protocol.DidOpenTextDocumentParams{}); err != nil {
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
	if err := listener.Close(); err != nil {
		t.Error(err)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	if err := server.Wait(); err != nil {
		log.Fatal(err)
	}
}
