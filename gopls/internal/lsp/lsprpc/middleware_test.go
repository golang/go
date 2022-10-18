// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	. "golang.org/x/tools/gopls/internal/lsp/lsprpc"
	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
)

var noopBinder = BinderFunc(func(context.Context, *jsonrpc2_v2.Connection) jsonrpc2_v2.ConnectionOptions {
	return jsonrpc2_v2.ConnectionOptions{}
})

func TestHandshakeMiddleware(t *testing.T) {
	sh := &Handshaker{
		Metadata: Metadata{
			"answer": 42,
		},
	}
	ctx := context.Background()
	env := new(TestEnv)
	defer env.Shutdown(t)
	l, _ := env.serve(ctx, t, sh.Middleware(noopBinder))
	conn := env.dial(ctx, t, l.Dialer(), noopBinder, false)
	ch := &Handshaker{
		Metadata: Metadata{
			"question": 6 * 9,
		},
	}

	check := func(connected bool) error {
		clients := sh.Peers()
		servers := ch.Peers()
		want := 0
		if connected {
			want = 1
		}
		if got := len(clients); got != want {
			return fmt.Errorf("got %d clients on the server, want %d", got, want)
		}
		if got := len(servers); got != want {
			return fmt.Errorf("got %d servers on the client, want %d", got, want)
		}
		if !connected {
			return nil
		}
		client := clients[0]
		server := servers[0]
		if _, ok := client.Metadata["question"]; !ok {
			return errors.New("no client metadata")
		}
		if _, ok := server.Metadata["answer"]; !ok {
			return errors.New("no server metadata")
		}
		if client.LocalID != server.RemoteID {
			return fmt.Errorf("client.LocalID == %d, server.PeerID == %d", client.LocalID, server.RemoteID)
		}
		if client.RemoteID != server.LocalID {
			return fmt.Errorf("client.PeerID == %d, server.LocalID == %d", client.RemoteID, server.LocalID)
		}
		return nil
	}

	if err := check(false); err != nil {
		t.Fatalf("before handshake: %v", err)
	}
	ch.ClientHandshake(ctx, conn)
	if err := check(true); err != nil {
		t.Fatalf("after handshake: %v", err)
	}
	conn.Close()
	// Wait for up to ~2s for connections to get cleaned up.
	delay := 25 * time.Millisecond
	for retries := 3; retries >= 0; retries-- {
		time.Sleep(delay)
		err := check(false)
		if err == nil {
			return
		}
		if retries == 0 {
			t.Fatalf("after closing connection: %v", err)
		}
		delay *= 4
	}
}
