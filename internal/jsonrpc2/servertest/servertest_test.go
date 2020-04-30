// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package servertest

import (
	"context"
	"testing"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
)

type msg struct {
	Msg string
}

func fakeHandler(ctx context.Context, reply jsonrpc2.Replier, req jsonrpc2.Request) error {
	return reply(ctx, &msg{"pong"}, nil)
}

func TestTestServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	server := jsonrpc2.HandlerServer(fakeHandler)
	tcpTS := NewTCPServer(ctx, server, nil)
	defer tcpTS.Close()
	pipeTS := NewPipeServer(ctx, server, nil)
	defer pipeTS.Close()

	tests := []struct {
		name      string
		connector Connector
	}{
		{"tcp", tcpTS},
		{"pipe", pipeTS},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			conn := test.connector.Connect(ctx)
			conn.Go(ctx, jsonrpc2.MethodNotFound)
			var got msg
			if _, err := conn.Call(ctx, "ping", &msg{"ping"}, &got); err != nil {
				t.Fatal(err)
			}
			if want := "pong"; got.Msg != want {
				t.Errorf("conn.Call(...): returned %q, want %q", got, want)
			}
		})
	}
}
