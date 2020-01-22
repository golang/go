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

type fakeHandler struct {
	jsonrpc2.EmptyHandler
}

type msg struct {
	Msg string
}

func (fakeHandler) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	if err := r.Reply(ctx, &msg{"pong"}, nil); err != nil {
		panic(err)
	}
	return true
}

func TestTestServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ts := NewServer(ctx, jsonrpc2.HandlerServer(fakeHandler{}))
	defer ts.Close()
	conn := ts.Connect(ctx)
	var got msg
	if err := conn.Call(ctx, "ping", &msg{"ping"}, &got); err != nil {
		t.Fatal(err)
	}
	if want := "pong"; got.Msg != want {
		t.Errorf("conn.Call(...): returned %q, want %q", got, want)
	}
}
