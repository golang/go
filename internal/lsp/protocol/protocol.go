// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	"golang.org/x/tools/internal/xcontext"
)

const (
	// RequestCancelledError should be used when a request is cancelled early.
	RequestCancelledError = -32800
)

type canceller struct{ jsonrpc2.EmptyHandler }

type clientHandler struct {
	jsonrpc2.EmptyHandler
	client Client
}

type serverHandler struct {
	jsonrpc2.EmptyHandler
	server Server
}

func (canceller) Request(ctx context.Context, conn *jsonrpc2.Conn, direction jsonrpc2.Direction, r *jsonrpc2.WireRequest) context.Context {
	if direction == jsonrpc2.Receive && r.Method == "$/cancelRequest" {
		var params CancelParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			log.Error(ctx, "", err)
		} else {
			v := jsonrpc2.ID{}
			if n, ok := params.ID.(float64); ok {
				v.Number = int64(n)
			} else if s, ok := params.ID.(string); ok {
				v.Name = s
			} else {
				log.Error(ctx, fmt.Sprintf("Request ID %v malformed", params.ID), nil)
				return ctx
			}
			conn.Cancel(v)
		}
	}
	return ctx
}

func (canceller) Cancel(ctx context.Context, conn *jsonrpc2.Conn, id jsonrpc2.ID, cancelled bool) bool {
	if cancelled {
		return false
	}
	ctx = xcontext.Detach(ctx)
	ctx, done := trace.StartSpan(ctx, "protocol.canceller")
	defer done()
	conn.Notify(ctx, "$/cancelRequest", &CancelParams{ID: id})
	return true
}

func (canceller) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	// Hide cancellations from downstream handlers.
	return r.Method == "$/cancelRequest"
}

func NewClient(ctx context.Context, stream jsonrpc2.Stream, client Client) (context.Context, *jsonrpc2.Conn, Server) {
	ctx = WithClient(ctx, client)
	conn := jsonrpc2.NewConn(stream)
	conn.AddHandler(&clientHandler{client: client})
	conn.AddHandler(&canceller{})
	return ctx, conn, &serverDispatcher{Conn: conn}
}

func NewServer(ctx context.Context, stream jsonrpc2.Stream, server Server) (context.Context, *jsonrpc2.Conn, Client) {
	conn := jsonrpc2.NewConn(stream)
	client := &clientDispatcher{Conn: conn}
	ctx = WithClient(ctx, client)
	conn.AddHandler(&serverHandler{server: server})
	conn.AddHandler(&canceller{})
	return ctx, conn, client
}

func sendParseError(ctx context.Context, req *jsonrpc2.Request, err error) {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	if err := req.Reply(ctx, nil, err); err != nil {
		log.Error(ctx, "", err)
	}
}
