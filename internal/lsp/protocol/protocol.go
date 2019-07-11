// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/xcontext"
)

type DocumentUri = string

type canceller struct{ jsonrpc2.EmptyHandler }

type clientHandler struct {
	canceller
	log    xlog.Logger
	client Client
}

type serverHandler struct {
	canceller
	log    xlog.Logger
	server Server
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

func NewClient(ctx context.Context, stream jsonrpc2.Stream, client Client) (context.Context, *jsonrpc2.Conn, Server) {
	ctx = xlog.With(ctx, NewLogger(client))
	conn := jsonrpc2.NewConn(stream)
	conn.AddHandler(&clientHandler{log: xlog.From(ctx), client: client})
	return ctx, conn, &serverDispatcher{Conn: conn}
}

func NewServer(ctx context.Context, stream jsonrpc2.Stream, server Server) (context.Context, *jsonrpc2.Conn, Client) {
	conn := jsonrpc2.NewConn(stream)
	client := &clientDispatcher{Conn: conn}
	ctx = xlog.With(ctx, NewLogger(client))
	conn.AddHandler(&serverHandler{log: xlog.From(ctx), server: server})
	return ctx, conn, client
}

func sendParseError(ctx context.Context, log xlog.Logger, req *jsonrpc2.Request, err error) {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	if err := req.Reply(ctx, nil, err); err != nil {
		xlog.Errorf(ctx, "%v", err)
	}
}
