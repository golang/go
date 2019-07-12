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

func NewClient(stream jsonrpc2.Stream, client Client) (*jsonrpc2.Conn, Server, xlog.Logger) {
	log := xlog.New(NewLogger(client))
	conn := jsonrpc2.NewConn(stream)
	conn.AddHandler(&clientHandler{log: log, client: client})
	return conn, &serverDispatcher{Conn: conn}, log
}

func NewServer(stream jsonrpc2.Stream, server Server) (*jsonrpc2.Conn, Client, xlog.Logger) {
	conn := jsonrpc2.NewConn(stream)
	client := &clientDispatcher{Conn: conn}
	log := xlog.New(NewLogger(client))
	conn.AddHandler(&serverHandler{log: log, server: server})
	return conn, client, log
}

func sendParseError(ctx context.Context, log xlog.Logger, req *jsonrpc2.Request, err error) {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	if err := req.Reply(ctx, nil, err); err != nil {
		log.Errorf(ctx, "%v", err)
	}
}
