// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/xlog"
)

const defaultMessageBufferSize = 20

func canceller(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request) {
	conn.Notify(context.Background(), "$/cancelRequest", &CancelParams{ID: *req.ID})
}

func NewClient(stream jsonrpc2.Stream, client Client) (*jsonrpc2.Conn, Server, xlog.Logger) {
	log := xlog.New(NewLogger(client))
	conn := jsonrpc2.NewConn(stream)
	conn.Capacity = defaultMessageBufferSize
	conn.RejectIfOverloaded = true
	conn.Handler = clientHandler(log, client)
	conn.Canceler = jsonrpc2.Canceler(canceller)
	return conn, &serverDispatcher{Conn: conn}, log
}

func NewServer(stream jsonrpc2.Stream, server Server) (*jsonrpc2.Conn, Client, xlog.Logger) {
	conn := jsonrpc2.NewConn(stream)
	client := &clientDispatcher{Conn: conn}
	log := xlog.New(NewLogger(client))
	conn.Capacity = defaultMessageBufferSize
	conn.RejectIfOverloaded = true
	conn.Handler = serverHandler(log, server)
	conn.Canceler = jsonrpc2.Canceler(canceller)
	return conn, client, log
}

func sendParseError(ctx context.Context, log xlog.Logger, conn *jsonrpc2.Conn, req *jsonrpc2.Request, err error) {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	if err := conn.Reply(ctx, req, nil, err); err != nil {
		log.Errorf(ctx, "%v", err)
	}
}
