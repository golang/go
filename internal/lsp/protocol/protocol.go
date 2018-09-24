// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"

	"golang.org/x/tools/internal/jsonrpc2"
)

func canceller(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request) {
	conn.Notify(context.Background(), "$/cancelRequest", &CancelParams{ID: *req.ID})
}

func RunClient(ctx context.Context, stream jsonrpc2.Stream, client Client, opts ...interface{}) (*jsonrpc2.Conn, Server) {
	opts = append([]interface{}{clientHandler(client), canceller}, opts...)
	conn := jsonrpc2.NewConn(ctx, stream, opts...)
	return conn, &serverDispatcher{Conn: conn}
}

func RunServer(ctx context.Context, stream jsonrpc2.Stream, server Server, opts ...interface{}) (*jsonrpc2.Conn, Client) {
	opts = append([]interface{}{serverHandler(server), canceller}, opts...)
	conn := jsonrpc2.NewConn(ctx, stream, opts...)
	return conn, &clientDispatcher{Conn: conn}
}

func toJSONError(err error) *jsonrpc2.Error {
	if jsonError, ok := err.(*jsonrpc2.Error); ok {
		return jsonError
	}
	return jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
}
