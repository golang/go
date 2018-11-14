// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"
	"log"

	"golang.org/x/tools/internal/jsonrpc2"
)

func canceller(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request) {
	conn.Notify(context.Background(), "$/cancelRequest", &CancelParams{ID: *req.ID})
}

func RunClient(ctx context.Context, stream jsonrpc2.Stream, client Client, opts ...interface{}) (*jsonrpc2.Conn, Server) {
	opts = append([]interface{}{clientHandler(client), jsonrpc2.Canceler(canceller)}, opts...)
	conn := jsonrpc2.NewConn(ctx, stream, opts...)
	return conn, &serverDispatcher{Conn: conn}
}

func RunServer(ctx context.Context, stream jsonrpc2.Stream, server Server, opts ...interface{}) (*jsonrpc2.Conn, Client) {
	opts = append([]interface{}{serverHandler(server), jsonrpc2.Canceler(canceller)}, opts...)
	conn := jsonrpc2.NewConn(ctx, stream, opts...)
	return conn, &clientDispatcher{Conn: conn}
}

func sendParseError(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, err error) {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	unhandledError(conn.Reply(ctx, req, nil, err))
}

// unhandledError is used in places where an error may occur that cannot be handled.
// This occurs in things like rpc handlers that are a notify, where we cannot
// reply to the caller, or in a call when we are actually attempting to reply.
// In these cases, there is nothing we can do with the error except log it, so
// we do that in this function, and the presence of this function acts as a
// useful reminder of why we are effectively dropping the error and also a
// good place to hook in when debugging those kinds of errors.
func unhandledError(err error) {
	if err == nil {
		return
	}
	log.Printf("%v", err)
}
