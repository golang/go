// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/telemetry/event"
	"golang.org/x/tools/internal/xcontext"
)

const (
	// RequestCancelledError should be used when a request is cancelled early.
	RequestCancelledError = -32800
)

// ClientDispatcher returns a Client that dispatches LSP requests across the
// given jsonrpc2 connection.
func ClientDispatcher(conn *jsonrpc2.Conn) Client {
	conn.OnCancelled(cancelCall)
	return &clientDispatcher{Conn: conn}
}

// ServerDispatcher returns a Server that dispatches LSP requests across the
// given jsonrpc2 connection.
func ServerDispatcher(conn *jsonrpc2.Conn) Server {
	conn.OnCancelled(cancelCall)
	return &serverDispatcher{Conn: conn}
}

func Handlers(handler jsonrpc2.Handler) jsonrpc2.Handler {
	return CancelHandler(
		CancelHandler(
			jsonrpc2.AsyncHandler(
				jsonrpc2.MustReply(handler))))
}

func CancelHandler(handler jsonrpc2.Handler) jsonrpc2.Handler {
	handler, canceller := jsonrpc2.CancelHandler(handler)
	return func(ctx context.Context, req *jsonrpc2.Request) error {
		if req.Method != "$/cancelRequest" {
			return handler(ctx, req)
		}
		var params CancelParams
		if err := json.Unmarshal(*req.Params, &params); err != nil {
			return sendParseError(ctx, req, err)
		}
		v := jsonrpc2.ID{}
		if n, ok := params.ID.(float64); ok {
			v.Number = int64(n)
		} else if s, ok := params.ID.(string); ok {
			v.Name = s
		} else {
			return sendParseError(ctx, req, fmt.Errorf("Request ID %v malformed", params.ID))
		}
		canceller(v)
		return req.Reply(ctx, nil, nil)
	}
}

func cancelCall(ctx context.Context, conn *jsonrpc2.Conn, id jsonrpc2.ID) {
	ctx = xcontext.Detach(ctx)
	ctx, done := event.StartSpan(ctx, "protocol.canceller")
	defer done()
	// Note that only *jsonrpc2.ID implements json.Marshaler.
	conn.Notify(ctx, "$/cancelRequest", &CancelParams{ID: &id})
}

func sendParseError(ctx context.Context, req *jsonrpc2.Request, err error) error {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	return req.Reply(ctx, nil, err)
}
