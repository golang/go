// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/xcontext"
)

var (
	// RequestCancelledError should be used when a request is cancelled early.
	RequestCancelledError = jsonrpc2.NewError(-32800, "JSON RPC cancelled")
)

// ClientDispatcher returns a Client that dispatches LSP requests across the
// given jsonrpc2 connection.
func ClientDispatcher(conn *jsonrpc2.Conn) Client {
	return &clientDispatcher{Conn: conn}
}

// ServerDispatcher returns a Server that dispatches LSP requests across the
// given jsonrpc2 connection.
func ServerDispatcher(conn *jsonrpc2.Conn) Server {
	return &serverDispatcher{Conn: conn}
}

func Handlers(handler jsonrpc2.Handler) jsonrpc2.Handler {
	return CancelHandler(
		jsonrpc2.AsyncHandler(
			jsonrpc2.MustReplyHandler(handler)))
}

func CancelHandler(handler jsonrpc2.Handler) jsonrpc2.Handler {
	handler, canceller := jsonrpc2.CancelHandler(handler)
	return func(ctx context.Context, reply jsonrpc2.Replier, req jsonrpc2.Request) error {
		if req.Method() != "$/cancelRequest" {
			// TODO(iancottrell): See if we can generate a reply for the request to be cancelled
			// at the point of cancellation rather than waiting for gopls to naturally reply.
			// To do that, we need to keep track of whether a reply has been sent already and
			// be careful about racing between the two paths.
			// TODO(iancottrell): Add a test that watches the stream and verifies the response
			// for the cancelled request flows.
			replyWithDetachedContext := func(ctx context.Context, resp interface{}, err error) error {
				// https://microsoft.github.io/language-server-protocol/specifications/specification-current/#cancelRequest
				if ctx.Err() != nil && err == nil {
					err = RequestCancelledError
				}
				ctx = xcontext.Detach(ctx)
				return reply(ctx, resp, err)
			}
			return handler(ctx, replyWithDetachedContext, req)
		}
		var params CancelParams
		if err := json.Unmarshal(req.Params(), &params); err != nil {
			return sendParseError(ctx, reply, err)
		}
		if n, ok := params.ID.(float64); ok {
			canceller(jsonrpc2.NewIntID(int64(n)))
		} else if s, ok := params.ID.(string); ok {
			canceller(jsonrpc2.NewStringID(s))
		} else {
			return sendParseError(ctx, reply, fmt.Errorf("request ID %v malformed", params.ID))
		}
		return reply(ctx, nil, nil)
	}
}

func Call(ctx context.Context, conn *jsonrpc2.Conn, method string, params interface{}, result interface{}) error {
	id, err := conn.Call(ctx, method, params, result)
	if ctx.Err() != nil {
		cancelCall(ctx, conn, id)
	}
	return err
}

func cancelCall(ctx context.Context, conn *jsonrpc2.Conn, id jsonrpc2.ID) {
	ctx = xcontext.Detach(ctx)
	ctx, done := event.Start(ctx, "protocol.canceller")
	defer done()
	// Note that only *jsonrpc2.ID implements json.Marshaler.
	conn.Notify(ctx, "$/cancelRequest", &CancelParams{ID: &id})
}

func sendParseError(ctx context.Context, reply jsonrpc2.Replier, err error) error {
	return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
}
