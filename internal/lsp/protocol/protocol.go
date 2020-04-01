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
	return &clientDispatcher{Conn: conn}
}

// ServerDispatcher returns a Server that dispatches LSP requests across the
// given jsonrpc2 connection.
func ServerDispatcher(conn *jsonrpc2.Conn) Server {
	return &serverDispatcher{Conn: conn}
}

// Canceller is a jsonrpc2.Handler that handles LSP request cancellation.
type Canceller struct{}

func (Canceller) Request(ctx context.Context, conn *jsonrpc2.Conn, direction jsonrpc2.Direction, r *jsonrpc2.WireRequest) context.Context {
	if direction == jsonrpc2.Receive && r.Method == "$/cancelRequest" {
		var params CancelParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			event.Error(ctx, "", err)
		} else {
			v := jsonrpc2.ID{}
			if n, ok := params.ID.(float64); ok {
				v.Number = int64(n)
			} else if s, ok := params.ID.(string); ok {
				v.Name = s
			} else {
				event.Error(ctx, fmt.Sprintf("Request ID %v malformed", params.ID), nil)
				return ctx
			}
			conn.Cancel(v)
		}
	}
	return ctx
}

func (Canceller) Cancel(ctx context.Context, conn *jsonrpc2.Conn, id jsonrpc2.ID, cancelled bool) bool {
	if cancelled {
		return false
	}
	ctx = xcontext.Detach(ctx)
	ctx, done := event.StartSpan(ctx, "protocol.canceller")
	defer done()
	// Note that only *jsonrpc2.ID implements json.Marshaler.
	conn.Notify(ctx, "$/cancelRequest", &CancelParams{ID: &id})
	return true
}

func (Canceller) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	// Hide cancellations from downstream handlers.
	if r.Method != "$/cancelRequest" {
		return false
	}
	r.Reply(ctx, nil, nil)
	return true
}

func sendParseError(ctx context.Context, req *jsonrpc2.Request, err error) error {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	return req.Reply(ctx, nil, err)
}
