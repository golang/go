// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
)

// Handler is the interface used to hook into the message handling of an rpc
// connection.
type Handler interface {
	// Deliver is invoked to handle incoming requests.
	// If the request returns false from IsNotify then the Handler must eventually
	// call Reply on the Conn with the supplied request.
	// Handlers are called synchronously, they should pass the work off to a go
	// routine if they are going to take a long time.
	// If Deliver returns true all subsequent handlers will be invoked with
	// delivered set to true, and should not attempt to deliver the message.
	Deliver(ctx context.Context, r *Request, delivered bool) bool

	// Cancel is invoked for cancelled outgoing requests.
	// It is okay to use the connection to send notifications, but the context will
	// be in the cancelled state, so you must do it with the background context
	// instead.
	// If Cancel returns true all subsequent handlers will be invoked with
	// cancelled set to true, and should not attempt to cancel the message.
	Cancel(ctx context.Context, conn *Conn, id ID, cancelled bool) bool

	// Request is called near the start of processing any request.
	Request(ctx context.Context, conn *Conn, direction Direction, r *WireRequest) context.Context
}

// Direction is used to indicate to a logger whether the logged message was being
// sent or received.
type Direction bool

const (
	// Send indicates the message is outgoing.
	Send = Direction(true)
	// Receive indicates the message is incoming.
	Receive = Direction(false)
)

func (d Direction) String() string {
	switch d {
	case Send:
		return "send"
	case Receive:
		return "receive"
	default:
		panic("unreachable")
	}
}

type EmptyHandler struct{}

func (EmptyHandler) Deliver(ctx context.Context, r *Request, delivered bool) bool {
	return false
}

func (EmptyHandler) Cancel(ctx context.Context, conn *Conn, id ID, cancelled bool) bool {
	return false
}

func (EmptyHandler) Request(ctx context.Context, conn *Conn, direction Direction, r *WireRequest) context.Context {
	return ctx
}

func (EmptyHandler) Response(ctx context.Context, conn *Conn, direction Direction, r *WireResponse) context.Context {
	return ctx
}

type defaultHandler struct{ EmptyHandler }

func (defaultHandler) Deliver(ctx context.Context, r *Request, delivered bool) bool {
	if delivered {
		return false
	}
	if !r.IsNotify() {
		r.Reply(ctx, nil, NewErrorf(CodeMethodNotFound, "method %q not found", r.Method))
	}
	return true
}
