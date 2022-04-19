// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/internal/xcontext"
)

var (
	// RequestCancelledError should be used when a request is cancelled early.
	RequestCancelledError   = jsonrpc2.NewError(-32800, "JSON RPC cancelled")
	RequestCancelledErrorV2 = jsonrpc2_v2.NewError(-32800, "JSON RPC cancelled")
)

type ClientCloser interface {
	Client
	io.Closer
}

type connSender interface {
	io.Closer

	Notify(ctx context.Context, method string, params interface{}) error
	Call(ctx context.Context, method string, params, result interface{}) error
}

type clientDispatcher struct {
	sender connSender
}

func (c *clientDispatcher) Close() error {
	return c.sender.Close()
}

// ClientDispatcher returns a Client that dispatches LSP requests across the
// given jsonrpc2 connection.
func ClientDispatcher(conn jsonrpc2.Conn) ClientCloser {
	return &clientDispatcher{sender: clientConn{conn}}
}

type clientConn struct {
	conn jsonrpc2.Conn
}

func (c clientConn) Close() error {
	return c.conn.Close()
}

func (c clientConn) Notify(ctx context.Context, method string, params interface{}) error {
	return c.conn.Notify(ctx, method, params)
}

func (c clientConn) Call(ctx context.Context, method string, params interface{}, result interface{}) error {
	id, err := c.conn.Call(ctx, method, params, result)
	if ctx.Err() != nil {
		cancelCall(ctx, c, id)
	}
	return err
}

func ClientDispatcherV2(conn *jsonrpc2_v2.Connection) ClientCloser {
	return &clientDispatcher{clientConnV2{conn}}
}

type clientConnV2 struct {
	conn *jsonrpc2_v2.Connection
}

func (c clientConnV2) Close() error {
	return c.conn.Close()
}

func (c clientConnV2) Notify(ctx context.Context, method string, params interface{}) error {
	return c.conn.Notify(ctx, method, params)
}

func (c clientConnV2) Call(ctx context.Context, method string, params interface{}, result interface{}) error {
	call := c.conn.Call(ctx, method, params)
	err := call.Await(ctx, result)
	if ctx.Err() != nil {
		detached := xcontext.Detach(ctx)
		c.conn.Notify(detached, "$/cancelRequest", &CancelParams{ID: call.ID().Raw()})
	}
	return err
}

// ServerDispatcher returns a Server that dispatches LSP requests across the
// given jsonrpc2 connection.
func ServerDispatcher(conn jsonrpc2.Conn) Server {
	return &serverDispatcher{sender: clientConn{conn}}
}

func ServerDispatcherV2(conn *jsonrpc2_v2.Connection) Server {
	return &serverDispatcher{sender: clientConnV2{conn}}
}

type serverDispatcher struct {
	sender connSender
}

func ClientHandler(client Client, handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, req jsonrpc2.Request) error {
		if ctx.Err() != nil {
			ctx := xcontext.Detach(ctx)
			return reply(ctx, nil, RequestCancelledError)
		}
		handled, err := clientDispatch(ctx, client, reply, req)
		if handled || err != nil {
			return err
		}
		return handler(ctx, reply, req)
	}
}

func ClientHandlerV2(client Client) jsonrpc2_v2.Handler {
	return jsonrpc2_v2.HandlerFunc(func(ctx context.Context, req *jsonrpc2_v2.Request) (interface{}, error) {
		if ctx.Err() != nil {
			return nil, RequestCancelledErrorV2
		}
		req1 := req2to1(req)
		var (
			result interface{}
			resErr error
		)
		replier := func(_ context.Context, res interface{}, err error) error {
			if err != nil {
				resErr = err
				return nil
			}
			result = res
			return nil
		}
		_, err := clientDispatch(ctx, client, replier, req1)
		if err != nil {
			return nil, err
		}
		return result, resErr
	})
}

func ServerHandler(server Server, handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, req jsonrpc2.Request) error {
		if ctx.Err() != nil {
			ctx := xcontext.Detach(ctx)
			return reply(ctx, nil, RequestCancelledError)
		}
		handled, err := serverDispatch(ctx, server, reply, req)
		if handled || err != nil {
			return err
		}
		//TODO: This code is wrong, it ignores handler and assumes non standard
		// request handles everything
		// non standard request should just be a layered handler.
		var params interface{}
		if err := json.Unmarshal(req.Params(), &params); err != nil {
			return sendParseError(ctx, reply, err)
		}
		resp, err := server.NonstandardRequest(ctx, req.Method(), params)
		return reply(ctx, resp, err)

	}
}

func ServerHandlerV2(server Server) jsonrpc2_v2.Handler {
	return jsonrpc2_v2.HandlerFunc(func(ctx context.Context, req *jsonrpc2_v2.Request) (interface{}, error) {
		if ctx.Err() != nil {
			return nil, RequestCancelledErrorV2
		}
		req1 := req2to1(req)
		var (
			result interface{}
			resErr error
		)
		replier := func(_ context.Context, res interface{}, err error) error {
			if err != nil {
				resErr = err
				return nil
			}
			result = res
			return nil
		}
		_, err := serverDispatch(ctx, server, replier, req1)
		if err != nil {
			return nil, err
		}
		return result, resErr
	})
}

func req2to1(req2 *jsonrpc2_v2.Request) jsonrpc2.Request {
	if req2.ID.IsValid() {
		raw := req2.ID.Raw()
		var idv1 jsonrpc2.ID
		switch v := raw.(type) {
		case int64:
			idv1 = jsonrpc2.NewIntID(v)
		case string:
			idv1 = jsonrpc2.NewStringID(v)
		default:
			panic(fmt.Sprintf("unsupported ID type %T", raw))
		}
		req1, err := jsonrpc2.NewCall(idv1, req2.Method, req2.Params)
		if err != nil {
			panic(err)
		}
		return req1
	}
	req1, err := jsonrpc2.NewNotification(req2.Method, req2.Params)
	if err != nil {
		panic(err)
	}
	return req1
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

func Call(ctx context.Context, conn jsonrpc2.Conn, method string, params interface{}, result interface{}) error {
	id, err := conn.Call(ctx, method, params, result)
	if ctx.Err() != nil {
		cancelCall(ctx, clientConn{conn}, id)
	}
	return err
}

func cancelCall(ctx context.Context, sender connSender, id jsonrpc2.ID) {
	ctx = xcontext.Detach(ctx)
	ctx, done := event.Start(ctx, "protocol.canceller")
	defer done()
	// Note that only *jsonrpc2.ID implements json.Marshaler.
	sender.Notify(ctx, "$/cancelRequest", &CancelParams{ID: &id})
}

func sendParseError(ctx context.Context, reply jsonrpc2.Replier, err error) error {
	return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
}
