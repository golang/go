// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jsonrpc2 is a minimal implementation of the JSON RPC 2 spec.
// https://www.jsonrpc.org/specification
// It is intended to be compatible with other implementations at the wire level.
package jsonrpc2

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/telemetry/event"
)

const (
	// ErrIdleTimeout is returned when serving timed out waiting for new connections.
	ErrIdleTimeout = constError("timed out waiting for new connections")

	// ErrDisconnected signals that the stream or connection exited normally.
	ErrDisconnected = constError("disconnected")
)

// Conn is a JSON RPC 2 client server connection.
// Conn is bidirectional; it does not have a designated server or client end.
type Conn struct {
	seq       int64 // must only be accessed using atomic operations
	stream    Stream
	pendingMu sync.Mutex // protects the pending map
	pending   map[ID]chan *wireResponse
}

// Request is sent to a server to represent a Call or Notify operaton.
type Request struct {
	// The Wire values of the request.
	// Method is a string containing the method name to invoke.
	Method string
	// Params is either a struct or an array with the parameters of the method.
	Params *json.RawMessage
	// The id of this request, used to tie the Response back to the request.
	// Will be either a string or a number. If not set, the Request is a notify,
	// and no response is possible.
	ID *ID
}

type constError string

func (e constError) Error() string { return string(e) }

// NewConn creates a new connection object around the supplied stream.
// You must call Run for the connection to be active.
func NewConn(s Stream) *Conn {
	conn := &Conn{
		stream:  s,
		pending: make(map[ID]chan *wireResponse),
	}
	return conn
}

// Notify is called to send a notification request over the connection.
// It will return as soon as the notification has been sent, as no response is
// possible.
func (c *Conn) Notify(ctx context.Context, method string, params interface{}) (err error) {
	jsonParams, err := marshalToRaw(params)
	if err != nil {
		return fmt.Errorf("marshaling notify parameters: %v", err)
	}
	request := &wireRequest{
		Method: method,
		Params: jsonParams,
	}
	data, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("marshaling notify request: %v", err)
	}
	ctx, done := event.StartSpan(ctx, request.Method,
		tag.Method.Of(request.Method),
		tag.RPCDirection.Of(tag.Outbound),
		tag.RPCID.Of(fmt.Sprintf("%q", request.ID)),
	)
	defer func() {
		recordStatus(ctx, err)
		done()
	}()

	event.Record(ctx, tag.Started.Of(1))
	n, err := c.stream.Write(ctx, data)
	event.Record(ctx, tag.SentBytes.Of(n))
	return err
}

// Call sends a request over the connection and then waits for a response.
// If the response is not an error, it will be decoded into result.
// result must be of a type you an pass to json.Unmarshal.
func (c *Conn) Call(ctx context.Context, method string, params, result interface{}) (_ ID, err error) {
	// generate a new request identifier
	id := ID{number: atomic.AddInt64(&c.seq, 1)}
	jsonParams, err := marshalToRaw(params)
	if err != nil {
		return id, fmt.Errorf("marshaling call parameters: %v", err)
	}
	request := &wireRequest{
		ID:     &id,
		Method: method,
		Params: jsonParams,
	}
	// marshal the request now it is complete
	data, err := json.Marshal(request)
	if err != nil {
		return id, fmt.Errorf("marshaling call request: %v", err)
	}
	ctx, done := event.StartSpan(ctx, request.Method,
		tag.Method.Of(request.Method),
		tag.RPCDirection.Of(tag.Outbound),
		tag.RPCID.Of(fmt.Sprintf("%q", request.ID)),
	)
	defer func() {
		recordStatus(ctx, err)
		done()
	}()
	event.Record(ctx, tag.Started.Of(1))
	// We have to add ourselves to the pending map before we send, otherwise we
	// are racing the response. Also add a buffer to rchan, so that if we get a
	// wire response between the time this call is cancelled and id is deleted
	// from c.pending, the send to rchan will not block.
	rchan := make(chan *wireResponse, 1)
	c.pendingMu.Lock()
	c.pending[id] = rchan
	c.pendingMu.Unlock()
	defer func() {
		c.pendingMu.Lock()
		delete(c.pending, id)
		c.pendingMu.Unlock()
	}()
	// now we are ready to send
	n, err := c.stream.Write(ctx, data)
	event.Record(ctx, tag.SentBytes.Of(n))
	if err != nil {
		// sending failed, we will never get a response, so don't leave it pending
		return id, err
	}
	// now wait for the response
	select {
	case response := <-rchan:
		// is it an error response?
		if response.Error != nil {
			return id, response.Error
		}
		if result == nil || response.Result == nil {
			return id, nil
		}
		if err := json.Unmarshal(*response.Result, result); err != nil {
			return id, fmt.Errorf("unmarshaling result: %v", err)
		}
		return id, nil
	case <-ctx.Done():
		return id, ctx.Err()
	}
}

func replier(conn *Conn, r *Request, spanDone func()) Replier {
	return func(ctx context.Context, result interface{}, err error) error {
		defer func() {
			recordStatus(ctx, err)
			spanDone()
		}()

		if r.ID == nil {
			// request was a notify, no need to respond
			return nil
		}

		var raw *json.RawMessage
		if err == nil {
			raw, err = marshalToRaw(result)
		}
		response := &wireResponse{
			Result: raw,
			ID:     r.ID,
		}
		if err != nil {
			if callErr, ok := err.(*wireError); ok {
				response.Error = callErr
			} else {
				response.Error = &wireError{Message: err.Error()}
				var wrapped *wireError
				if errors.As(err, &wrapped) {
					// if we wrapped a wire error, keep the code from the wrapped error
					// but the message from the outer error
					response.Error.Code = wrapped.Code
				}
			}
		}
		data, err := json.Marshal(response)
		if err != nil {
			return err
		}
		n, err := conn.stream.Write(ctx, data)
		event.Record(ctx, tag.SentBytes.Of(n))

		if err != nil {
			// TODO(iancottrell): if a stream write fails, we really need to shut down
			// the whole stream
			return err
		}
		return nil
	}
}

// Run blocks until the connection is terminated, and returns any error that
// caused the termination.
// It must be called exactly once for each Conn.
// It returns only when the reader is closed or there is an error in the stream.
func (c *Conn) Run(runCtx context.Context, handler Handler) error {
	for {
		// get the data for a message
		data, n, err := c.stream.Read(runCtx)
		if err != nil {
			// The stream failed, we cannot continue. If the client disconnected
			// normally, we should get ErrDisconnected here.
			return err
		}
		// read a combined message
		msg := &wireCombined{}
		if err := json.Unmarshal(data, msg); err != nil {
			// a badly formed message arrived, log it and continue
			// we trust the stream to have isolated the error to just this message
			continue
		}
		// Work out whether this is a request or response.
		switch {
		case msg.Method != "":
			// If method is set it must be a request.
			reqCtx, spanDone := event.StartSpan(runCtx, msg.Method,
				tag.Method.Of(msg.Method),
				tag.RPCDirection.Of(tag.Inbound),
				tag.RPCID.Of(fmt.Sprintf("%q", msg.ID)),
			)
			event.Record(reqCtx,
				tag.Started.Of(1),
				tag.ReceivedBytes.Of(n))

			req := &Request{
				Method: msg.Method,
				Params: msg.Params,
				ID:     msg.ID,
			}

			if err := handler(reqCtx, replier(c, req, spanDone), req); err != nil {
				// delivery failed, not much we can do
				event.Error(reqCtx, "jsonrpc2 message delivery failed", err)
			}
		case msg.ID != nil:
			// If method is not set, this should be a response, in which case we must
			// have an id to send the response back to the caller.
			c.pendingMu.Lock()
			rchan, ok := c.pending[*msg.ID]
			c.pendingMu.Unlock()
			if ok {
				response := &wireResponse{
					Result: msg.Result,
					Error:  msg.Error,
					ID:     msg.ID,
				}
				rchan <- response
			}
		default:
		}
	}
}

func marshalToRaw(obj interface{}) (*json.RawMessage, error) {
	data, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	raw := json.RawMessage(data)
	return &raw, nil
}

func recordStatus(ctx context.Context, err error) {
	if err != nil {
		event.Label(ctx, tag.StatusCode.Of("ERROR"))
	} else {
		event.Label(ctx, tag.StatusCode.Of("OK"))
	}
}
