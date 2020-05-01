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
	"fmt"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/lsp/debug/tag"
)

const (
	// ErrIdleTimeout is returned when serving timed out waiting for new connections.
	ErrIdleTimeout = constError("timed out waiting for new connections")
)

// Conn is a JSON RPC 2 client server connection.
// Conn is bidirectional; it does not have a designated server or client end.
type Conn struct {
	seq       int64      // must only be accessed using atomic operations
	writeMu   sync.Mutex // protects writes to the stream
	stream    Stream
	pendingMu sync.Mutex // protects the pending map
	pending   map[ID]chan *Response
}

type constError string

func (e constError) Error() string { return string(e) }

// NewConn creates a new connection object around the supplied stream.
// You must call Run for the connection to be active.
func NewConn(s Stream) *Conn {
	conn := &Conn{
		stream:  s,
		pending: make(map[ID]chan *Response),
	}
	return conn
}

// Notify is called to send a notification request over the connection.
// It will return as soon as the notification has been sent, as no response is
// possible.
func (c *Conn) Notify(ctx context.Context, method string, params interface{}) (err error) {
	notify, err := NewNotification(method, params)
	if err != nil {
		return fmt.Errorf("marshaling notify parameters: %v", err)
	}
	ctx, done := event.Start(ctx, method,
		tag.Method.Of(method),
		tag.RPCDirection.Of(tag.Outbound),
	)
	defer func() {
		recordStatus(ctx, err)
		done()
	}()

	event.Metric(ctx, tag.Started.Of(1))
	n, err := c.write(ctx, notify)
	event.Metric(ctx, tag.SentBytes.Of(n))
	return err
}

// Call sends a request over the connection and then waits for a response.
// If the response is not an error, it will be decoded into result.
// result must be of a type you an pass to json.Unmarshal.
func (c *Conn) Call(ctx context.Context, method string, params, result interface{}) (_ ID, err error) {
	// generate a new request identifier
	id := ID{number: atomic.AddInt64(&c.seq, 1)}
	call, err := NewCall(id, method, params)
	if err != nil {
		return id, fmt.Errorf("marshaling call parameters: %v", err)
	}
	ctx, done := event.Start(ctx, method,
		tag.Method.Of(method),
		tag.RPCDirection.Of(tag.Outbound),
		tag.RPCID.Of(fmt.Sprintf("%q", id)),
	)
	defer func() {
		recordStatus(ctx, err)
		done()
	}()
	event.Metric(ctx, tag.Started.Of(1))
	// We have to add ourselves to the pending map before we send, otherwise we
	// are racing the response. Also add a buffer to rchan, so that if we get a
	// wire response between the time this call is cancelled and id is deleted
	// from c.pending, the send to rchan will not block.
	rchan := make(chan *Response, 1)
	c.pendingMu.Lock()
	c.pending[id] = rchan
	c.pendingMu.Unlock()
	defer func() {
		c.pendingMu.Lock()
		delete(c.pending, id)
		c.pendingMu.Unlock()
	}()
	// now we are ready to send
	n, err := c.write(ctx, call)
	event.Metric(ctx, tag.SentBytes.Of(n))
	if err != nil {
		// sending failed, we will never get a response, so don't leave it pending
		return id, err
	}
	// now wait for the response
	select {
	case response := <-rchan:
		// is it an error response?
		if response.err != nil {
			return id, response.err
		}
		if result == nil || len(response.result) == 0 {
			return id, nil
		}
		if err := json.Unmarshal(response.result, result); err != nil {
			return id, fmt.Errorf("unmarshaling result: %v", err)
		}
		return id, nil
	case <-ctx.Done():
		return id, ctx.Err()
	}
}

func replier(conn *Conn, req Request, spanDone func()) Replier {
	return func(ctx context.Context, result interface{}, err error) error {
		defer func() {
			recordStatus(ctx, err)
			spanDone()
		}()
		call, ok := req.(*Call)
		if !ok {
			// request was a notify, no need to respond
			return nil
		}
		response, err := NewResponse(call.id, result, err)
		if err != nil {
			return err
		}
		n, err := conn.write(ctx, response)
		event.Metric(ctx, tag.SentBytes.Of(n))
		if err != nil {
			// TODO(iancottrell): if a stream write fails, we really need to shut down
			// the whole stream
			return err
		}
		return nil
	}
}

func (c *Conn) write(ctx context.Context, msg Message) (int64, error) {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return c.stream.Write(ctx, msg)
}

// Run blocks until the connection is terminated, and returns any error that
// caused the termination.
// It must be called exactly once for each Conn.
// It returns only when the reader is closed or there is an error in the stream.
func (c *Conn) Run(runCtx context.Context, handler Handler) error {
	for {
		// get the next message
		msg, n, err := c.stream.Read(runCtx)
		if err != nil {
			// The stream failed, we cannot continue. If the client disconnected
			// normally, we should get ErrDisconnected here.
			return err
		}
		switch msg := msg.(type) {
		case Request:
			labels := []label.Label{
				tag.Method.Of(msg.Method()),
				tag.RPCDirection.Of(tag.Inbound),
				{}, // reserved for ID if present
			}
			if call, ok := msg.(*Call); ok {
				labels[len(labels)-1] = tag.RPCID.Of(fmt.Sprintf("%q", call.ID()))
			} else {
				labels = labels[:len(labels)-1]
			}
			reqCtx, spanDone := event.Start(runCtx, msg.Method(), labels...)
			event.Metric(reqCtx,
				tag.Started.Of(1),
				tag.ReceivedBytes.Of(n))
			if err := handler(reqCtx, replier(c, msg, spanDone), msg); err != nil {
				// delivery failed, not much we can do
				event.Error(reqCtx, "jsonrpc2 message delivery failed", err)
			}
		case *Response:
			// If method is not set, this should be a response, in which case we must
			// have an id to send the response back to the caller.
			c.pendingMu.Lock()
			rchan, ok := c.pending[msg.id]
			c.pendingMu.Unlock()
			if ok {
				rchan <- msg
			}
		}
	}
}

func marshalToRaw(obj interface{}) (json.RawMessage, error) {
	data, err := json.Marshal(obj)
	if err != nil {
		return json.RawMessage{}, err
	}
	return json.RawMessage(data), nil
}

func recordStatus(ctx context.Context, err error) {
	if err != nil {
		event.Label(ctx, tag.StatusCode.Of("ERROR"))
	} else {
		event.Label(ctx, tag.StatusCode.Of("OK"))
	}
}
