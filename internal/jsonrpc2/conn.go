// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

// Conn is the common interface to jsonrpc clients and servers.
// Conn is bidirectional; it does not have a designated server or client end.
// It manages the jsonrpc2 protocol, connecting responses back to their calls.
type Conn interface {
	// Call invokes the target method and waits for a response.
	// The params will be marshaled to JSON before sending over the wire, and will
	// be handed to the method invoked.
	// The response will be unmarshaled from JSON into the result.
	// The id returned will be unique from this connection, and can be used for
	// logging or tracking.
	Call(ctx context.Context, method string, params, result interface{}) (ID, error)

	// Notify invokes the target method but does not wait for a response.
	// The params will be marshaled to JSON before sending over the wire, and will
	// be handed to the method invoked.
	Notify(ctx context.Context, method string, params interface{}) error

	// Go starts a goroutine to handle the connection.
	// It must be called exactly once for each Conn.
	// It returns immediately.
	// You must block on Done() to wait for the connection to shut down.
	// This is a temporary measure, this should be started automatically in the
	// future.
	Go(ctx context.Context, handler Handler)

	// Close closes the connection and it's underlying stream.
	// It does not wait for the close to complete, use the Done() channel for
	// that.
	Close() error

	// Done returns a channel that will be closed when the processing goroutine
	// has terminated, which will happen if Close() is called or an underlying
	// stream is closed.
	Done() <-chan struct{}

	// Err returns an error if there was one from within the processing goroutine.
	// If err returns non nil, the connection will be already closed or closing.
	Err() error
}

type conn struct {
	seq       int64      // must only be accessed using atomic operations
	writeMu   sync.Mutex // protects writes to the stream
	stream    Stream
	pendingMu sync.Mutex // protects the pending map
	pending   map[ID]chan *Response

	done chan struct{}
	err  atomic.Value
}

// NewConn creates a new connection object around the supplied stream.
func NewConn(s Stream) Conn {
	conn := &conn{
		stream:  s,
		pending: make(map[ID]chan *Response),
		done:    make(chan struct{}),
	}
	return conn
}

func (c *conn) Notify(ctx context.Context, method string, params interface{}) (err error) {
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

func (c *conn) Call(ctx context.Context, method string, params, result interface{}) (_ ID, err error) {
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

func (c *conn) replier(req Request, spanDone func()) Replier {
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
		n, err := c.write(ctx, response)
		event.Metric(ctx, tag.SentBytes.Of(n))
		if err != nil {
			// TODO(iancottrell): if a stream write fails, we really need to shut down
			// the whole stream
			return err
		}
		return nil
	}
}

func (c *conn) write(ctx context.Context, msg Message) (int64, error) {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return c.stream.Write(ctx, msg)
}

func (c *conn) Go(ctx context.Context, handler Handler) {
	go c.run(ctx, handler)
}

func (c *conn) run(ctx context.Context, handler Handler) {
	defer close(c.done)
	for {
		// get the next message
		msg, n, err := c.stream.Read(ctx)
		if err != nil {
			// The stream failed, we cannot continue.
			c.fail(err)
			return
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
			reqCtx, spanDone := event.Start(ctx, msg.Method(), labels...)
			event.Metric(reqCtx,
				tag.Started.Of(1),
				tag.ReceivedBytes.Of(n))
			if err := handler(reqCtx, c.replier(msg, spanDone), msg); err != nil {
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

func (c *conn) Close() error {
	return c.stream.Close()
}

func (c *conn) Done() <-chan struct{} {
	return c.done
}

func (c *conn) Err() error {
	if err := c.err.Load(); err != nil {
		return err.(error)
	}
	return nil
}

// fail sets a failure condition on the stream and closes it.
func (c *conn) fail(err error) {
	c.err.Store(err)
	c.stream.Close()
}

func recordStatus(ctx context.Context, err error) {
	if err != nil {
		event.Label(ctx, tag.StatusCode.Of("ERROR"))
	} else {
		event.Label(ctx, tag.StatusCode.Of("OK"))
	}
}
