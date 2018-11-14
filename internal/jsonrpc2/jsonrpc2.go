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
	"time"
)

// Conn is a JSON RPC 2 client server connection.
// Conn is bidirectional; it does not have a designated server or client end.
type Conn struct {
	handle     Handler
	cancel     Canceler
	log        Logger
	stream     Stream
	done       chan struct{}
	err        error
	seq        int64      // must only be accessed using atomic operations
	pendingMu  sync.Mutex // protects the pending map
	pending    map[ID]chan *Response
	handlingMu sync.Mutex // protects the handling map
	handling   map[ID]handling
}

// Handler is an option you can pass to NewConn to handle incoming requests.
// If the request returns false from IsNotify then the Handler must eventually
// call Reply on the Conn with the supplied request.
// Handlers are called synchronously, they should pass the work off to a go
// routine if they are going to take a long time.
type Handler func(context.Context, *Conn, *Request)

// Canceler is an option you can pass to NewConn which is invoked for
// cancelled outgoing requests.
// The request will have the ID filled in, which can be used to propagate the
// cancel to the other process if needed.
// It is okay to use the connection to send notifications, but the context will
// be in the cancelled state, so you must do it with the background context
// instead.
type Canceler func(context.Context, *Conn, *Request)

// NewErrorf builds a Error struct for the suppied message and code.
// If args is not empty, message and args will be passed to Sprintf.
func NewErrorf(code int64, format string, args ...interface{}) *Error {
	return &Error{
		Code:    code,
		Message: fmt.Sprintf(format, args...),
	}
}

// NewConn creates a new connection object that reads and writes messages from
// the supplied stream and dispatches incoming messages to the supplied handler.
func NewConn(ctx context.Context, s Stream, options ...interface{}) *Conn {
	conn := &Conn{
		stream:   s,
		done:     make(chan struct{}),
		pending:  make(map[ID]chan *Response),
		handling: make(map[ID]handling),
	}
	for _, opt := range options {
		switch opt := opt.(type) {
		case Handler:
			if conn.handle != nil {
				panic("Duplicate Handler function in options list")
			}
			conn.handle = opt
		case Canceler:
			if conn.cancel != nil {
				panic("Duplicate Canceler function in options list")
			}
			conn.cancel = opt
		case Logger:
			if conn.log != nil {
				panic("Duplicate Logger function in options list")
			}
			conn.log = opt
		default:
			panic(fmt.Errorf("Unknown option type %T in options list", opt))
		}
	}
	if conn.handle == nil {
		// the default handler reports a method error
		conn.handle = func(ctx context.Context, c *Conn, r *Request) {
			if r.IsNotify() {
				c.Reply(ctx, r, nil, NewErrorf(CodeMethodNotFound, "method %q not found", r.Method))
			}
		}
	}
	if conn.cancel == nil {
		// the default canceller does nothing
		conn.cancel = func(context.Context, *Conn, *Request) {}
	}
	if conn.log == nil {
		// the default logger does nothing
		conn.log = func(Direction, *ID, time.Duration, string, *json.RawMessage, *Error) {}
	}
	go func() {
		conn.err = conn.run(ctx)
		close(conn.done)
	}()
	return conn
}

// Wait blocks until the connection is terminated, and returns any error that
// cause the termination.
func (c *Conn) Wait(ctx context.Context) error {
	select {
	case <-c.done:
		return c.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Cancel cancels a pending Call on the server side.
// The call is identified by its id.
// JSON RPC 2 does not specify a cancel message, so cancellation support is not
// directly wired in. This method allows a higher level protocol to choose how
// to propagate the cancel.
func (c *Conn) Cancel(id ID) {
	c.handlingMu.Lock()
	handling, found := c.handling[id]
	c.handlingMu.Unlock()
	if found {
		handling.cancel()
	}
}

// Notify is called to send a notification request over the connection.
// It will return as soon as the notification has been sent, as no response is
// possible.
func (c *Conn) Notify(ctx context.Context, method string, params interface{}) error {
	jsonParams, err := marshalToRaw(params)
	if err != nil {
		return fmt.Errorf("marshalling notify parameters: %v", err)
	}
	request := &Request{
		Method: method,
		Params: jsonParams,
	}
	data, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("marshalling notify request: %v", err)
	}
	c.log(Send, nil, -1, request.Method, request.Params, nil)
	return c.stream.Write(ctx, data)
}

// Call sends a request over the connection and then waits for a response.
// If the response is not an error, it will be decoded into result.
// result must be of a type you an pass to json.Unmarshal.
func (c *Conn) Call(ctx context.Context, method string, params, result interface{}) error {
	jsonParams, err := marshalToRaw(params)
	if err != nil {
		return fmt.Errorf("marshalling call parameters: %v", err)
	}
	// generate a new request identifier
	id := ID{Number: atomic.AddInt64(&c.seq, 1)}
	request := &Request{
		ID:     &id,
		Method: method,
		Params: jsonParams,
	}
	// marshal the request now it is complete
	data, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("marshalling call request: %v", err)
	}
	// we have to add ourselves to the pending map before we send, otherwise we
	// are racing the response
	rchan := make(chan *Response)
	c.pendingMu.Lock()
	c.pending[id] = rchan
	c.pendingMu.Unlock()
	defer func() {
		// clean up the pending response handler on the way out
		c.pendingMu.Lock()
		delete(c.pending, id)
		c.pendingMu.Unlock()
	}()
	// now we are ready to send
	before := time.Now()
	c.log(Send, request.ID, -1, request.Method, request.Params, nil)
	if err := c.stream.Write(ctx, data); err != nil {
		// sending failed, we will never get a response, so don't leave it pending
		return err
	}
	// now wait for the response
	select {
	case response := <-rchan:
		elapsed := time.Since(before)
		c.log(Send, response.ID, elapsed, request.Method, response.Result, response.Error)
		// is it an error response?
		if response.Error != nil {
			return response.Error
		}
		if result == nil || response.Result == nil {
			return nil
		}
		if err := json.Unmarshal(*response.Result, result); err != nil {
			return fmt.Errorf("unmarshalling result: %v", err)
		}
		return nil
	case <-ctx.Done():
		// allow the handler to propagate the cancel
		c.cancel(ctx, c, request)
		return ctx.Err()
	}
}

// Reply sends a reply to the given request.
// It is an error to call this if request was not a call.
// You must call this exactly once for any given request.
// If err is set then result will be ignored.
func (c *Conn) Reply(ctx context.Context, req *Request, result interface{}, err error) error {
	if req.IsNotify() {
		return fmt.Errorf("reply not invoked with a valid call")
	}
	c.handlingMu.Lock()
	handling, found := c.handling[*req.ID]
	if found {
		delete(c.handling, *req.ID)
	}
	c.handlingMu.Unlock()
	if !found {
		return fmt.Errorf("not a call in progress: %v", req.ID)
	}

	elapsed := time.Since(handling.start)
	var raw *json.RawMessage
	if err == nil {
		raw, err = marshalToRaw(result)
	}
	response := &Response{
		Result: raw,
		ID:     req.ID,
	}
	if err != nil {
		if callErr, ok := err.(*Error); ok {
			response.Error = callErr
		} else {
			response.Error = NewErrorf(0, "%s", err)
		}
	}
	data, err := json.Marshal(response)
	if err != nil {
		return err
	}
	c.log(Send, response.ID, elapsed, req.Method, response.Result, response.Error)
	if err = c.stream.Write(ctx, data); err != nil {
		// TODO(iancottrell): if a stream write fails, we really need to shut down
		// the whole stream
		return err
	}
	return nil
}

type handling struct {
	request *Request
	cancel  context.CancelFunc
	start   time.Time
}

// combined has all the fields of both Request and Response.
// We can decode this and then work out which it is.
type combined struct {
	VersionTag VersionTag       `json:"jsonrpc"`
	ID         *ID              `json:"id,omitempty"`
	Method     string           `json:"method"`
	Params     *json.RawMessage `json:"params,omitempty"`
	Result     *json.RawMessage `json:"result,omitempty"`
	Error      *Error           `json:"error,omitempty"`
}

// Run starts a read loop on the supplied reader.
// It must be called exactly once for each Conn.
// It returns only when the reader is closed or there is an error in the stream.
func (c *Conn) run(ctx context.Context) error {
	for {
		// get the data for a message
		data, err := c.stream.Read(ctx)
		if err != nil {
			// the stream failed, we cannot continue
			return err
		}
		// read a combined message
		msg := &combined{}
		if err := json.Unmarshal(data, msg); err != nil {
			// a badly formed message arrived, log it and continue
			// we trust the stream to have isolated the error to just this message
			c.log(Receive, nil, -1, "", nil, NewErrorf(0, "unmarshal failed: %v", err))
			continue
		}
		// work out which kind of message we have
		switch {
		case msg.Method != "":
			// if method is set it must be a request
			request := &Request{
				Method: msg.Method,
				Params: msg.Params,
				ID:     msg.ID,
			}
			if request.IsNotify() {
				c.log(Receive, request.ID, -1, request.Method, request.Params, nil)
				// we have a Notify, forward to the handler in a go routine
				c.handle(ctx, c, request)
			} else {
				// we have a Call, forward to the handler in another go routine
				reqCtx, cancelReq := context.WithCancel(ctx)
				c.handlingMu.Lock()
				c.handling[*request.ID] = handling{
					request: request,
					cancel:  cancelReq,
					start:   time.Now(),
				}
				c.handlingMu.Unlock()
				c.log(Receive, request.ID, -1, request.Method, request.Params, nil)
				c.handle(reqCtx, c, request)
			}
		case msg.ID != nil:
			// we have a response, get the pending entry from the map
			c.pendingMu.Lock()
			rchan := c.pending[*msg.ID]
			if rchan != nil {
				delete(c.pending, *msg.ID)
			}
			c.pendingMu.Unlock()
			// and send the reply to the channel
			response := &Response{
				Result: msg.Result,
				Error:  msg.Error,
				ID:     msg.ID,
			}
			rchan <- response
			close(rchan)
		default:
			c.log(Receive, nil, -1, "", nil, NewErrorf(0, "message not a call, notify or response, ignoring"))
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
