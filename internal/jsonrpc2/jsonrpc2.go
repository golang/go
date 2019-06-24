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

	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/stats"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
)

// Conn is a JSON RPC 2 client server connection.
// Conn is bidirectional; it does not have a designated server or client end.
type Conn struct {
	seq                int64 // must only be accessed using atomic operations
	Handler            Handler
	Canceler           Canceler
	Logger             Logger
	Capacity           int
	RejectIfOverloaded bool
	stream             Stream
	err                error
	pendingMu          sync.Mutex // protects the pending map
	pending            map[ID]chan *wireResponse
	handlingMu         sync.Mutex // protects the handling map
	handling           map[ID]handling
}

// Request is sent to a server to represent a Call or Notify operaton.
type Request struct {
	conn *Conn

	// Method is a string containing the method name to invoke.
	Method string
	// Params is either a struct or an array with the parameters of the method.
	Params *json.RawMessage
	// The id of this request, used to tie the response back to the request.
	// Will be either a string or a number. If not set, the request is a notify,
	// and no response is possible.
	ID *ID
}

type queueEntry struct {
	ctx  context.Context
	r    *Request
	size int64
}

// Handler is an option you can pass to NewConn to handle incoming requests.
// If the request returns false from IsNotify then the Handler must eventually
// call Reply on the Conn with the supplied request.
// Handlers are called synchronously, they should pass the work off to a go
// routine if they are going to take a long time.
type Handler func(context.Context, *Request)

// Canceler is an option you can pass to NewConn which is invoked for
// cancelled outgoing requests.
// It is okay to use the connection to send notifications, but the context will
// be in the cancelled state, so you must do it with the background context
// instead.
type Canceler func(context.Context, *Conn, ID)

type rpcStats struct {
	server   bool
	method   string
	ctx      context.Context
	span     trace.Span
	start    time.Time
	received int64
	sent     int64
}

type statsKeyType string

const rpcStatsKey = statsKeyType("rpcStatsKey")

func start(ctx context.Context, server bool, method string, id *ID) (context.Context, *rpcStats) {
	s := &rpcStats{
		server: server,
		method: method,
		ctx:    ctx,
		start:  time.Now(),
	}
	s.ctx = context.WithValue(s.ctx, rpcStatsKey, s)
	tags := make([]tag.Mutator, 0, 4)
	tags = append(tags, tag.Upsert(telemetry.KeyMethod, method))
	mode := telemetry.Outbound
	spanKind := trace.SpanKindClient
	if server {
		spanKind = trace.SpanKindServer
		mode = telemetry.Inbound
	}
	tags = append(tags, tag.Upsert(telemetry.KeyRPCDirection, mode))
	if id != nil {
		tags = append(tags, tag.Upsert(telemetry.KeyRPCID, id.String()))
	}
	s.ctx, s.span = trace.StartSpan(ctx, method, trace.WithSpanKind(spanKind))
	s.ctx, _ = tag.New(s.ctx, tags...)
	stats.Record(s.ctx, telemetry.Started.M(1))
	return s.ctx, s
}

func (s *rpcStats) end(ctx context.Context, err *error) {
	if err != nil && *err != nil {
		ctx, _ = tag.New(ctx, tag.Upsert(telemetry.KeyStatus, "ERROR"))
	} else {
		ctx, _ = tag.New(ctx, tag.Upsert(telemetry.KeyStatus, "OK"))
	}
	elapsedTime := time.Since(s.start)
	latencyMillis := float64(elapsedTime) / float64(time.Millisecond)

	stats.Record(ctx,
		telemetry.ReceivedBytes.M(s.received),
		telemetry.SentBytes.M(s.sent),
		telemetry.Latency.M(latencyMillis),
	)

	s.span.End()
}

// NewErrorf builds a Error struct for the suppied message and code.
// If args is not empty, message and args will be passed to Sprintf.
func NewErrorf(code int64, format string, args ...interface{}) *Error {
	return &Error{
		Code:    code,
		Message: fmt.Sprintf(format, args...),
	}
}

// NewConn creates a new connection object around the supplied stream.
// You must call Run for the connection to be active.
func NewConn(s Stream) *Conn {
	conn := &Conn{
		stream:   s,
		pending:  make(map[ID]chan *wireResponse),
		handling: make(map[ID]handling),
	}
	// the default handler reports a method error
	conn.Handler = func(ctx context.Context, r *Request) {
		if r.IsNotify() {
			r.Reply(ctx, nil, NewErrorf(CodeMethodNotFound, "method %q not found", r.Method))
		}
	}
	// the default canceler does nothing
	conn.Canceler = func(context.Context, *Conn, ID) {}
	// the default logger does nothing
	conn.Logger = func(Direction, *ID, time.Duration, string, *json.RawMessage, *Error) {}
	return conn
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
func (c *Conn) Notify(ctx context.Context, method string, params interface{}) (err error) {
	ctx, rpcStats := start(ctx, false, method, nil)
	defer rpcStats.end(ctx, &err)

	jsonParams, err := marshalToRaw(params)
	if err != nil {
		return fmt.Errorf("marshalling notify parameters: %v", err)
	}
	request := &wireRequest{
		Method: method,
		Params: jsonParams,
	}
	data, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("marshalling notify request: %v", err)
	}
	c.Logger(Send, nil, -1, request.Method, request.Params, nil)
	n, err := c.stream.Write(ctx, data)
	rpcStats.sent += n
	return err
}

// Call sends a request over the connection and then waits for a response.
// If the response is not an error, it will be decoded into result.
// result must be of a type you an pass to json.Unmarshal.
func (c *Conn) Call(ctx context.Context, method string, params, result interface{}) (err error) {
	// generate a new request identifier
	id := ID{Number: atomic.AddInt64(&c.seq, 1)}
	ctx, rpcStats := start(ctx, false, method, &id)
	defer rpcStats.end(ctx, &err)
	jsonParams, err := marshalToRaw(params)
	if err != nil {
		return fmt.Errorf("marshalling call parameters: %v", err)
	}
	request := &wireRequest{
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
	rchan := make(chan *wireResponse)
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
	c.Logger(Send, request.ID, -1, request.Method, request.Params, nil)
	n, err := c.stream.Write(ctx, data)
	rpcStats.sent += n
	if err != nil {
		// sending failed, we will never get a response, so don't leave it pending
		return err
	}
	// now wait for the response
	select {
	case response := <-rchan:
		elapsed := time.Since(before)
		c.Logger(Receive, response.ID, elapsed, request.Method, response.Result, response.Error)
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
		c.Canceler(ctx, c, id)
		return ctx.Err()
	}
}

// Conn returns the connection that created this request.
func (r *Request) Conn() *Conn { return r.conn }

// IsNotify returns true if this request is a notification.
func (r *Request) IsNotify() bool {
	return r.ID == nil
}

// Reply sends a reply to the given request.
// It is an error to call this if request was not a call.
// You must call this exactly once for any given request.
// If err is set then result will be ignored.
func (r *Request) Reply(ctx context.Context, result interface{}, err error) error {
	ctx, st := trace.StartSpan(ctx, r.Method+":reply", trace.WithSpanKind(trace.SpanKindClient))
	defer st.End()

	if r.IsNotify() {
		return fmt.Errorf("reply not invoked with a valid call")
	}
	r.conn.handlingMu.Lock()
	handling, found := r.conn.handling[*r.ID]
	if found {
		delete(r.conn.handling, *r.ID)
	}
	r.conn.handlingMu.Unlock()
	if !found {
		return fmt.Errorf("not a call in progress: %v", r.ID)
	}

	elapsed := time.Since(handling.start)
	var raw *json.RawMessage
	if err == nil {
		raw, err = marshalToRaw(result)
	}
	response := &wireResponse{
		Result: raw,
		ID:     r.ID,
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
	r.conn.Logger(Send, response.ID, elapsed, r.Method, response.Result, response.Error)
	n, err := r.conn.stream.Write(ctx, data)

	v := ctx.Value(rpcStatsKey)
	if v != nil {
		s := v.(*rpcStats)
		s.sent += n
	} else {
		//panic("no stats available in reply")
	}

	if err != nil {
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

func (c *Conn) deliver(ctx context.Context, q chan queueEntry, request *Request, size int64) bool {
	e := queueEntry{ctx: ctx, r: request, size: size}
	if !c.RejectIfOverloaded {
		q <- e
		return true
	}
	select {
	case q <- e:
		return true
	default:
		return false
	}
}

// Run blocks until the connection is terminated, and returns any error that
// caused the termination.
// It must be called exactly once for each Conn.
// It returns only when the reader is closed or there is an error in the stream.
func (c *Conn) Run(ctx context.Context) error {
	q := make(chan queueEntry, c.Capacity)
	defer close(q)
	// start the queue processor
	go func() {
		// TODO: idle notification?
		for e := range q {
			if e.ctx.Err() != nil {
				continue
			}
			ctx, rpcStats := start(ctx, true, e.r.Method, e.r.ID)
			rpcStats.received += e.size
			c.Handler(ctx, e.r)
			rpcStats.end(ctx, nil)
		}
	}()
	for {
		// get the data for a message
		data, n, err := c.stream.Read(ctx)
		if err != nil {
			// the stream failed, we cannot continue
			return err
		}
		// read a combined message
		msg := &combined{}
		if err := json.Unmarshal(data, msg); err != nil {
			// a badly formed message arrived, log it and continue
			// we trust the stream to have isolated the error to just this message
			c.Logger(Receive, nil, -1, "", nil, NewErrorf(0, "unmarshal failed: %v", err))
			continue
		}
		// work out which kind of message we have
		switch {
		case msg.Method != "":
			// if method is set it must be a request
			request := &Request{
				conn:   c,
				Method: msg.Method,
				Params: msg.Params,
				ID:     msg.ID,
			}
			if request.IsNotify() {
				c.Logger(Receive, request.ID, -1, request.Method, request.Params, nil)
				// we have a Notify, add to the processor queue
				c.deliver(ctx, q, request, n)
				//TODO: log when we drop a message?
			} else {
				// we have a Call, add to the processor queue
				reqCtx, cancelReq := context.WithCancel(ctx)
				c.handlingMu.Lock()
				c.handling[*request.ID] = handling{
					request: request,
					cancel:  cancelReq,
					start:   time.Now(),
				}
				c.handlingMu.Unlock()
				c.Logger(Receive, request.ID, -1, request.Method, request.Params, nil)
				if !c.deliver(reqCtx, q, request, n) {
					// queue is full, reject the message by directly replying
					request.Reply(ctx, nil, NewErrorf(CodeServerOverloaded, "no room in queue"))
				}
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
			response := &wireResponse{
				Result: msg.Result,
				Error:  msg.Error,
				ID:     msg.ID,
			}
			rchan <- response
			close(rchan)
		default:
			c.Logger(Receive, nil, -1, "", nil, NewErrorf(0, "message not a call, notify or response, ignoring"))
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
