// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync/atomic"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/lsp/debug/tag"
)

// Binder builds a connection configuration.
// This may be used in servers to generate a new configuration per connection.
// ConnectionOptions itself implements Binder returning itself unmodified, to
// allow for the simple cases where no per connection information is needed.
type Binder interface {
	// Bind returns the ConnectionOptions to use when establishing the passed-in
	// Connection.
	// The connection is not ready to use when Bind is called.
	Bind(context.Context, *Connection) (ConnectionOptions, error)
}

// ConnectionOptions holds the options for new connections.
type ConnectionOptions struct {
	// Framer allows control over the message framing and encoding.
	// If nil, HeaderFramer will be used.
	Framer Framer
	// Preempter allows registration of a pre-queue message handler.
	// If nil, no messages will be preempted.
	Preempter Preempter
	// Handler is used as the queued message handler for inbound messages.
	// If nil, all responses will be ErrNotHandled.
	Handler Handler
}

// Connection manages the jsonrpc2 protocol, connecting responses back to their
// calls.
// Connection is bidirectional; it does not have a designated server or client
// end.
type Connection struct {
	seq         int64 // must only be accessed using atomic operations
	closer      io.Closer
	writerBox   chan Writer
	outgoingBox chan map[ID]chan<- *Response
	incomingBox chan map[ID]*incoming
	async       *async
}

type AsyncCall struct {
	id        ID
	response  chan *Response // the channel a response will be delivered on
	resultBox chan asyncResult
	endSpan   func() // close the tracing span when all processing for the message is complete
}

type asyncResult struct {
	result []byte
	err    error
}

// incoming is used to track an incoming request as it is being handled
type incoming struct {
	request   *Request        // the request being processed
	baseCtx   context.Context // a base context for the message processing
	done      func()          // a function called when all processing for the message is complete
	handleCtx context.Context // the context for handling the message, child of baseCtx
	cancel    func()          // a function that cancels the handling context
}

// Bind returns the options unmodified.
func (o ConnectionOptions) Bind(context.Context, *Connection) (ConnectionOptions, error) {
	return o, nil
}

// newConnection creates a new connection and runs it.
// This is used by the Dial and Serve functions to build the actual connection.
func newConnection(ctx context.Context, rwc io.ReadWriteCloser, binder Binder) (*Connection, error) {
	c := &Connection{
		closer:      rwc,
		writerBox:   make(chan Writer, 1),
		outgoingBox: make(chan map[ID]chan<- *Response, 1),
		incomingBox: make(chan map[ID]*incoming, 1),
		async:       newAsync(),
	}

	options, err := binder.Bind(ctx, c)
	if err != nil {
		return nil, err
	}
	if options.Framer == nil {
		options.Framer = HeaderFramer()
	}
	if options.Preempter == nil {
		options.Preempter = defaultHandler{}
	}
	if options.Handler == nil {
		options.Handler = defaultHandler{}
	}
	c.outgoingBox <- make(map[ID]chan<- *Response)
	c.incomingBox <- make(map[ID]*incoming)
	// the goroutines started here will continue until the underlying stream is closed
	reader := options.Framer.Reader(rwc)
	readToQueue := make(chan *incoming)
	queueToDeliver := make(chan *incoming)
	go c.readIncoming(ctx, reader, readToQueue)
	go c.manageQueue(ctx, options.Preempter, readToQueue, queueToDeliver)
	go c.deliverMessages(ctx, options.Handler, queueToDeliver)

	// releaseing the writer must be the last thing we do in case any requests
	// are blocked waiting for the connection to be ready
	c.writerBox <- options.Framer.Writer(rwc)
	return c, nil
}

// Notify invokes the target method but does not wait for a response.
// The params will be marshaled to JSON before sending over the wire, and will
// be handed to the method invoked.
func (c *Connection) Notify(ctx context.Context, method string, params interface{}) error {
	notify, err := NewNotification(method, params)
	if err != nil {
		return fmt.Errorf("marshaling notify parameters: %v", err)
	}
	ctx, done := event.Start(ctx, method,
		tag.Method.Of(method),
		tag.RPCDirection.Of(tag.Outbound),
	)
	event.Metric(ctx, tag.Started.Of(1))
	err = c.write(ctx, notify)
	switch {
	case err != nil:
		event.Label(ctx, tag.StatusCode.Of("ERROR"))
	default:
		event.Label(ctx, tag.StatusCode.Of("OK"))
	}
	done()
	return err
}

// Call invokes the target method and returns an object that can be used to await the response.
// The params will be marshaled to JSON before sending over the wire, and will
// be handed to the method invoked.
// You do not have to wait for the response, it can just be ignored if not needed.
// If sending the call failed, the response will be ready and have the error in it.
func (c *Connection) Call(ctx context.Context, method string, params interface{}) *AsyncCall {
	result := &AsyncCall{
		id:        Int64ID(atomic.AddInt64(&c.seq, 1)),
		resultBox: make(chan asyncResult, 1),
	}
	// generate a new request identifier
	call, err := NewCall(result.id, method, params)
	if err != nil {
		//set the result to failed
		result.resultBox <- asyncResult{err: fmt.Errorf("marshaling call parameters: %w", err)}
		return result
	}
	ctx, endSpan := event.Start(ctx, method,
		tag.Method.Of(method),
		tag.RPCDirection.Of(tag.Outbound),
		tag.RPCID.Of(fmt.Sprintf("%q", result.id)),
	)
	result.endSpan = endSpan
	event.Metric(ctx, tag.Started.Of(1))
	// We have to add ourselves to the pending map before we send, otherwise we
	// are racing the response.
	// rchan is buffered in case the response arrives without a listener.
	result.response = make(chan *Response, 1)
	pending := <-c.outgoingBox
	pending[result.id] = result.response
	c.outgoingBox <- pending
	// now we are ready to send
	if err := c.write(ctx, call); err != nil {
		// sending failed, we will never get a response, so deliver a fake one
		r, _ := NewResponse(result.id, nil, err)
		c.incomingResponse(r)
	}
	return result
}

// ID used for this call.
// This can be used to cancel the call if needed.
func (a *AsyncCall) ID() ID { return a.id }

// IsReady can be used to check if the result is already prepared.
// This is guaranteed to return true on a result for which Await has already
// returned, or a call that failed to send in the first place.
func (a *AsyncCall) IsReady() bool {
	select {
	case r := <-a.resultBox:
		a.resultBox <- r
		return true
	default:
		return false
	}
}

// Await the results of a Call.
// The response will be unmarshaled from JSON into the result.
func (a *AsyncCall) Await(ctx context.Context, result interface{}) error {
	defer a.endSpan()
	var r asyncResult
	select {
	case response := <-a.response:
		// response just arrived, prepare the result
		switch {
		case response.Error != nil:
			r.err = response.Error
			event.Label(ctx, tag.StatusCode.Of("ERROR"))
		default:
			r.result = response.Result
			event.Label(ctx, tag.StatusCode.Of("OK"))
		}
	case r = <-a.resultBox:
		// result already available
	case <-ctx.Done():
		event.Label(ctx, tag.StatusCode.Of("CANCELLED"))
		return ctx.Err()
	}
	// refill the box for the next caller
	a.resultBox <- r
	// and unpack the result
	if r.err != nil {
		return r.err
	}
	if result == nil || len(r.result) == 0 {
		return nil
	}
	return json.Unmarshal(r.result, result)
}

// Respond delivers a response to an incoming Call.
//
// Respond must be called exactly once for any message for which a handler
// returns ErrAsyncResponse. It must not be called for any other message.
func (c *Connection) Respond(id ID, result interface{}, rerr error) error {
	pending := <-c.incomingBox
	defer func() { c.incomingBox <- pending }()
	entry, found := pending[id]
	if !found {
		return nil
	}
	delete(pending, id)
	return c.respond(entry, result, rerr)
}

// Cancel is used to cancel an inbound message by ID, it does not cancel
// outgoing messages.
// This is only used inside a message handler that is layering a
// cancellation protocol on top of JSON RPC 2.
// It will not complain if the ID is not a currently active message, and it will
// not cause any messages that have not arrived yet with that ID to be
// cancelled.
func (c *Connection) Cancel(id ID) {
	pending := <-c.incomingBox
	defer func() { c.incomingBox <- pending }()
	if entry, found := pending[id]; found && entry.cancel != nil {
		entry.cancel()
		entry.cancel = nil
	}
}

// Wait blocks until the connection is fully closed, but does not close it.
func (c *Connection) Wait() error {
	return c.async.wait()
}

// Close can be used to close the underlying stream, and then wait for the connection to
// fully shut down.
// This does not cancel in flight requests, but waits for them to gracefully complete.
func (c *Connection) Close() error {
	// close the underlying stream
	if err := c.closer.Close(); err != nil && !isClosingError(err) {
		return err
	}
	// and then wait for it to cause the connection to close
	if err := c.Wait(); err != nil && !isClosingError(err) {
		return err
	}
	return nil
}

// readIncoming collects inbound messages from the reader and delivers them, either responding
// to outgoing calls or feeding requests to the queue.
func (c *Connection) readIncoming(ctx context.Context, reader Reader, toQueue chan<- *incoming) {
	defer close(toQueue)
	for {
		// get the next message
		// no lock is needed, this is the only reader
		msg, n, err := reader.Read(ctx)
		if err != nil {
			// The stream failed, we cannot continue
			c.async.setError(err)
			return
		}
		switch msg := msg.(type) {
		case *Request:
			entry := &incoming{
				request: msg,
			}
			// add a span to the context for this request
			labels := append(make([]label.Label, 0, 3), // make space for the id if present
				tag.Method.Of(msg.Method),
				tag.RPCDirection.Of(tag.Inbound),
			)
			if msg.IsCall() {
				labels = append(labels, tag.RPCID.Of(fmt.Sprintf("%q", msg.ID)))
			}
			entry.baseCtx, entry.done = event.Start(ctx, msg.Method, labels...)
			event.Metric(entry.baseCtx,
				tag.Started.Of(1),
				tag.ReceivedBytes.Of(n))
			// in theory notifications cannot be cancelled, but we build them a cancel context anyway
			entry.handleCtx, entry.cancel = context.WithCancel(entry.baseCtx)
			// if the request is a call, add it to the incoming map so it can be
			// cancelled by id
			if msg.IsCall() {
				pending := <-c.incomingBox
				pending[msg.ID] = entry
				c.incomingBox <- pending
			}
			// send the message to the incoming queue
			toQueue <- entry
		case *Response:
			// If method is not set, this should be a response, in which case we must
			// have an id to send the response back to the caller.
			c.incomingResponse(msg)
		}
	}
}

func (c *Connection) incomingResponse(msg *Response) {
	pending := <-c.outgoingBox
	response, ok := pending[msg.ID]
	if ok {
		delete(pending, msg.ID)
	}
	c.outgoingBox <- pending
	if response != nil {
		response <- msg
	}
}

// manageQueue reads incoming requests, attempts to process them with the preempter, or queue them
// up for normal handling.
func (c *Connection) manageQueue(ctx context.Context, preempter Preempter, fromRead <-chan *incoming, toDeliver chan<- *incoming) {
	defer close(toDeliver)
	q := []*incoming{}
	ok := true
	for {
		var nextReq *incoming
		if len(q) == 0 {
			// no messages in the queue
			// if we were closing, then we are done
			if !ok {
				return
			}
			// not closing, but nothing in the queue, so just block waiting for a read
			nextReq, ok = <-fromRead
		} else {
			// we have a non empty queue, so pick whichever of reading or delivering
			// that we can make progress on
			select {
			case nextReq, ok = <-fromRead:
			case toDeliver <- q[0]:
				//TODO: this causes a lot of shuffling, should we use a growing ring buffer? compaction?
				q = q[1:]
			}
		}
		if nextReq != nil {
			// TODO: should we allow to limit the queue size?
			var result interface{}
			rerr := nextReq.handleCtx.Err()
			if rerr == nil {
				// only preempt if not already cancelled
				result, rerr = preempter.Preempt(nextReq.handleCtx, nextReq.request)
			}
			switch {
			case rerr == ErrNotHandled:
				// message not handled, add it to the queue for the main handler
				q = append(q, nextReq)
			case rerr == ErrAsyncResponse:
				// message handled but the response will come later
			default:
				// anything else means the message is fully handled
				c.reply(nextReq, result, rerr)
			}
		}
	}
}

func (c *Connection) deliverMessages(ctx context.Context, handler Handler, fromQueue <-chan *incoming) {
	defer c.async.done()
	for entry := range fromQueue {
		// cancel any messages in the queue that we have a pending cancel for
		var result interface{}
		rerr := entry.handleCtx.Err()
		if rerr == nil {
			// only deliver if not already cancelled
			result, rerr = handler.Handle(entry.handleCtx, entry.request)
		}
		switch {
		case rerr == ErrNotHandled:
			// message not handled, report it back to the caller as an error
			c.reply(entry, nil, fmt.Errorf("%w: %q", ErrMethodNotFound, entry.request.Method))
		case rerr == ErrAsyncResponse:
			// message handled but the response will come later
		default:
			c.reply(entry, result, rerr)
		}
	}
}

// reply is used to reply to an incoming request that has just been handled
func (c *Connection) reply(entry *incoming, result interface{}, rerr error) {
	if entry.request.IsCall() {
		// we have a call finishing, remove it from the incoming map
		pending := <-c.incomingBox
		defer func() { c.incomingBox <- pending }()
		delete(pending, entry.request.ID)
	}
	if err := c.respond(entry, result, rerr); err != nil {
		// no way to propagate this error
		//TODO: should we do more than just log it?
		event.Error(entry.baseCtx, "jsonrpc2 message delivery failed", err)
	}
}

// respond sends a response.
// This is the code shared between reply and SendResponse.
func (c *Connection) respond(entry *incoming, result interface{}, rerr error) error {
	var err error
	if entry.request.IsCall() {
		// send the response
		if result == nil && rerr == nil {
			// call with no response, send an error anyway
			rerr = fmt.Errorf("%w: %q produced no response", ErrInternal, entry.request.Method)
		}
		var response *Response
		response, err = NewResponse(entry.request.ID, result, rerr)
		if err == nil {
			// we write the response with the base context, in case the message was cancelled
			err = c.write(entry.baseCtx, response)
		}
	} else {
		switch {
		case rerr != nil:
			// notification failed
			err = fmt.Errorf("%w: %q notification failed: %v", ErrInternal, entry.request.Method, rerr)
			rerr = nil
		case result != nil:
			//notification produced a response, which is an error
			err = fmt.Errorf("%w: %q produced unwanted response", ErrInternal, entry.request.Method)
		default:
			// normal notification finish
		}
	}
	switch {
	case rerr != nil || err != nil:
		event.Label(entry.baseCtx, tag.StatusCode.Of("ERROR"))
	default:
		event.Label(entry.baseCtx, tag.StatusCode.Of("OK"))
	}
	// and just to be clean, invoke and clear the cancel if needed
	if entry.cancel != nil {
		entry.cancel()
		entry.cancel = nil
	}
	// mark the entire request processing as done
	entry.done()
	return err
}

// write is used by all things that write outgoing messages, including replies.
// it makes sure that writes are atomic
func (c *Connection) write(ctx context.Context, msg Message) error {
	writer := <-c.writerBox
	defer func() { c.writerBox <- writer }()
	n, err := writer.Write(ctx, msg)
	event.Metric(ctx, tag.SentBytes.Of(n))
	return err
}
