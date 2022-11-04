// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/event/tag"
)

// Binder builds a connection configuration.
// This may be used in servers to generate a new configuration per connection.
// ConnectionOptions itself implements Binder returning itself unmodified, to
// allow for the simple cases where no per connection information is needed.
type Binder interface {
	// Bind returns the ConnectionOptions to use when establishing the passed-in
	// Connection.
	//
	// The connection is not ready to use when Bind is called,
	// but Bind may close it without reading or writing to it.
	Bind(context.Context, *Connection) ConnectionOptions
}

// A BinderFunc implements the Binder interface for a standalone Bind function.
type BinderFunc func(context.Context, *Connection) ConnectionOptions

func (f BinderFunc) Bind(ctx context.Context, c *Connection) ConnectionOptions {
	return f(ctx, c)
}

var _ Binder = BinderFunc(nil)

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
	// OnInternalError, if non-nil, is called with any internal errors that occur
	// while serving the connection, such as protocol errors or invariant
	// violations. (If nil, internal errors result in panics.)
	OnInternalError func(error)
}

// Connection manages the jsonrpc2 protocol, connecting responses back to their
// calls.
// Connection is bidirectional; it does not have a designated server or client
// end.
type Connection struct {
	seq int64 // must only be accessed using atomic operations

	stateMu sync.Mutex
	state   inFlightState // accessed only in updateInFlight
	done    chan struct{} // closed (under stateMu) when state.closed is true and all goroutines have completed

	writer chan Writer // 1-buffered; stores the writer when not in use

	handler Handler

	onInternalError func(error)
	onDone          func()
}

// inFlightState records the state of the incoming and outgoing calls on a
// Connection.
type inFlightState struct {
	connClosing bool  // true when the Connection's Close method has been called
	reading     bool  // true while the readIncoming goroutine is running
	readErr     error // non-nil when the readIncoming goroutine exits (typically io.EOF)
	writeErr    error // non-nil if a call to the Writer has failed with a non-canceled Context

	// closer shuts down and cleans up the Reader and Writer state, ideally
	// interrupting any Read or Write call that is currently blocked. It is closed
	// when the state is idle and one of: connClosing is true, readErr is non-nil,
	// or writeErr is non-nil.
	//
	// After the closer has been invoked, the closer field is set to nil
	// and the closeErr field is simultaneously set to its result.
	closer   io.Closer
	closeErr error // error returned from closer.Close

	outgoingCalls         map[ID]*AsyncCall // calls only
	outgoingNotifications int               // # of notifications awaiting "write"

	// incoming stores the total number of incoming calls and notifications
	// that have not yet written or processed a result.
	incoming int

	incomingByID map[ID]*incomingRequest // calls only

	// handlerQueue stores the backlog of calls and notifications that were not
	// already handled by a preempter.
	// The queue does not include the request currently being handled (if any).
	handlerQueue   []*incomingRequest
	handlerRunning bool
}

// updateInFlight locks the state of the connection's in-flight requests, allows
// f to mutate that state, and closes the connection if it is idle and either
// is closing or has a read or write error.
func (c *Connection) updateInFlight(f func(*inFlightState)) {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()

	s := &c.state

	f(s)

	select {
	case <-c.done:
		// The connection was already completely done at the start of this call to
		// updateInFlight, so it must remain so. (The call to f should have noticed
		// that and avoided making any updates that would cause the state to be
		// non-idle.)
		if !s.idle() {
			panic("jsonrpc2_v2: updateInFlight transitioned to non-idle when already done")
		}
		return
	default:
	}

	if s.idle() && s.shuttingDown(ErrUnknown) != nil {
		if s.closer != nil {
			s.closeErr = s.closer.Close()
			s.closer = nil // prevent duplicate Close calls
		}
		if s.reading {
			// The readIncoming goroutine is still running. Our call to Close should
			// cause it to exit soon, at which point it will make another call to
			// updateInFlight, set s.reading to false, and mark the Connection done.
		} else {
			// The readIncoming goroutine has exited, or never started to begin with.
			// Since everything else is idle, we're completely done.
			if c.onDone != nil {
				c.onDone()
			}
			close(c.done)
		}
	}
}

// idle reports whether the connction is in a state with no pending calls or
// notifications.
//
// If idle returns true, the readIncoming goroutine may still be running,
// but no other goroutines are doing work on behalf of the connnection.
func (s *inFlightState) idle() bool {
	return len(s.outgoingCalls) == 0 && s.outgoingNotifications == 0 && s.incoming == 0 && !s.handlerRunning
}

// shuttingDown reports whether the connection is in a state that should
// disallow new (incoming and outgoing) calls. It returns either nil or
// an error that is or wraps the provided errClosing.
func (s *inFlightState) shuttingDown(errClosing error) error {
	if s.connClosing {
		// If Close has been called explicitly, it doesn't matter what state the
		// Reader and Writer are in: we shouldn't be starting new work because the
		// caller told us not to start new work.
		return errClosing
	}
	if s.readErr != nil {
		// If the read side of the connection is broken, we cannot read new call
		// requests, and cannot read responses to our outgoing calls.
		return fmt.Errorf("%w: %v", errClosing, s.readErr)
	}
	if s.writeErr != nil {
		// If the write side of the connection is broken, we cannot write responses
		// for incoming calls, and cannot write requests for outgoing calls.
		return fmt.Errorf("%w: %v", errClosing, s.writeErr)
	}
	return nil
}

// incomingRequest is used to track an incoming request as it is being handled
type incomingRequest struct {
	*Request // the request being processed
	ctx      context.Context
	cancel   context.CancelFunc
	endSpan  func() // called (and set to nil) when the response is sent
}

// Bind returns the options unmodified.
func (o ConnectionOptions) Bind(context.Context, *Connection) ConnectionOptions {
	return o
}

// newConnection creates a new connection and runs it.
//
// This is used by the Dial and Serve functions to build the actual connection.
//
// The connection is closed automatically (and its resources cleaned up) when
// the last request has completed after the underlying ReadWriteCloser breaks,
// but it may be stopped earlier by calling Close (for a clean shutdown).
func newConnection(bindCtx context.Context, rwc io.ReadWriteCloser, binder Binder, onDone func()) *Connection {
	// TODO: Should we create a new event span here?
	// This will propagate cancellation from ctx; should it?
	ctx := notDone{bindCtx}

	c := &Connection{
		state:  inFlightState{closer: rwc},
		done:   make(chan struct{}),
		writer: make(chan Writer, 1),
		onDone: onDone,
	}
	// It's tempting to set a finalizer on c to verify that the state has gone
	// idle when the connection becomes unreachable. Unfortunately, the Binder
	// interface makes that unsafe: it allows the Handler to close over the
	// Connection, which could create a reference cycle that would cause the
	// Connection to become uncollectable.

	options := binder.Bind(bindCtx, c)
	framer := options.Framer
	if framer == nil {
		framer = HeaderFramer()
	}
	c.handler = options.Handler
	if c.handler == nil {
		c.handler = defaultHandler{}
	}
	c.onInternalError = options.OnInternalError

	c.writer <- framer.Writer(rwc)
	reader := framer.Reader(rwc)

	c.updateInFlight(func(s *inFlightState) {
		select {
		case <-c.done:
			// Bind already closed the connection; don't start a goroutine to read it.
			return
		default:
		}

		// The goroutine started here will continue until the underlying stream is closed.
		//
		// (If the Binder closed the Connection already, this should error out and
		// return almost immediately.)
		s.reading = true
		go c.readIncoming(ctx, reader, options.Preempter)
	})
	return c
}

// Notify invokes the target method but does not wait for a response.
// The params will be marshaled to JSON before sending over the wire, and will
// be handed to the method invoked.
func (c *Connection) Notify(ctx context.Context, method string, params interface{}) (err error) {
	ctx, done := event.Start(ctx, method,
		tag.Method.Of(method),
		tag.RPCDirection.Of(tag.Outbound),
	)
	attempted := false

	defer func() {
		labelStatus(ctx, err)
		done()
		if attempted {
			c.updateInFlight(func(s *inFlightState) {
				s.outgoingNotifications--
			})
		}
	}()

	c.updateInFlight(func(s *inFlightState) {
		// If the connection is shutting down, allow outgoing notifications only if
		// there is at least one call still in flight. The number of calls in flight
		// cannot increase once shutdown begins, and allowing outgoing notifications
		// may permit notifications that will cancel in-flight calls.
		if len(s.outgoingCalls) == 0 && len(s.incomingByID) == 0 {
			err = s.shuttingDown(ErrClientClosing)
			if err != nil {
				return
			}
		}
		s.outgoingNotifications++
		attempted = true
	})
	if err != nil {
		return err
	}

	notify, err := NewNotification(method, params)
	if err != nil {
		return fmt.Errorf("marshaling notify parameters: %v", err)
	}

	event.Metric(ctx, tag.Started.Of(1))
	return c.write(ctx, notify)
}

// Call invokes the target method and returns an object that can be used to await the response.
// The params will be marshaled to JSON before sending over the wire, and will
// be handed to the method invoked.
// You do not have to wait for the response, it can just be ignored if not needed.
// If sending the call failed, the response will be ready and have the error in it.
func (c *Connection) Call(ctx context.Context, method string, params interface{}) *AsyncCall {
	// Generate a new request identifier.
	id := Int64ID(atomic.AddInt64(&c.seq, 1))
	ctx, endSpan := event.Start(ctx, method,
		tag.Method.Of(method),
		tag.RPCDirection.Of(tag.Outbound),
		tag.RPCID.Of(fmt.Sprintf("%q", id)),
	)

	ac := &AsyncCall{
		id:      id,
		ready:   make(chan struct{}),
		ctx:     ctx,
		endSpan: endSpan,
	}
	// When this method returns, either ac is retired, or the request has been
	// written successfully and the call is awaiting a response (to be provided by
	// the readIncoming goroutine).

	call, err := NewCall(ac.id, method, params)
	if err != nil {
		ac.retire(&Response{ID: id, Error: fmt.Errorf("marshaling call parameters: %w", err)})
		return ac
	}

	c.updateInFlight(func(s *inFlightState) {
		err = s.shuttingDown(ErrClientClosing)
		if err != nil {
			return
		}
		if s.outgoingCalls == nil {
			s.outgoingCalls = make(map[ID]*AsyncCall)
		}
		s.outgoingCalls[ac.id] = ac
	})
	if err != nil {
		ac.retire(&Response{ID: id, Error: err})
		return ac
	}

	event.Metric(ctx, tag.Started.Of(1))
	if err := c.write(ctx, call); err != nil {
		// Sending failed. We will never get a response, so deliver a fake one if it
		// wasn't already retired by the connection breaking.
		c.updateInFlight(func(s *inFlightState) {
			if s.outgoingCalls[ac.id] == ac {
				delete(s.outgoingCalls, ac.id)
				ac.retire(&Response{ID: id, Error: err})
			} else {
				// ac was already retired by the readIncoming goroutine:
				// perhaps our write raced with the Read side of the connection breaking.
			}
		})
	}
	return ac
}

type AsyncCall struct {
	id       ID
	ready    chan struct{} // closed after response has been set and span has been ended
	response *Response
	ctx      context.Context // for event logging only
	endSpan  func()          // close the tracing span when all processing for the message is complete
}

// ID used for this call.
// This can be used to cancel the call if needed.
func (ac *AsyncCall) ID() ID { return ac.id }

// IsReady can be used to check if the result is already prepared.
// This is guaranteed to return true on a result for which Await has already
// returned, or a call that failed to send in the first place.
func (ac *AsyncCall) IsReady() bool {
	select {
	case <-ac.ready:
		return true
	default:
		return false
	}
}

// retire processes the response to the call.
func (ac *AsyncCall) retire(response *Response) {
	select {
	case <-ac.ready:
		panic(fmt.Sprintf("jsonrpc2: retire called twice for ID %v", ac.id))
	default:
	}

	ac.response = response
	labelStatus(ac.ctx, response.Error)
	ac.endSpan()
	// Allow the trace context, which may retain a lot of reachable values,
	// to be garbage-collected.
	ac.ctx, ac.endSpan = nil, nil

	close(ac.ready)
}

// Await waits for (and decodes) the results of a Call.
// The response will be unmarshaled from JSON into the result.
func (ac *AsyncCall) Await(ctx context.Context, result interface{}) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-ac.ready:
	}
	if ac.response.Error != nil {
		return ac.response.Error
	}
	if result == nil {
		return nil
	}
	return json.Unmarshal(ac.response.Result, result)
}

// Respond delivers a response to an incoming Call.
//
// Respond must be called exactly once for any message for which a handler
// returns ErrAsyncResponse. It must not be called for any other message.
func (c *Connection) Respond(id ID, result interface{}, err error) error {
	var req *incomingRequest
	c.updateInFlight(func(s *inFlightState) {
		req = s.incomingByID[id]
	})
	if req == nil {
		return c.internalErrorf("Request not found for ID %v", id)
	}

	if err == ErrAsyncResponse {
		// Respond is supposed to supply the asynchronous response, so it would be
		// confusing to call Respond with an error that promises to call Respond
		// again.
		err = c.internalErrorf("Respond called with ErrAsyncResponse for %q", req.Method)
	}
	return c.processResult("Respond", req, result, err)
}

// Cancel cancels the Context passed to the Handle call for the inbound message
// with the given ID.
//
// Cancel will not complain if the ID is not a currently active message, and it
// will not cause any messages that have not arrived yet with that ID to be
// cancelled.
func (c *Connection) Cancel(id ID) {
	var req *incomingRequest
	c.updateInFlight(func(s *inFlightState) {
		req = s.incomingByID[id]
	})
	if req != nil {
		req.cancel()
	}
}

// Wait blocks until the connection is fully closed, but does not close it.
func (c *Connection) Wait() error {
	var err error
	<-c.done
	c.updateInFlight(func(s *inFlightState) {
		err = s.closeErr
	})
	return err
}

// Close stops accepting new requests, waits for in-flight requests and enqueued
// Handle calls to complete, and then closes the underlying stream.
//
// After the start of a Close, notification requests (that lack IDs and do not
// receive responses) will continue to be passed to the Preempter, but calls
// with IDs will receive immediate responses with ErrServerClosing, and no new
// requests (not even notifications!) will be enqueued to the Handler.
func (c *Connection) Close() error {
	// Stop handling new requests, and interrupt the reader (by closing the
	// connection) as soon as the active requests finish.
	c.updateInFlight(func(s *inFlightState) { s.connClosing = true })

	return c.Wait()
}

// readIncoming collects inbound messages from the reader and delivers them, either responding
// to outgoing calls or feeding requests to the queue.
func (c *Connection) readIncoming(ctx context.Context, reader Reader, preempter Preempter) {
	var err error
	for {
		var (
			msg Message
			n   int64
		)
		msg, n, err = reader.Read(ctx)
		if err != nil {
			break
		}

		switch msg := msg.(type) {
		case *Request:
			c.acceptRequest(ctx, msg, n, preempter)

		case *Response:
			c.updateInFlight(func(s *inFlightState) {
				if ac, ok := s.outgoingCalls[msg.ID]; ok {
					delete(s.outgoingCalls, msg.ID)
					ac.retire(msg)
				} else {
					// TODO: How should we report unexpected responses?
				}
			})

		default:
			c.internalErrorf("Read returned an unexpected message of type %T", msg)
		}
	}

	c.updateInFlight(func(s *inFlightState) {
		s.reading = false
		s.readErr = err

		// Retire any outgoing requests that were still in flight: with the Reader no
		// longer being processed, they necessarily cannot receive a response.
		for id, ac := range s.outgoingCalls {
			ac.retire(&Response{ID: id, Error: err})
		}
		s.outgoingCalls = nil
	})
}

// acceptRequest either handles msg synchronously or enqueues it to be handled
// asynchronously.
func (c *Connection) acceptRequest(ctx context.Context, msg *Request, msgBytes int64, preempter Preempter) {
	// Add a span to the context for this request.
	labels := append(make([]label.Label, 0, 3), // Make space for the ID if present.
		tag.Method.Of(msg.Method),
		tag.RPCDirection.Of(tag.Inbound),
	)
	if msg.IsCall() {
		labels = append(labels, tag.RPCID.Of(fmt.Sprintf("%q", msg.ID)))
	}
	ctx, endSpan := event.Start(ctx, msg.Method, labels...)
	event.Metric(ctx,
		tag.Started.Of(1),
		tag.ReceivedBytes.Of(msgBytes))

	// In theory notifications cannot be cancelled, but we build them a cancel
	// context anyway.
	ctx, cancel := context.WithCancel(ctx)
	req := &incomingRequest{
		Request: msg,
		ctx:     ctx,
		cancel:  cancel,
		endSpan: endSpan,
	}

	// If the request is a call, add it to the incoming map so it can be
	// cancelled (or responded) by ID.
	var err error
	c.updateInFlight(func(s *inFlightState) {
		s.incoming++

		if req.IsCall() {
			if s.incomingByID[req.ID] != nil {
				err = fmt.Errorf("%w: request ID %v already in use", ErrInvalidRequest, req.ID)
				req.ID = ID{} // Don't misattribute this error to the existing request.
				return
			}

			if s.incomingByID == nil {
				s.incomingByID = make(map[ID]*incomingRequest)
			}
			s.incomingByID[req.ID] = req

			// When shutting down, reject all new Call requests, even if they could
			// theoretically be handled by the preempter. The preempter could return
			// ErrAsyncResponse, which would increase the amount of work in flight
			// when we're trying to ensure that it strictly decreases.
			err = s.shuttingDown(ErrServerClosing)
		}
	})
	if err != nil {
		c.processResult("acceptRequest", req, nil, err)
		return
	}

	if preempter != nil {
		result, err := preempter.Preempt(req.ctx, req.Request)

		if req.IsCall() && errors.Is(err, ErrAsyncResponse) {
			// This request will remain in flight until Respond is called for it.
			return
		}

		if !errors.Is(err, ErrNotHandled) {
			c.processResult("Preempt", req, result, err)
			return
		}
	}

	c.updateInFlight(func(s *inFlightState) {
		// If the connection is shutting down, don't enqueue anything to the
		// handler — not even notifications. That ensures that if the handler
		// continues to make progress, it will eventually become idle and
		// close the connection.
		err = s.shuttingDown(ErrServerClosing)
		if err != nil {
			return
		}

		// We enqueue requests that have not been preempted to an unbounded slice.
		// Unfortunately, we cannot in general limit the size of the handler
		// queue: we have to read every response that comes in on the wire
		// (because it may be responding to a request issued by, say, an
		// asynchronous handler), and in order to get to that response we have
		// to read all of the requests that came in ahead of it.
		s.handlerQueue = append(s.handlerQueue, req)
		if !s.handlerRunning {
			// We start the handleAsync goroutine when it has work to do, and let it
			// exit when the queue empties.
			//
			// Otherwise, in order to synchronize the handler we would need some other
			// goroutine (probably readIncoming?) to explicitly wait for handleAsync
			// to finish, and that would complicate error reporting: either the error
			// report from the goroutine would be blocked on the handler emptying its
			// queue (which was tried, and introduced a deadlock detected by
			// TestCloseCallRace), or the error would need to be reported separately
			// from synchronizing completion. Allowing the handler goroutine to exit
			// when idle seems simpler than trying to implement either of those
			// alternatives correctly.
			s.handlerRunning = true
			go c.handleAsync()
		}
	})
	if err != nil {
		c.processResult("acceptRequest", req, nil, err)
	}
}

// handleAsync invokes the handler on the requests in the handler queue
// sequentially until the queue is empty.
func (c *Connection) handleAsync() {
	for {
		var req *incomingRequest
		c.updateInFlight(func(s *inFlightState) {
			if len(s.handlerQueue) > 0 {
				req, s.handlerQueue = s.handlerQueue[0], s.handlerQueue[1:]
			} else {
				s.handlerRunning = false
			}
		})
		if req == nil {
			return
		}

		// Only deliver to the Handler if not already canceled.
		if err := req.ctx.Err(); err != nil {
			c.updateInFlight(func(s *inFlightState) {
				if s.writeErr != nil {
					// Assume that req.ctx was canceled due to s.writeErr.
					// TODO(#51365): use a Context API to plumb this through req.ctx.
					err = fmt.Errorf("%w: %v", ErrServerClosing, s.writeErr)
				}
			})
			c.processResult("handleAsync", req, nil, err)
			continue
		}

		result, err := c.handler.Handle(req.ctx, req.Request)
		c.processResult(c.handler, req, result, err)
	}
}

// processResult processes the result of a request and, if appropriate, sends a response.
func (c *Connection) processResult(from interface{}, req *incomingRequest, result interface{}, err error) error {
	switch err {
	case ErrAsyncResponse:
		if !req.IsCall() {
			return c.internalErrorf("%#v returned ErrAsyncResponse for a %q Request without an ID", from, req.Method)
		}
		return nil // This request is still in flight, so don't record the result yet.
	case ErrNotHandled, ErrMethodNotFound:
		// Add detail describing the unhandled method.
		err = fmt.Errorf("%w: %q", ErrMethodNotFound, req.Method)
	}

	if req.endSpan == nil {
		return c.internalErrorf("%#v produced a duplicate %q Response", from, req.Method)
	}

	if result != nil && err != nil {
		c.internalErrorf("%#v returned a non-nil result with a non-nil error for %s:\n%v\n%#v", from, req.Method, err, result)
		result = nil // Discard the spurious result and respond with err.
	}

	if req.IsCall() {
		if result == nil && err == nil {
			err = c.internalErrorf("%#v returned a nil result and nil error for a %q Request that requires a Response", from, req.Method)
		}

		response, respErr := NewResponse(req.ID, result, err)

		// The caller could theoretically reuse the request's ID as soon as we've
		// sent the response, so ensure that it is removed from the incoming map
		// before sending.
		c.updateInFlight(func(s *inFlightState) {
			delete(s.incomingByID, req.ID)
		})
		if respErr == nil {
			writeErr := c.write(notDone{req.ctx}, response)
			if err == nil {
				err = writeErr
			}
		} else {
			err = c.internalErrorf("%#v returned a malformed result for %q: %w", from, req.Method, respErr)
		}
	} else { // req is a notification
		if result != nil {
			err = c.internalErrorf("%#v returned a non-nil result for a %q Request without an ID", from, req.Method)
		} else if err != nil {
			err = fmt.Errorf("%w: %q notification failed: %v", ErrInternal, req.Method, err)
		}
		if err != nil {
			// TODO: can/should we do anything with this error beyond writing it to the event log?
			// (Is this the right label to attach to the log?)
			event.Label(req.ctx, keys.Err.Of(err))
		}
	}

	labelStatus(req.ctx, err)

	// Cancel the request and finalize the event span to free any associated resources.
	req.cancel()
	req.endSpan()
	req.endSpan = nil
	c.updateInFlight(func(s *inFlightState) {
		if s.incoming == 0 {
			panic("jsonrpc2_v2: processResult called when incoming count is already zero")
		}
		s.incoming--
	})
	return nil
}

// write is used by all things that write outgoing messages, including replies.
// it makes sure that writes are atomic
func (c *Connection) write(ctx context.Context, msg Message) error {
	writer := <-c.writer
	defer func() { c.writer <- writer }()
	n, err := writer.Write(ctx, msg)
	event.Metric(ctx, tag.SentBytes.Of(n))

	if err != nil && ctx.Err() == nil {
		// The call to Write failed, and since ctx.Err() is nil we can't attribute
		// the failure (even indirectly) to Context cancellation. The writer appears
		// to be broken, and future writes are likely to also fail.
		//
		// If the read side of the connection is also broken, we might not even be
		// able to receive cancellation notifications. Since we can't reliably write
		// the results of incoming calls and can't receive explicit cancellations,
		// cancel the calls now.
		c.updateInFlight(func(s *inFlightState) {
			if s.writeErr == nil {
				s.writeErr = err
				for _, r := range s.incomingByID {
					r.cancel()
				}
			}
		})
	}

	return err
}

// internalErrorf reports an internal error. By default it panics, but if
// c.onInternalError is non-nil it instead calls that and returns an error
// wrapping ErrInternal.
func (c *Connection) internalErrorf(format string, args ...interface{}) error {
	err := fmt.Errorf(format, args...)
	if c.onInternalError == nil {
		panic("jsonrpc2: " + err.Error())
	}
	c.onInternalError(err)

	return fmt.Errorf("%w: %v", ErrInternal, err)
}

// labelStatus labels the status of the event in ctx based on whether err is nil.
func labelStatus(ctx context.Context, err error) {
	if err == nil {
		event.Label(ctx, tag.StatusCode.Of("OK"))
	} else {
		event.Label(ctx, tag.StatusCode.Of("ERROR"))
	}
}

// notDone is a context.Context wrapper that returns a nil Done channel.
type notDone struct{ ctx context.Context }

func (ic notDone) Value(key interface{}) interface{} {
	return ic.ctx.Value(key)
}

func (notDone) Done() <-chan struct{}       { return nil }
func (notDone) Err() error                  { return nil }
func (notDone) Deadline() (time.Time, bool) { return time.Time{}, false }
