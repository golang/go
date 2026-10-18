// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http/httptrace"
	"net/url"
	"sync"
)

// A ClientConn is a client connection to an HTTP server.
//
// Unlike a [Transport], a ClientConn represents a single connection.
// Most users should use a Transport rather than creating client connections directly.
type ClientConn struct {
	cc genericClientConn

	stateHookMu      sync.Mutex
	userStateHook    func(*ClientConn)
	stateHookRunning bool
	lastAvailable    int
	lastInFlight     int
	lastClosed       bool
}

// newClientConner is the interface implemented by HTTP/2 transports to create new client conns.
//
// The http package (this package) needs a way to ask the http2 package to
// create a client connection.
//
// Transport.TLSNextProto["h2"] contains a function which appears to do this,
// but for historical reasons it does not: The TLSNextProto function adds a
// *tls.Conn to the http2.Transport's connection pool and returns a RoundTripper
// which is backed by that connection pool. NewClientConn needs a way to get a
// single client connection out of the http2 package.
//
// The http2 package registers a RoundTripper with Transport.RegisterProtocol.
// If this RoundTripper implements newClientConner, then Transport.NewClientConn will use
// it to create new HTTP/2 client connections.
type newClientConner interface {
	// NewClientConn creates a new client connection from a net.Conn.
	//
	// The RoundTripper returned by NewClientConn must implement externalClientConn.
	// (We don't define NewClientConn as returning externalClientConn,
	// because either we'd need to make externalClientConn an exported type
	// or define it as a type alias. Neither is particularly appealing.)
	//
	// The state hook passed here is the internal state hook
	// (ClientConn.maybeRunStateHook). The internal state hook calls
	// the user state hook (if any), which is set by the user with
	// ClientConn.SetStateHook.
	//
	// The client connection should arrange to call the internal state hook
	// when the connection closes, when requests complete, and when the
	// connection concurrency limit changes.
	//
	// The client connection must call the internal state hook when the connection state
	// changes asynchronously, such as when a request completes.
	//
	// The internal state hook need not be called after synchronous changes to the state:
	// Close, Reserve, Release, and RoundTrip calls which don't start a request
	// do not need to call the hook.
	//
	// The general idea is that if we call (for example) Close,
	// we know that the connection state has probably changed and we
	// don't need the state hook to tell us that.
	// However, if the connection closes asynchronously
	// (because, for example, the other end of the conn closed it),
	// the state hook needs to inform us.
	NewClientConn(nc net.Conn, internalStateHook func()) (RoundTripper, error)
}

// externalClientConn is an interface implemented by HTTP/2 and HTTP/3 client conns
// returned from newClientConner.NewClientConn and dialClientConner.DialClientConn.
//
// See the newClientConner doc comment for more information.
type externalClientConn interface {
	Close() error
	Err() error
	RoundTrip(req *Request) (*Response, error)
	Reserve() error
	Release()
	Available() int
	InFlight() int
}

// genericClientConn is an internal interface representing any client connection
// (HTTP/1, HTTP/2, HTTP/3, or custom RoundTripper).
type genericClientConn interface {
	externalClientConn

	// IsAlt reports whether this "connection" is backed by a RoundTripper returned
	// from TLSNextProto or a similar source. Historically, this means the "connection"
	// is backed by an http2.Transport, with its own connection pool.
	IsAlt() bool

	// IsReused reports whether the connection has successfully made at least one roundtrip.
	IsReused() bool

	// NetConn is the net.Conn to use in httptrace.GotConnInfo.
	NetConn() net.Conn

	// CloseWithError closes the connection, with an error given as the reason.
	CloseWithError(err error)
}

// NewClientConn creates a new client connection to the given address.
//
// If scheme is "http", the connection is unencrypted.
// If scheme is "https", the connection uses TLS.
//
// The protocol used for the new connection is determined by the scheme,
// Transport.Protocols configuration field, and protocols supported by the
// server. See Transport.Protocols for more details.
//
// If Transport.Proxy is set and indicates that a request sent to the given
// address should use a proxy, the new connection uses that proxy.
//
// NewClientConn always creates a new connection,
// even if the Transport has an existing cached connection to the given host.
//
// The new connection is not added to the Transport's connection cache,
// and will not be used by [Transport.RoundTrip].
// It does not count against the MaxIdleConns and MaxConnsPerHost limits.
//
// The caller is responsible for closing the new connection.
func (t *Transport) NewClientConn(ctx context.Context, scheme, address string) (*ClientConn, error) {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)

	if t.h2Config != nil {
		// Handle x/net/http2.Transport.NewClientConn passing us a net.Conn
		// to create a ClientConn from.
		if cc, err := t.http2NewClientConnFromContext(ctx); err != errors.ErrUnsupported {
			return cc, err
		}
	}

	switch scheme {
	case "http", "https":
	default:
		return nil, fmt.Errorf("net/http: invalid scheme %q", scheme)
	}

	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}
	if port == "" {
		port = schemePort(scheme)
	}

	var proxyURL *url.URL
	if t.Proxy != nil {
		// Transport.Proxy takes a *Request, so create a fake one to pass it.
		req := &Request{
			ctx:    ctx,
			Method: "GET",
			URL: &url.URL{
				Scheme: scheme,
				Host:   host,
				Path:   "/",
			},
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header:     make(Header),
			Body:       NoBody,
			Host:       host,
		}
		var err error
		proxyURL, err = t.Proxy(req)
		if err != nil {
			return nil, err
		}
	}

	cm := connectMethod{
		targetScheme: scheme,
		targetAddr:   net.JoinHostPort(host, port),
		proxyURL:     proxyURL,
	}

	// The state hook is a bit tricky:
	// The persistConn has a state hook which calls ClientConn.maybeRunStateHook,
	// which in turn calls the user-provided state hook (if any).
	//
	// ClientConn.maybeRunStateHook handles debouncing hook calls for both
	// HTTP/1 and HTTP/2.
	//
	// Since there's no need to change the persistConn's hook, we set it at creation time.
	cc := &ClientConn{}
	const isClientConn = true
	gc, err := t.dialConn(ctx, cm, isClientConn, nil, cc.maybeRunStateHook)
	if err != nil {
		return nil, err
	}

	// Note that cc.maybeRunStateHook may have been called
	// in the short window between dialConn and now.
	// This is fine.
	cc.stateHookMu.Lock()
	defer cc.stateHookMu.Unlock()
	cc.cc = gc
	cc.lastAvailable = gc.Available()
	return cc, nil
}

// Close closes the connection.
// Outstanding RoundTrip calls are interrupted.
func (cc *ClientConn) Close() error {
	defer cc.maybeRunStateHook()
	return cc.cc.Close()
}

// Err reports any fatal connection errors.
// It returns nil if the connection is usable.
// If it returns non-nil, the connection can no longer be used.
func (cc *ClientConn) Err() error {
	return cc.cc.Err()
}

func validateClientConnRequest(req *Request) error {
	if req.URL == nil {
		return errors.New("http: nil Request.URL")
	}
	if req.Header == nil {
		return errors.New("http: nil Request.Header")
	}
	// Validate the outgoing headers.
	if err := validateHeaders(req.Header); err != "" {
		return fmt.Errorf("http: invalid header %s", err)
	}
	// Validate the outgoing trailers too.
	if err := validateHeaders(req.Trailer); err != "" {
		return fmt.Errorf("http: invalid trailer %s", err)
	}
	if req.Method != "" && !validMethod(req.Method) {
		return fmt.Errorf("http: invalid method %q", req.Method)
	}
	if req.URL.Host == "" {
		return errors.New("http: no Host in request URL")
	}
	return nil
}

// RoundTrip implements the [RoundTripper] interface.
//
// The request is sent on the client connection,
// regardless of the URL being requested or any proxy settings.
//
// If the connection is at its concurrency limit,
// RoundTrip waits for the connection to become available
// before sending the request.
func (cc *ClientConn) RoundTrip(req *Request) (*Response, error) {
	defer cc.maybeRunStateHook()
	if req.URL == nil && req.Method == ":ping" {
		// Undocumented feature for sending a PING frame to a HTTP/2 connection,
		// included to support x/net/http2.ClientConn.Ping.
		pinger, ok := cc.cc.(interface {
			Ping(context.Context) error
		})
		if !ok {
			return nil, errors.New("http: ClientConn does not support PING")
		}
		if err := pinger.Ping(req.Context()); err != nil {
			return nil, err
		}
		return &Response{
			StatusCode: 200,
			Body:       NoBody,
		}, nil
	}
	if err := validateClientConnRequest(req); err != nil {
		cc.Release()
		return nil, err
	}
	return cc.cc.RoundTrip(req)
}

// Available reports the number of requests that may be sent
// to the connection without blocking.
// It returns 0 if the connection is closed.
func (cc *ClientConn) Available() int {
	return cc.cc.Available()
}

// InFlight reports the number of requests in flight,
// including reserved requests.
// It returns 0 if the connection is closed.
func (cc *ClientConn) InFlight() int {
	return cc.cc.InFlight()
}

// Reserve reserves a concurrency slot on the connection.
// If Reserve returns nil, one additional RoundTrip call may be made
// without waiting for an existing request to complete.
//
// The reserved concurrency slot is accounted as an in-flight request.
// A successful call to RoundTrip will decrement the Available count
// and increment the InFlight count.
//
// Each successful call to Reserve should be followed by exactly one call
// to RoundTrip or Release, which will consume or release the reservation.
//
// If the connection is closed or at its concurrency limit,
// Reserve returns an error.
func (cc *ClientConn) Reserve() error {
	defer cc.maybeRunStateHook()
	return cc.cc.Reserve()
}

// Release releases an unused concurrency slot reserved by Reserve.
// If there are no reserved concurrency slots, it has no effect.
func (cc *ClientConn) Release() {
	defer cc.maybeRunStateHook()
	cc.cc.Release()
}

// shouldRunStateHook returns the user's state hook if we should call it,
// or nil if we don't need to call it at this time.
func (cc *ClientConn) shouldRunStateHook(stopRunning bool) func(*ClientConn) {
	cc.stateHookMu.Lock()
	defer cc.stateHookMu.Unlock()
	if cc.cc == nil {
		return nil
	}
	if stopRunning {
		cc.stateHookRunning = false
	}
	if cc.userStateHook == nil {
		return nil
	}
	if cc.stateHookRunning {
		return nil
	}
	var (
		available = cc.Available()
		inFlight  = cc.InFlight()
		closed    = cc.Err() != nil
	)
	var hook func(*ClientConn)
	if available > cc.lastAvailable || inFlight < cc.lastInFlight || closed != cc.lastClosed {
		hook = cc.userStateHook
		cc.stateHookRunning = true
	}
	cc.lastAvailable = available
	cc.lastInFlight = inFlight
	cc.lastClosed = closed
	return hook
}

func (cc *ClientConn) maybeRunStateHook() {
	hook := cc.shouldRunStateHook(false)
	if hook == nil {
		return
	}
	// Run the hook synchronously.
	//
	// This means that if, for example, the user calls resp.Body.Close to finish a request,
	// the Close call will synchronously run the hook, giving the hook the chance to
	// return the ClientConn to a connection pool before the next request is made.
	hook(cc)
	// The connection state may have changed while the hook was running,
	// in which case we need to run it again.
	//
	// If we do need to run the hook again, do so in a new goroutine to avoid blocking
	// the current goroutine indefinitely.
	hook = cc.shouldRunStateHook(true)
	if hook != nil {
		go func() {
			for hook != nil {
				hook(cc)
				hook = cc.shouldRunStateHook(true)
			}
		}()
	}
}

// SetStateHook arranges for f to be called when the state of the connection changes.
// At most one call to f is made at a time.
// If the connection's state has changed since it was created,
// f is called immediately in a separate goroutine.
// f may be called synchronously from RoundTrip or Response.Body.Close.
//
// If SetStateHook is called multiple times, the new hook replaces the old one.
// If f is nil, no further calls will be made to f after SetStateHook returns.
//
// f is called when Available increases (more requests may be sent on the connection),
// InFlight decreases (existing requests complete), or Err begins returning non-nil
// (the connection is no longer usable).
func (cc *ClientConn) SetStateHook(f func(*ClientConn)) {
	cc.stateHookMu.Lock()
	cc.userStateHook = f
	cc.stateHookMu.Unlock()
	cc.maybeRunStateHook()
}

// The following makes persistConn implement genericClientConn for HTTP/1 connections.
// TODO: Move to somewhere more associated with persistConn; these functions are here
// because they predate persistConn implementing genericClientConn.

func (pc *persistConn) RoundTrip(req *Request) (*Response, error) {
	ctx := req.Context()
	trace := httptrace.ContextClientTrace(ctx)

	// Convert Request.Cancel into context cancellation.
	ctx, cancel := context.WithCancelCause(req.Context())
	if req.Cancel != nil {
		go awaitLegacyCancel(ctx, cancel, req)
	}

	treq := &transportRequest{Request: req, trace: trace, ctx: ctx, cancel: cancel}
	resp, err := pc.roundTrip(treq)
	if err != nil {
		cancel(err)
		return nil, err
	}
	resp.Request = req
	return resp, nil
}

func (pc *persistConn) Close() error {
	pc.CloseWithError(errors.New("ClientConn closed"))
	return nil
}

func (pc *persistConn) Err() error {
	select {
	case <-pc.closech:
		return pc.closed
	default:
		return nil
	}
}

func (pc *persistConn) Available() int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed != nil || pc.reserved || pc.inFlight {
		return 0
	}
	return 1
}

func (pc *persistConn) InFlight() int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed == nil && (pc.reserved || pc.inFlight) {
		return 1
	}
	return 0
}

func (pc *persistConn) Reserve() error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed != nil {
		return pc.closed
	}
	select {
	case <-pc.availch:
	default:
		return errors.New("connection is unavailable")
	}
	pc.reserved = true
	return nil
}

func (pc *persistConn) Release() {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.reserved {
		select {
		case pc.availch <- struct{}{}:
		default:
			panic("cannot release reservation")
		}
		pc.reserved = false
	}
}

func (pc *persistConn) IsAlt() bool       { return false }
func (pc *persistConn) NetConn() net.Conn { return pc.conn }

// externalConn wraps an externalClientConn to implement genericClientConn.
type externalConn struct {
	externalClientConn
}

func (cc externalConn) IsAlt() bool          { return false }
func (cc externalConn) NetConn() net.Conn    { return nil }
func (cc externalConn) IsReused() bool       { return false }
func (cc externalConn) CloseWithError(error) { cc.Close() }

func (cc externalConn) Ping(ctx context.Context) error {
	pinger, ok := cc.externalClientConn.(interface {
		Ping(context.Context) error
	})
	if !ok {
		return errors.New("http: ClientConn does not support PING")
	}
	return pinger.Ping(ctx)
}

// altClientConn implements the genericClientConn interface for a RoundTripper
// (such as from TLSNextProto or http2RoundTripper).
//
// An altClientConn is, historically, backed by an http2 Transport which manages its
// own connection pool, and is not really a "connection".
type altClientConn struct {
	rt     RoundTripper
	mu     sync.Mutex
	closed error
}

func (cc *altClientConn) IsAlt() bool          { return true }
func (cc *altClientConn) NetConn() net.Conn    { return nil }
func (cc *altClientConn) IsReused() bool       { return false }
func (cc *altClientConn) CloseWithError(error) { cc.Close() }
func (cc *altClientConn) Reserve() error       { return nil }
func (cc *altClientConn) Release()             {}
func (cc *altClientConn) Available() int       { return 1 }
func (cc *altClientConn) InFlight() int        { return 0 }

func (cc *altClientConn) RoundTrip(req *Request) (*Response, error) {
	return cc.rt.RoundTrip(req)
}

func (cc *altClientConn) Ping(ctx context.Context) error {
	pinger, ok := cc.rt.(interface {
		Ping(context.Context) error
	})
	if !ok {
		return errors.New("http: ClientConn does not support PING")
	}
	return pinger.Ping(ctx)
}

func (cc *altClientConn) Close() error {
	cc.mu.Lock()
	if cc.closed != nil {
		cc.mu.Unlock()
		return nil
	}
	cc.closed = net.ErrClosed
	cc.mu.Unlock()
	if c, ok := cc.rt.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

func (cc *altClientConn) Err() error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.closed
}
