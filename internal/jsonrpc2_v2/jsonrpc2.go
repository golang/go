// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jsonrpc2 is a minimal implementation of the JSON RPC 2 spec.
// https://www.jsonrpc.org/specification
// It is intended to be compatible with other implementations at the wire level.
package jsonrpc2

import (
	"context"
	"errors"
)

var (
	// ErrIdleTimeout is returned when serving timed out waiting for new connections.
	ErrIdleTimeout = errors.New("timed out waiting for new connections")

	// ErrNotHandled is returned from a Handler or Preempter to indicate it did
	// not handle the request.
	//
	// If a Handler returns ErrNotHandled, the server replies with
	// ErrMethodNotFound.
	ErrNotHandled = errors.New("JSON RPC not handled")

	// ErrAsyncResponse is returned from a handler to indicate it will generate a
	// response asynchronously.
	//
	// ErrAsyncResponse must not be returned for notifications,
	// which do not receive responses.
	ErrAsyncResponse = errors.New("JSON RPC asynchronous response")
)

// Preempter handles messages on a connection before they are queued to the main
// handler.
// Primarily this is used for cancel handlers or notifications for which out of
// order processing is not an issue.
type Preempter interface {
	// Preempt is invoked for each incoming request before it is queued for handling.
	//
	// If Preempt returns ErrNotHandled, the request will be queued,
	// and eventually passed to a Handle call.
	//
	// Otherwise, the result and error are processed as if returned by Handle.
	//
	// Preempt must not block. (The Context passed to it is for Values only.)
	Preempt(ctx context.Context, req *Request) (result interface{}, err error)
}

// A PreempterFunc implements the Preempter interface for a standalone Preempt function.
type PreempterFunc func(ctx context.Context, req *Request) (interface{}, error)

func (f PreempterFunc) Preempt(ctx context.Context, req *Request) (interface{}, error) {
	return f(ctx, req)
}

var _ Preempter = PreempterFunc(nil)

// Handler handles messages on a connection.
type Handler interface {
	// Handle is invoked sequentially for each incoming request that has not
	// already been handled by a Preempter.
	//
	// If the Request has a nil ID, Handle must return a nil result,
	// and any error may be logged but will not be reported to the caller.
	//
	// If the Request has a non-nil ID, Handle must return either a
	// non-nil, JSON-marshalable result, or a non-nil error.
	//
	// The Context passed to Handle will be canceled if the
	// connection is broken or the request is canceled or completed.
	// (If Handle returns ErrAsyncResponse, ctx will remain uncanceled
	// until either Cancel or Respond is called for the request's ID.)
	Handle(ctx context.Context, req *Request) (result interface{}, err error)
}

type defaultHandler struct{}

func (defaultHandler) Preempt(context.Context, *Request) (interface{}, error) {
	return nil, ErrNotHandled
}

func (defaultHandler) Handle(context.Context, *Request) (interface{}, error) {
	return nil, ErrNotHandled
}

// A HandlerFunc implements the Handler interface for a standalone Handle function.
type HandlerFunc func(ctx context.Context, req *Request) (interface{}, error)

func (f HandlerFunc) Handle(ctx context.Context, req *Request) (interface{}, error) {
	return f(ctx, req)
}

var _ Handler = HandlerFunc(nil)

// async is a small helper for operations with an asynchronous result that you
// can wait for.
type async struct {
	ready    chan struct{} // closed when done
	firstErr chan error    // 1-buffered; contains either nil or the first non-nil error
}

func newAsync() *async {
	var a async
	a.ready = make(chan struct{})
	a.firstErr = make(chan error, 1)
	a.firstErr <- nil
	return &a
}

func (a *async) done() {
	close(a.ready)
}

func (a *async) isDone() bool {
	select {
	case <-a.ready:
		return true
	default:
		return false
	}
}

func (a *async) wait() error {
	<-a.ready
	err := <-a.firstErr
	a.firstErr <- err
	return err
}

func (a *async) setError(err error) {
	storedErr := <-a.firstErr
	if storedErr == nil {
		storedErr = err
	}
	a.firstErr <- storedErr
}
