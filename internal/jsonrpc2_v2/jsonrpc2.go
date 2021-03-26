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
	// ErrNotHandled is returned from a handler to indicate it did not handle the
	// message.
	ErrNotHandled = errors.New("JSON RPC not handled")
	// ErrAsyncResponse is returned from a handler to indicate it will generate a
	// response asynchronously.
	ErrAsyncResponse = errors.New("JSON RPC asynchronous response")
)

// Preempter handles messages on a connection before they are queued to the main
// handler.
// Primarily this is used for cancel handlers or notifications for which out of
// order processing is not an issue.
type Preempter interface {
	// Preempt is invoked for each incoming request before it is queued.
	// If the request is a call, it must return a value or an error for the reply.
	// Preempt should not block or start any new messages on the connection.
	Preempt(ctx context.Context, req *Request) (interface{}, error)
}

// Handler handles messages on a connection.
type Handler interface {
	// Handle is invoked for each incoming request.
	// If the request is a call, it must return a value or an error for the reply.
	Handle(ctx context.Context, req *Request) (interface{}, error)
}

type defaultHandler struct{}

func (defaultHandler) Preempt(context.Context, *Request) (interface{}, error) {
	return nil, ErrNotHandled
}

func (defaultHandler) Handle(context.Context, *Request) (interface{}, error) {
	return nil, ErrNotHandled
}

// async is a small helper for things with an asynchronous result that you can
// wait for.
type async struct {
	ready  chan struct{}
	errBox chan error
}

func (a *async) init() {
	a.ready = make(chan struct{})
	a.errBox = make(chan error, 1)
	a.errBox <- nil
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
	err := <-a.errBox
	a.errBox <- err
	return err
}

func (a *async) setError(err error) {
	storedErr := <-a.errBox
	if storedErr == nil {
		storedErr = err
	}
	a.errBox <- storedErr
}
