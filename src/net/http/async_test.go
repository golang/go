// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"errors"
	"internal/synctest"
)

var errStillRunning = errors.New("async op still running")

type asyncResult[T any] struct {
	donec chan struct{}
	res   T
	err   error
}

// runAsync runs f in a new goroutine.
// It returns an asyncResult which acts as a future.
//
// Must be called from within a synctest bubble.
func runAsync[T any](f func() (T, error)) *asyncResult[T] {
	r := &asyncResult[T]{
		donec: make(chan struct{}),
	}
	go func() {
		defer close(r.donec)
		r.res, r.err = f()
	}()
	synctest.Wait()
	return r
}

// done reports whether the function has returned.
func (r *asyncResult[T]) done() bool {
	_, err := r.result()
	return err != errStillRunning
}

// result returns the result of the function.
// If the function hasn't completed yet, it returns errStillRunning.
func (r *asyncResult[T]) result() (T, error) {
	select {
	case <-r.donec:
		return r.res, r.err
	default:
		var zero T
		return zero, errStillRunning
	}
}
