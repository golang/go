// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the errorsas checker.

package a

import "errors"

type myError[T any] struct{ t T }

func (myError[T]) Error() string { return "" }

type twice[T any] struct {
	t T
}

func perr[T any]() *T { return nil }

func two[T any]() (error, *T) { return nil, nil }

func _[E error](e E) {
	var (
		m  myError[int]
		tw twice[myError[int]]
	)
	errors.As(nil, &e)
	errors.As(nil, &m)            // *T where T implemements error
	errors.As(nil, &tw.t)         // *T where T implements error
	errors.As(nil, perr[error]()) // *error, via a call

	errors.As(nil, e)    // want `second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type`
	errors.As(nil, m)    // want `second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type`
	errors.As(nil, tw.t) // want `second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type`
	errors.As(two[error]())
}
