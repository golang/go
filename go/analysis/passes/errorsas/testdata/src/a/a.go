// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the errorsas checker.

package a

import "errors"

type myError int

func (myError) Error() string { return "" }

func perr() *error { return nil }

type iface interface {
	m()
}

func two() (error, interface{}) { return nil, nil }

func _() {
	var (
		e  error
		m  myError
		i  int
		f  iface
		ei interface{}
	)
	errors.As(nil, &e)     // *error
	errors.As(nil, &m)     // *T where T implemements error
	errors.As(nil, &f)     // *interface
	errors.As(nil, perr()) // *error, via a call
	errors.As(nil, ei)     //  empty interface

	errors.As(nil, nil) // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, e)   // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, m)   // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, f)   // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, &i)  // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(two())
}
