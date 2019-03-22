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

func _() {
	var (
		e error
		m myError
		i int
		f iface
	)
	errors.As(nil, &e)
	errors.As(nil, &m)
	errors.As(nil, &f)
	errors.As(nil, perr())

	errors.As(nil, nil) // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, e)   // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, m)   // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, f)   // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
	errors.As(nil, &i)  // want `second argument to errors.As must be a pointer to an interface or a type implementing error`
}
