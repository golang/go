// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package error2

type I0 interface {
	// When embedded, the locally-declared error interface
	// is only visible if all declarations are shown.
	error
}

type T0 struct {
	ExportedField interface {
		// error should not be visible
		error
	}
}

type S0 struct {
	// In struct types, an embedded error must only be visible
	// if AllDecls is set.
	error
}

// This error declaration shadows the predeclared error type.
type error interface {
	Error() string
}
