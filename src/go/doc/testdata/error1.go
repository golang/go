// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package error1

type I0 interface {
	// When embedded, the predeclared error interface
	// must remain visible in interface types.
	error
}

type T0 struct {
	ExportedField interface {
		// error should be visible
		error
	}
}

type S0 struct {
	// In struct types, an embedded error must only be visible
	// if AllDecls is set.
	error
}
