// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rt is the loGo/s support package: helpers the compiler's
// lowering calls, so lowered files import only logos/rt (plus unsafe
// where a milestone needs it).
package rt

import (
	"context"
	"errors"
	"reflect"
)

// Site records one error-propagation site: the file and line where a ?
// error path recorded the error. See TraceErr and ErrorTrace.
type Site struct {
	File string
	Line int
}

// Cold marks a branch unlikely. The backend lowers it to BranchUnlikely;
// it emits no code and costs nothing at run time.
func Cold() {}

// Join combines two errors, matching errors.Join semantics. The lowering
// keeps a nil result alone and calls Join only for two non-nil errors.
func Join(a, b error) error {
	return errors.Join(a, b)
}

// Context aliases context.Context so lowered code spells the group
// context without importing context itself.
type Context = context.Context

// UnionAssertionError is the panic value of a failed single-value
// assertion to a closed union: x.(U) panics with
// &UnionAssertionError{Value: x, Union: "<pkg>.U"} when the dynamic type
// of x is not a member of U.
type UnionAssertionError struct {
	Value any
	Union string
}

// Error formats the Go assertion message with the union in place of the
// asserted type: `interface conversion: <T> is not a member of <pkg>.U`,
// where <T> is the dynamic type of Value, or `nil` when Value is nil.
func (e *UnionAssertionError) Error() string {
	actual := "nil"
	if e.Value != nil {
		actual = reflect.TypeOf(e.Value).String()
	}
	return "interface conversion: " + actual + " is not a member of " + e.Union
}
