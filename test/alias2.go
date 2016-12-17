// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test basic restrictions on type aliases.

// The compiler doesn't implement type aliases yet,
// so for now we get the same error (unimplemented)
// everywhere, OR-ed into the ERROR checks.
// TODO(gri) remove the need for "unimplemented"

package p

import (
	"reflect"
	. "reflect"
)

// Valid type alias declarations.

type _ = int           // ERROR "unimplemented"
type _ = struct{}      // ERROR "unimplemented"
type _ = reflect.Value // ERROR "unimplemented"
type _ = Value         // ERROR "unimplemented"

type (
	a1 = int           // ERROR "unimplemented"
	a2 = struct{}      // ERROR "unimplemented"
	a3 = reflect.Value // ERROR "unimplemented"
	a4 = Value         // ERROR "unimplemented"
)

func _() {
	type _ = int           // ERROR "unimplemented"
	type _ = struct{}      // ERROR "unimplemented"
	type _ = reflect.Value // ERROR "unimplemented"
	type _ = Value         // ERROR "unimplemented"

	type (
		a1 = int           // ERROR "unimplemented"
		a2 = struct{}      // ERROR "unimplemented"
		a3 = reflect.Value // ERROR "unimplemented"
		a4 = Value         // ERROR "unimplemented"
	)
}

// Invalid type alias declarations.

type _ = reflect.ValueOf // ERROR "reflect.ValueOf is not a type|unimplemented"

type b1 = struct{} // ERROR "unimplemented"
func (b1) m()      {} // disabled ERROR "invalid receiver type"

// TODO(gri) expand
// It appears that type-checking exits after some more severe errors, so we may
// need more test files.
