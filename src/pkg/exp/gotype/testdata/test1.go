// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	// the scope of a local type declaration starts immediately after the type name
	type T struct{ _ *T }
}

func _(x interface{}) {
	// the variable defined by a TypeSwitchGuard is declared in each TypeCaseClause
	switch t := x.(type) {
	case int:
		_ = t
	case float32:
		_ = t
	default:
		_ = t
	}

	// the variable defined by a TypeSwitchGuard must not conflict with other
	// variables declared in the initial simple statement
	switch t := 0; t := x.(type) {
	}
}
