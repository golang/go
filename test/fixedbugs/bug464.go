// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3937: unhelpful typechecking loop message
// for identifiers wrongly used as types.

package main

func foo(x foo) {} // ERROR "expected type|not a type"
