// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug066

type Scope struct {
	entries map[string] *Object;
}


type Type struct {
	scope *Scope;
}


type Object struct {
	typ *Type;
}


func Lookup(scope *Scope) *Object {
	return scope.entries["foo"];
}
