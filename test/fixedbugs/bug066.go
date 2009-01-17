// $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

export type (
	Type struct;
	Object struct;
)

export type Scope struct {
	entries map[string] *Object;
}


export type Type struct {
	scope *Scope;
}


export type Object struct {
	typ *Type;
}


export func Lookup(scope *Scope) *Object {
	return scope.entries["foo"];
}
