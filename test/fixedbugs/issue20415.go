// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure redeclaration errors report correct position.

package p

// 1
var f byte

var f interface{} // ERROR "issue20415.go:12: previous declaration|redefinition|f redeclared"

func _(f int) {
}

// 2
var g byte

func _(g int) {
}

var g interface{} // ERROR "issue20415.go:20: previous declaration|redefinition|g redeclared"

// 3
func _(h int) {
}

var h byte

var h interface{} // ERROR "issue20415.go:31: previous declaration|redefinition|h redeclared"
