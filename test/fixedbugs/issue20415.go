// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure redeclaration errors report correct position.

package p

// 1
var f byte

var f interface{} // ERROR "previous declaration at issue20415.go:12|redefinition"

func _(f int) {
}

// 2
var g byte

func _(g int) {
}

var g interface{} // ERROR "previous declaration at issue20415.go:20|redefinition"

// 3
func _(h int) {
}

var h byte

var h interface{} // ERROR "previous declaration at issue20415.go:31|redefinition"
