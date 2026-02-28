// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Compiler rejected initialization of structs to composite literals
// in a non-static setting (e.g. in a function)
// when the struct contained a field named _.

package p

type T struct {
	_ string
}

func ok() {
	var x = T{"check"}
	_ = x
	_ = T{"et"}
}

var (
	y = T{"stare"}
	w = T{_: "look"} // ERROR "invalid field name _ in struct initializer|expected struct field name|unknown field _ in struct literal of type T"
	_ = T{"page"}
	_ = T{_: "out"} // ERROR "invalid field name _ in struct initializer|expected struct field name|unknown field _ in struct literal of type T"
)

func bad() {
	var z = T{_: "verse"} // ERROR "invalid field name _ in struct initializer|expected struct field name|unknown field _ in struct literal of type T"
	_ = z
	_ = T{_: "itinerary"} // ERROR "invalid field name _ in struct initializer|expected struct field name|unknown field _ in struct literal of type T"
}
