// errorcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Self recursion.
type i interface{ m() interface{ i } } // ERROR "invalid recursive type"
type _ interface{ i }                  // no redundant error

// Mutual recursion.
type j interface{ m() interface{ k } } // ERROR "invalid recursive type"
type k interface{ m() interface{ j } }

// Both self and mutual recursion.
type (
	a interface { // ERROR "invalid recursive type"
		m() interface {
			a
			b
		}
	}
	b interface {
		m() interface {
			a
			b
		}
	}
)

// Self recursion through other types.
func _() { type i interface{ m() *interface{ i } } }        // ERROR "invalid recursive type"
func _() { type i interface{ m() []interface{ i } } }       // ERROR "invalid recursive type"
func _() { type i interface{ m() [0]interface{ i } } }      // ERROR "invalid recursive type"
func _() { type i interface{ m() chan interface{ i } } }    // ERROR "invalid recursive type"
func _() { type i interface{ m() map[interface{ i }]int } } // ERROR "invalid recursive type"
func _() { type i interface{ m() map[int]interface{ i } } } // ERROR "invalid recursive type"
func _() { type i interface{ m() func(interface{ i }) } }   // ERROR "invalid recursive type"
func _() { type i interface{ m() func() interface{ i } } }  // ERROR "invalid recursive type"
func _() {
	type i interface { // ERROR "invalid recursive type"
		m() struct{ i interface{ i } }
	}
}
