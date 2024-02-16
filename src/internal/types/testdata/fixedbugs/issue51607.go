// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Interface types must be ignored during overlap test.

type (
	T1 interface{int}
	T2 interface{~int}
	T3 interface{T1 | bool | string}
	T4 interface{T2 | ~bool | ~string}
)

type (
	// overlap errors for non-interface terms
	// (like the interface terms, but explicitly inlined)
	_ interface{int | int /* ERROR "overlapping terms int and int" */ }
	_ interface{int | ~ /* ERROR "overlapping terms ~int and int" */ int}
	_ interface{~int | int /* ERROR "overlapping terms int and ~int" */ }
	_ interface{~int | ~ /* ERROR "overlapping terms ~int and ~int" */ int}

	_ interface{T1 | bool | string | T1 | bool /* ERROR "overlapping terms bool and bool" */ | string /* ERROR "overlapping terms string and string" */ }
	_ interface{T1 | bool | string | T2 | ~ /* ERROR "overlapping terms ~bool and bool" */ bool | ~ /* ERROR "overlapping terms ~string and string" */ string}

	// no errors for interface terms
	_ interface{T1 | T1}
	_ interface{T1 | T2}
	_ interface{T2 | T1}
	_ interface{T2 | T2}

	_ interface{T3 | T3 | int}
	_ interface{T3 | T4 | bool }
	_ interface{T4 | T3 | string }
	_ interface{T4 | T4 | float64 }
)

func _[_ T1 | bool | string | T1 | bool /* ERROR "overlapping terms" */ ]() {}
func _[_ T1 | bool | string | T2 | ~ /* ERROR "overlapping terms" */ bool ]() {}
func _[_ T2 | ~bool | ~string | T1 | bool /* ERROR "overlapping terms" */ ]() {}
func _[_ T2 | ~bool | ~string | T2 | ~ /* ERROR "overlapping terms" */ bool ]() {}

func _[_ T3 | T3 | int]() {}
func _[_ T3 | T4 | bool]() {}
func _[_ T4 | T3 | string]() {}
func _[_ T4 | T4 | float64]() {}

// test cases from issue

type _ interface {
	interface {bool | int} | interface {bool | string}
}

type _ interface {
	interface {bool | int} ; interface {bool | string}
}

type _ interface {
	interface {bool; int} ; interface {bool; string}
}

type _ interface {
	interface {bool; int} | interface {bool; string}
}