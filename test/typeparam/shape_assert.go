// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that type assertions and type switches in generic functions
// produce correct results when the compiler eliminates impossible
// cases based on shape type analysis.

package main

import "fmt"

func switchStringOrBytes[S string | []byte](x S) string {
	switch any(x).(type) {
	case string:
		return "string"
	case []byte:
		return "[]byte"
	}
	return "unknown"
}

func switchThree[S string | []byte | int](x S) string {
	switch any(x).(type) {
	case string:
		return "string"
	case []byte:
		return "[]byte"
	case int:
		return "int"
	}
	return "unknown"
}

type MyString string

func switchNamed[S string | MyString](x S) string {
	switch any(x).(type) {
	case string:
		return "string"
	case MyString:
		return "MyString"
	}
	return "unknown"
}

func commaOkString[S string | []byte](x S) (string, bool) {
	v, ok := any(x).(string)
	return v, ok
}

func commaOkBytes[S string | []byte](x S) ([]byte, bool) {
	v, ok := any(x).([]byte)
	return v, ok
}

func commaOkChain[S string | []byte | int](x S) string {
	if _, ok := any(x).(string); ok {
		return "string"
	}
	if _, ok := any(x).([]byte); ok {
		return "[]byte"
	}
	if _, ok := any(x).(int); ok {
		return "int"
	}
	return "unknown"
}

// Intermediate variable tests.
func commaOkViaVar[S string | []byte](x S) (string, bool) {
	iface := any(x)
	v, ok := iface.(string)
	return v, ok
}

func switchViaVar[S string | []byte](x S) string {
	iface := any(x)
	switch iface.(type) {
	case string:
		return "string"
	case []byte:
		return "[]byte"
	}
	return "unknown"
}

// When no switch case matches the shape, the default is taken.
func switchFallsToDefault[S string | []byte | int](x S) string {
	switch any(x).(type) {
	case string:
		return "string"
	case []byte:
		return "[]byte"
	}
	return "other"
}

func main() {
	check("switchStringOrBytes string", switchStringOrBytes("hello"), "string")
	check("switchStringOrBytes []byte", switchStringOrBytes([]byte("hello")), "[]byte")

	check("switchThree string", switchThree("x"), "string")
	check("switchThree []byte", switchThree([]byte("x")), "[]byte")
	check("switchThree int", switchThree(42), "int")

	check("switchNamed string", switchNamed("hi"), "string")
	check("switchNamed MyString", switchNamed(MyString("hi")), "MyString")

	v1, ok1 := commaOkString("hello")
	check("commaOkString[string] val", v1, "hello")
	checkBool("commaOkString[string] ok", ok1, true)
	v2, ok2 := commaOkString([]byte("hello"))
	check("commaOkString[[]byte] val", v2, "")
	checkBool("commaOkString[[]byte] ok", ok2, false)

	v3, ok3 := commaOkBytes([]byte("world"))
	check("commaOkBytes[[]byte] val", string(v3), "world")
	checkBool("commaOkBytes[[]byte] ok", ok3, true)
	v4, ok4 := commaOkBytes("world")
	check("commaOkBytes[string] val", string(v4), "")
	checkBool("commaOkBytes[string] ok", ok4, false)

	check("commaOkChain string", commaOkChain("x"), "string")
	check("commaOkChain []byte", commaOkChain([]byte("x")), "[]byte")
	check("commaOkChain int", commaOkChain(42), "int")

	// Intermediate variable: comma-ok
	v5, ok5 := commaOkViaVar("hello")
	check("commaOkViaVar[string] val", v5, "hello")
	checkBool("commaOkViaVar[string] ok", ok5, true)
	v6, ok6 := commaOkViaVar([]byte("hello"))
	check("commaOkViaVar[[]byte] val", v6, "")
	checkBool("commaOkViaVar[[]byte] ok", ok6, false)

	// Intermediate variable: type switch
	check("switchViaVar string", switchViaVar("x"), "string")
	check("switchViaVar []byte", switchViaVar([]byte("x")), "[]byte")

	// All cases impossible: int instantiation hits default
	check("switchFallsToDefault string", switchFallsToDefault("x"), "string")
	check("switchFallsToDefault []byte", switchFallsToDefault([]byte("x")), "[]byte")
	check("switchFallsToDefault int", switchFallsToDefault(42), "other")
}

func check(name, got, want string) {
	if got != want {
		panic(fmt.Sprintf("%s: got %q, want %q", name, got, want))
	}
}

func checkBool(name string, got, want bool) {
	if got != want {
		panic(fmt.Sprintf("%s: got %v, want %v", name, got, want))
	}
}
