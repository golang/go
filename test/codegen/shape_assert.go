// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that type assertions and type switch cases that are impossible
// based on shape type analysis are eliminated from generated code.

package codegen

// -- Type switch elimination --

func switchStringOrBytes[S string | []byte](x S) string {
	switch any(x).(type) {
	case string:
		return "string"
	case []byte:
		return "[]byte"
	}
	return ""
}

// In the string instantiation, the []byte case is impossible
// and should be eliminated.
func SwitchStringInst(x string) string {
	// amd64:-"type:.*uint8"
	return switchStringOrBytes(x)
}

// In the []byte instantiation, the string case is impossible
// and should be eliminated.
func SwitchBytesInst(x []byte) string {
	// amd64:-"type:string"
	return switchStringOrBytes(x)
}

// -- Comma-ok type assertion elimination --

func commaOkString[S string | []byte](x S) (string, bool) {
	v, ok := any(x).(string)
	return v, ok
}

// In the []byte instantiation, .(string) always fails.
// The type comparison against type:string should be eliminated.
func CommaOkStringBytesInst(x []byte) (string, bool) {
	// amd64:-"type:string"
	return commaOkString(x)
}

// In the string instantiation, the comparison against type:string
// is also eliminated because the assertion always succeeds.
func CommaOkStringStringInst(x string) (string, bool) {
	// amd64:-"LEAQ\ttype:string"
	return commaOkString(x)
}

// -- Intermediate variable: comma-ok --

func commaOkViaVar[S string | []byte](x S) (string, bool) {
	iface := any(x)
	v, ok := iface.(string)
	return v, ok
}

func CommaOkViaVarBytesInst(x []byte) (string, bool) {
	// amd64:-"type:string"
	return commaOkViaVar(x)
}

// -- Intermediate variable: type switch --

func switchViaVar[S string | []byte](x S) string {
	iface := any(x)
	switch iface.(type) {
	case string:
		return "string"
	case []byte:
		return "[]byte"
	}
	return ""
}

func SwitchViaVarStringInst(x string) string {
	// amd64:-"type:.*uint8"
	return switchViaVar(x)
}

func SwitchViaVarBytesInst(x []byte) string {
	// amd64:-"type:string"
	return switchViaVar(x)
}

// -- All cases eliminated for one instantiation --

func switchFallsToDefault[S string | []byte | int](x S) string {
	switch any(x).(type) {
	case string:
		return "string"
	case []byte:
		return "[]byte"
	}
	return "other"
}

// int instantiation: both cases are impossible.
func SwitchFallsToDefaultIntInst(x int) string {
	// amd64:-"type:string"
	// amd64:-"type:.*uint8"
	return switchFallsToDefault(x)
}
