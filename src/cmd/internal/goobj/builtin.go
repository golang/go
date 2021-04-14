// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj

// Builtin (compiler-generated) function references appear
// frequently. We assign special indices for them, so they
// don't need to be referenced by name.

// NBuiltin returns the number of listed builtin
// symbols.
func NBuiltin() int {
	return len(builtins)
}

// BuiltinName returns the name and ABI of the i-th
// builtin symbol.
func BuiltinName(i int) (string, int) {
	return builtins[i].name, builtins[i].abi
}

// BuiltinIdx returns the index of the builtin with the
// given name and abi, or -1 if it is not a builtin.
func BuiltinIdx(name string, abi int) int {
	i, ok := builtinMap[name]
	if !ok {
		return -1
	}
	if builtins[i].abi != abi {
		return -1
	}
	return i
}

//go:generate go run mkbuiltin.go

var builtinMap map[string]int

func init() {
	builtinMap = make(map[string]int, len(builtins))
	for i, b := range builtins {
		builtinMap[b.name] = i
	}
}
