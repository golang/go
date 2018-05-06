// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

// Package js gives access to the WebAssembly host environment when using the js/wasm architecture.
// Its API is based on JavaScript semantics.
//
// This package is EXPERIMENTAL. Its current scope is only to allow tests to run, but not yet to provide a
// comprehensive API for users. It is exempt from the Go compatibility promise.
package js

import "unsafe"

// Value represents a JavaScript value.
type Value struct {
	ref uint32
}

// Error wraps a JavaScript error.
type Error struct {
	// Value is the underlying JavaScript error value.
	Value
}

// Error implements the error interface.
func (e Error) Error() string {
	return "JavaScript error: " + e.Get("message").String()
}

var (
	// Undefined is the JavaScript value "undefined". The zero Value equals to Undefined.
	Undefined = Value{0}

	// Null is the JavaScript value "null".
	Null = Value{1}

	// Global is the JavaScript global object, usually "window" or "global".
	Global = Value{2}

	memory = Value{3}
)

var uint8Array = Global.Get("Uint8Array")

// ValueOf returns x as a JavaScript value.
func ValueOf(x interface{}) Value {
	switch x := x.(type) {
	case Value:
		return x
	case nil:
		return Null
	case bool:
		return boolVal(x)
	case int:
		return intVal(x)
	case int8:
		return intVal(int(x))
	case int16:
		return intVal(int(x))
	case int32:
		return intVal(int(x))
	case int64:
		return intVal(int(x))
	case uint:
		return intVal(int(x))
	case uint8:
		return intVal(int(x))
	case uint16:
		return intVal(int(x))
	case uint32:
		return intVal(int(x))
	case uint64:
		return intVal(int(x))
	case uintptr:
		return intVal(int(x))
	case unsafe.Pointer:
		return intVal(int(uintptr(x)))
	case float32:
		return floatVal(float64(x))
	case float64:
		return floatVal(x)
	case string:
		return stringVal(x)
	case []byte:
		if len(x) == 0 {
			return uint8Array.New(memory.Get("buffer"), 0, 0)
		}
		return uint8Array.New(memory.Get("buffer"), unsafe.Pointer(&x[0]), len(x))
	default:
		panic("invalid value")
	}
}

func boolVal(x bool) Value

func intVal(x int) Value

func floatVal(x float64) Value

func stringVal(x string) Value

// Get returns the JavaScript property p of value v.
func (v Value) Get(p string) Value

// Set sets the JavaScript property p of value v to x.
func (v Value) Set(p string, x interface{}) {
	v.set(p, ValueOf(x))
}

func (v Value) set(p string, x Value)

// Index returns JavaScript index i of value v.
func (v Value) Index(i int) Value

// SetIndex sets the JavaScript index i of value v to x.
func (v Value) SetIndex(i int, x interface{}) {
	v.setIndex(i, ValueOf(x))
}

func (v Value) setIndex(i int, x Value)

func makeArgs(args []interface{}) []Value {
	argVals := make([]Value, len(args))
	for i, arg := range args {
		argVals[i] = ValueOf(arg)
	}
	return argVals
}

// Length returns the JavaScript property "length" of v.
func (v Value) Length() int

// Call does a JavaScript call to the method m of value v with the given arguments.
// It panics if v has no method m.
func (v Value) Call(m string, args ...interface{}) Value {
	res, ok := v.call(m, makeArgs(args))
	if !ok {
		panic(Error{res})
	}
	return res
}

func (v Value) call(m string, args []Value) (Value, bool)

// Invoke does a JavaScript call of the value v with the given arguments.
// It panics if v is not a function.
func (v Value) Invoke(args ...interface{}) Value {
	res, ok := v.invoke(makeArgs(args))
	if !ok {
		panic(Error{res})
	}
	return res
}

func (v Value) invoke(args []Value) (Value, bool)

// New uses JavaScript's "new" operator with value v as constructor and the given arguments.
// It panics if v is not a function.
func (v Value) New(args ...interface{}) Value {
	res, ok := v.new(makeArgs(args))
	if !ok {
		panic(Error{res})
	}
	return res
}

func (v Value) new(args []Value) (Value, bool)

// Float returns the value v converted to float64 according to JavaScript type conversions (parseFloat).
func (v Value) Float() float64

// Int returns the value v converted to int according to JavaScript type conversions (parseInt).
func (v Value) Int() int

// Bool returns the value v converted to bool according to JavaScript type conversions.
func (v Value) Bool() bool

// String returns the value v converted to string according to JavaScript type conversions.
func (v Value) String() string {
	str, length := v.prepareString()
	b := make([]byte, length)
	str.loadString(b)
	return string(b)
}

func (v Value) prepareString() (Value, int)

func (v Value) loadString(b []byte)
