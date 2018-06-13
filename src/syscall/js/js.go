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

import (
	"unsafe"
)

// ref is used to identify a JavaScript value, since the value itself can not be passed to WebAssembly itself.
type ref uint32

// Value represents a JavaScript value.
type Value struct {
	ref ref
}

func makeValue(v ref) Value {
	return Value{ref: v}
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
	Undefined = makeValue(0)

	// Null is the JavaScript value "null".
	Null = makeValue(1)

	// Global is the JavaScript global object, usually "window" or "global".
	Global = makeValue(2)

	// memory is the WebAssembly linear memory.
	memory = makeValue(3)

	// resolveCallbackPromise is a function that the callback helper uses to resume the execution of Go's WebAssembly code.
	resolveCallbackPromise = makeValue(4)
)

var uint8Array = Global.Get("Uint8Array")

// ValueOf returns x as a JavaScript value.
func ValueOf(x interface{}) Value {
	switch x := x.(type) {
	case Value:
		return x
	case Callback:
		return x.enqueueFn
	case nil:
		return Null
	case bool:
		return makeValue(boolVal(x))
	case int:
		return makeValue(intVal(x))
	case int8:
		return makeValue(intVal(int(x)))
	case int16:
		return makeValue(intVal(int(x)))
	case int32:
		return makeValue(intVal(int(x)))
	case int64:
		return makeValue(intVal(int(x)))
	case uint:
		return makeValue(intVal(int(x)))
	case uint8:
		return makeValue(intVal(int(x)))
	case uint16:
		return makeValue(intVal(int(x)))
	case uint32:
		return makeValue(intVal(int(x)))
	case uint64:
		return makeValue(intVal(int(x)))
	case uintptr:
		return makeValue(intVal(int(x)))
	case unsafe.Pointer:
		return makeValue(intVal(int(uintptr(x))))
	case float32:
		return makeValue(floatVal(float64(x)))
	case float64:
		return makeValue(floatVal(x))
	case string:
		return makeValue(stringVal(x))
	case []byte:
		if len(x) == 0 {
			return uint8Array.New(memory.Get("buffer"), 0, 0)
		}
		return uint8Array.New(memory.Get("buffer"), unsafe.Pointer(&x[0]), len(x))
	default:
		panic("invalid value")
	}
}

func boolVal(x bool) ref

func intVal(x int) ref

func floatVal(x float64) ref

func stringVal(x string) ref

// Get returns the JavaScript property p of value v.
func (v Value) Get(p string) Value {
	return makeValue(valueGet(v.ref, p))
}

func valueGet(v ref, p string) ref

// Set sets the JavaScript property p of value v to x.
func (v Value) Set(p string, x interface{}) {
	valueSet(v.ref, p, ValueOf(x).ref)
}

func valueSet(v ref, p string, x ref)

// Index returns JavaScript index i of value v.
func (v Value) Index(i int) Value {
	return makeValue(valueIndex(v.ref, i))
}

func valueIndex(v ref, i int) ref

// SetIndex sets the JavaScript index i of value v to x.
func (v Value) SetIndex(i int, x interface{}) {
	valueSetIndex(v.ref, i, ValueOf(x).ref)
}

func valueSetIndex(v ref, i int, x ref)

func makeArgs(args []interface{}) []ref {
	argVals := make([]ref, len(args))
	for i, arg := range args {
		argVals[i] = ValueOf(arg).ref
	}
	return argVals
}

// Length returns the JavaScript property "length" of v.
func (v Value) Length() int {
	return valueLength(v.ref)
}

func valueLength(v ref) int

// Call does a JavaScript call to the method m of value v with the given arguments.
// It panics if v has no method m.
func (v Value) Call(m string, args ...interface{}) Value {
	res, ok := valueCall(v.ref, m, makeArgs(args))
	if !ok {
		panic(Error{makeValue(res)})
	}
	return makeValue(res)
}

func valueCall(v ref, m string, args []ref) (ref, bool)

// Invoke does a JavaScript call of the value v with the given arguments.
// It panics if v is not a function.
func (v Value) Invoke(args ...interface{}) Value {
	res, ok := valueInvoke(v.ref, makeArgs(args))
	if !ok {
		panic(Error{makeValue(res)})
	}
	return makeValue(res)
}

func valueInvoke(v ref, args []ref) (ref, bool)

// New uses JavaScript's "new" operator with value v as constructor and the given arguments.
// It panics if v is not a function.
func (v Value) New(args ...interface{}) Value {
	res, ok := valueNew(v.ref, makeArgs(args))
	if !ok {
		panic(Error{makeValue(res)})
	}
	return makeValue(res)
}

func valueNew(v ref, args []ref) (ref, bool)

// Float returns the value v converted to float64 according to JavaScript type conversions (parseFloat).
func (v Value) Float() float64 {
	return valueFloat(v.ref)
}

func valueFloat(v ref) float64

// Int returns the value v converted to int according to JavaScript type conversions (parseInt).
func (v Value) Int() int {
	return valueInt(v.ref)
}

func valueInt(v ref) int

// Bool returns the value v converted to bool according to JavaScript type conversions.
func (v Value) Bool() bool {
	return valueBool(v.ref)
}

func valueBool(v ref) bool

// String returns the value v converted to string according to JavaScript type conversions.
func (v Value) String() string {
	str, length := valuePrepareString(v.ref)
	b := make([]byte, length)
	valueLoadString(str, b)
	return string(b)
}

func valuePrepareString(v ref) (ref, int)

func valueLoadString(v ref, b []byte)
