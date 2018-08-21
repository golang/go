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

// ref is used to identify a JavaScript value, since the value itself can not be passed to WebAssembly.
// A JavaScript number (64-bit float, except NaN) is represented by its IEEE 754 binary representation.
// All other values are represented as an IEEE 754 binary representation of NaN with bits 0-31 used as
// an ID and bits 32-33 used to differentiate between string, symbol, function and object.
type ref uint64

// nanHead are the upper 32 bits of a ref which are set if the value is not a JavaScript number or NaN itself.
const nanHead = 0x7FF80000

// Value represents a JavaScript value.
type Value struct {
	ref ref
}

func makeValue(v ref) Value {
	return Value{ref: v}
}

func predefValue(id uint32) Value {
	return Value{ref: nanHead<<32 | ref(id)}
}

func floatValue(f float64) Value {
	if f != f {
		return valueNaN
	}
	return Value{ref: *(*ref)(unsafe.Pointer(&f))}
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
	valueNaN       = predefValue(0)
	valueUndefined = predefValue(1)
	valueNull      = predefValue(2)
	valueTrue      = predefValue(3)
	valueFalse     = predefValue(4)
	valueGlobal    = predefValue(5)
	memory         = predefValue(6) // WebAssembly linear memory
	jsGo           = predefValue(7) // instance of the Go class in JavaScript

	objectConstructor = valueGlobal.Get("Object")
	arrayConstructor  = valueGlobal.Get("Array")
)

// Undefined returns the JavaScript value "undefined".
func Undefined() Value {
	return valueUndefined
}

// Null returns the JavaScript value "null".
func Null() Value {
	return valueNull
}

// Global returns the JavaScript global object, usually "window" or "global".
func Global() Value {
	return valueGlobal
}

// ValueOf returns x as a JavaScript value:
//
//  | Go                     | JavaScript             |
//  | ---------------------- | ---------------------- |
//  | js.Value               | [its value]            |
//  | js.TypedArray          | typed array            |
//  | js.Callback            | function               |
//  | nil                    | null                   |
//  | bool                   | boolean                |
//  | integers and floats    | number                 |
//  | string                 | string                 |
//  | []interface{}          | new array              |
//  | map[string]interface{} | new object             |
func ValueOf(x interface{}) Value {
	switch x := x.(type) {
	case Value:
		return x
	case TypedArray:
		return x.Value
	case Callback:
		return x.Value
	case nil:
		return valueNull
	case bool:
		if x {
			return valueTrue
		} else {
			return valueFalse
		}
	case int:
		return floatValue(float64(x))
	case int8:
		return floatValue(float64(x))
	case int16:
		return floatValue(float64(x))
	case int32:
		return floatValue(float64(x))
	case int64:
		return floatValue(float64(x))
	case uint:
		return floatValue(float64(x))
	case uint8:
		return floatValue(float64(x))
	case uint16:
		return floatValue(float64(x))
	case uint32:
		return floatValue(float64(x))
	case uint64:
		return floatValue(float64(x))
	case uintptr:
		return floatValue(float64(x))
	case unsafe.Pointer:
		return floatValue(float64(uintptr(x)))
	case float32:
		return floatValue(float64(x))
	case float64:
		return floatValue(x)
	case string:
		return makeValue(stringVal(x))
	case []interface{}:
		a := arrayConstructor.New(len(x))
		for i, s := range x {
			a.SetIndex(i, s)
		}
		return a
	case map[string]interface{}:
		o := objectConstructor.New()
		for k, v := range x {
			o.Set(k, v)
		}
		return o
	default:
		panic("ValueOf: invalid value")
	}
}

func stringVal(x string) ref

// Type represents the JavaScript type of a Value.
type Type int

const (
	TypeUndefined Type = iota
	TypeNull
	TypeBoolean
	TypeNumber
	TypeString
	TypeSymbol
	TypeObject
	TypeFunction
)

func (t Type) String() string {
	switch t {
	case TypeUndefined:
		return "undefined"
	case TypeNull:
		return "null"
	case TypeBoolean:
		return "boolean"
	case TypeNumber:
		return "number"
	case TypeString:
		return "string"
	case TypeSymbol:
		return "symbol"
	case TypeObject:
		return "object"
	case TypeFunction:
		return "function"
	default:
		panic("bad type")
	}
}

// Type returns the JavaScript type of the value v. It is similar to JavaScript's typeof operator,
// except that it returns TypeNull instead of TypeObject for null.
func (v Value) Type() Type {
	switch v.ref {
	case valueUndefined.ref:
		return TypeUndefined
	case valueNull.ref:
		return TypeNull
	case valueTrue.ref, valueFalse.ref:
		return TypeBoolean
	}
	if v.isNumber() {
		return TypeNumber
	}
	typeFlag := v.ref >> 32 & 3
	switch typeFlag {
	case 1:
		return TypeString
	case 2:
		return TypeSymbol
	case 3:
		return TypeFunction
	default:
		return TypeObject
	}
}

// Get returns the JavaScript property p of value v.
func (v Value) Get(p string) Value {
	return makeValue(valueGet(v.ref, p))
}

func valueGet(v ref, p string) ref

// Set sets the JavaScript property p of value v to ValueOf(x).
func (v Value) Set(p string, x interface{}) {
	valueSet(v.ref, p, ValueOf(x).ref)
}

func valueSet(v ref, p string, x ref)

// Index returns JavaScript index i of value v.
func (v Value) Index(i int) Value {
	return makeValue(valueIndex(v.ref, i))
}

func valueIndex(v ref, i int) ref

// SetIndex sets the JavaScript index i of value v to ValueOf(x).
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
// The arguments get mapped to JavaScript values according to the ValueOf function.
func (v Value) Call(m string, args ...interface{}) Value {
	res, ok := valueCall(v.ref, m, makeArgs(args))
	if !ok {
		if vType := v.Type(); vType != TypeObject && vType != TypeFunction { // check here to avoid overhead in success case
			panic(&ValueError{"Value.Call", vType})
		}
		if propType := v.Get(m).Type(); propType != TypeFunction {
			panic("syscall/js: Value.Call: property " + m + " is not a function, got " + propType.String())
		}
		panic(Error{makeValue(res)})
	}
	return makeValue(res)
}

func valueCall(v ref, m string, args []ref) (ref, bool)

// Invoke does a JavaScript call of the value v with the given arguments.
// It panics if v is not a function.
// The arguments get mapped to JavaScript values according to the ValueOf function.
func (v Value) Invoke(args ...interface{}) Value {
	res, ok := valueInvoke(v.ref, makeArgs(args))
	if !ok {
		if vType := v.Type(); vType != TypeFunction { // check here to avoid overhead in success case
			panic(&ValueError{"Value.Invoke", vType})
		}
		panic(Error{makeValue(res)})
	}
	return makeValue(res)
}

func valueInvoke(v ref, args []ref) (ref, bool)

// New uses JavaScript's "new" operator with value v as constructor and the given arguments.
// It panics if v is not a function.
// The arguments get mapped to JavaScript values according to the ValueOf function.
func (v Value) New(args ...interface{}) Value {
	res, ok := valueNew(v.ref, makeArgs(args))
	if !ok {
		panic(Error{makeValue(res)})
	}
	return makeValue(res)
}

func valueNew(v ref, args []ref) (ref, bool)

func (v Value) isNumber() bool {
	return v.ref>>32&nanHead != nanHead || v.ref == valueNaN.ref
}

func (v Value) float(method string) float64 {
	if !v.isNumber() {
		panic(&ValueError{method, v.Type()})
	}
	return *(*float64)(unsafe.Pointer(&v.ref))
}

// Float returns the value v as a float64. It panics if v is not a JavaScript number.
func (v Value) Float() float64 {
	return v.float("Value.Float")
}

// Int returns the value v truncated to an int. It panics if v is not a JavaScript number.
func (v Value) Int() int {
	return int(v.float("Value.Int"))
}

// Bool returns the value v as a bool. It panics if v is not a JavaScript boolean.
func (v Value) Bool() bool {
	switch v.ref {
	case valueTrue.ref:
		return true
	case valueFalse.ref:
		return false
	default:
		panic(&ValueError{"Value.Bool", v.Type()})
	}
}

// String returns the value v converted to string according to JavaScript type conversions.
func (v Value) String() string {
	str, length := valuePrepareString(v.ref)
	b := make([]byte, length)
	valueLoadString(str, b)
	return string(b)
}

func valuePrepareString(v ref) (ref, int)

func valueLoadString(v ref, b []byte)

// InstanceOf reports whether v is an instance of type t according to JavaScript's instanceof operator.
func (v Value) InstanceOf(t Value) bool {
	return valueInstanceOf(v.ref, t.ref)
}

func valueInstanceOf(v ref, t ref) bool

// A ValueError occurs when a Value method is invoked on
// a Value that does not support it. Such cases are documented
// in the description of each method.
type ValueError struct {
	Method string
	Type   Type
}

func (e *ValueError) Error() string {
	return "syscall/js: call of " + e.Method + " on " + e.Type.String()
}
