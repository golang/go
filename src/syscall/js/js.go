// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

// Package js gives access to the WebAssembly host environment when using the js/wasm architecture.
// Its API is based on JavaScript semantics.
//
// This package is EXPERIMENTAL. Its current scope is only to allow tests to run, but not yet to provide a
// comprehensive API for users. It is exempt from the Go compatibility promise.
package js

import (
	"runtime"
	"unsafe"
)

// ref is used to identify a JavaScript value, since the value itself can not be passed to WebAssembly.
//
// The JavaScript value "undefined" is represented by the value 0.
// A JavaScript number (64-bit float, except 0 and NaN) is represented by its IEEE 754 binary representation.
// All other values are represented as an IEEE 754 binary representation of NaN with bits 0-31 used as
// an ID and bits 32-34 used to differentiate between string, symbol, function and object.
type ref uint64

// nanHead are the upper 32 bits of a ref which are set if the value is not encoded as an IEEE 754 number (see above).
const nanHead = 0x7FF80000

// Value represents a JavaScript value. The zero value is the JavaScript value "undefined".
// Values can be checked for equality with the Equal method.
type Value struct {
	_     [0]func() // uncomparable; to make == not compile
	ref   ref       // identifies a JavaScript value, see ref type
	gcPtr *ref      // used to trigger the finalizer when the Value is not referenced any more
}

const (
	// the type flags need to be in sync with wasm_exec.js
	typeFlagNone = iota
	typeFlagObject
	typeFlagString
	typeFlagSymbol
	typeFlagFunction
)

func makeValue(r ref) Value {
	var gcPtr *ref
	typeFlag := (r >> 32) & 7
	if (r>>32)&nanHead == nanHead && typeFlag != typeFlagNone {
		gcPtr = new(ref)
		*gcPtr = r
		runtime.SetFinalizer(gcPtr, func(p *ref) {
			finalizeRef(*p)
		})
	}

	return Value{ref: r, gcPtr: gcPtr}
}

func finalizeRef(r ref)

func predefValue(id uint32, typeFlag byte) Value {
	return Value{ref: (nanHead|ref(typeFlag))<<32 | ref(id)}
}

func floatValue(f float64) Value {
	if f == 0 {
		return valueZero
	}
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
	valueUndefined = Value{ref: 0}
	valueNaN       = predefValue(0, typeFlagNone)
	valueZero      = predefValue(1, typeFlagNone)
	valueNull      = predefValue(2, typeFlagNone)
	valueTrue      = predefValue(3, typeFlagNone)
	valueFalse     = predefValue(4, typeFlagNone)
	valueGlobal    = predefValue(5, typeFlagObject)
	jsGo           = predefValue(6, typeFlagObject) // instance of the Go class in JavaScript

	objectConstructor = valueGlobal.Get("Object")
	arrayConstructor  = valueGlobal.Get("Array")
)

// Equal reports whether v and w are equal according to JavaScript's === operator.
func (v Value) Equal(w Value) bool {
	return v.ref == w.ref && v.ref != valueNaN.ref
}

// Undefined returns the JavaScript value "undefined".
func Undefined() Value {
	return valueUndefined
}

// IsUndefined reports whether v is the JavaScript value "undefined".
func (v Value) IsUndefined() bool {
	return v.ref == valueUndefined.ref
}

// Null returns the JavaScript value "null".
func Null() Value {
	return valueNull
}

// IsNull reports whether v is the JavaScript value "null".
func (v Value) IsNull() bool {
	return v.ref == valueNull.ref
}

// IsNaN reports whether v is the JavaScript value "NaN".
func (v Value) IsNaN() bool {
	return v.ref == valueNaN.ref
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
//  | js.Func                | function               |
//  | nil                    | null                   |
//  | bool                   | boolean                |
//  | integers and floats    | number                 |
//  | string                 | string                 |
//  | []interface{}          | new array              |
//  | map[string]interface{} | new object             |
//
// Panics if x is not one of the expected types.
func ValueOf(x any) Value {
	switch x := x.(type) {
	case Value:
		return x
	case Func:
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
	case []any:
		a := arrayConstructor.New(len(x))
		for i, s := range x {
			a.SetIndex(i, s)
		}
		return a
	case map[string]any:
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

func (t Type) isObject() bool {
	return t == TypeObject || t == TypeFunction
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
	typeFlag := (v.ref >> 32) & 7
	switch typeFlag {
	case typeFlagObject:
		return TypeObject
	case typeFlagString:
		return TypeString
	case typeFlagSymbol:
		return TypeSymbol
	case typeFlagFunction:
		return TypeFunction
	default:
		panic("bad type flag")
	}
}

// Get returns the JavaScript property p of value v.
// It panics if v is not a JavaScript object.
func (v Value) Get(p string) Value {
	if vType := v.Type(); !vType.isObject() {
		panic(&ValueError{"Value.Get", vType})
	}
	r := makeValue(valueGet(v.ref, p))
	runtime.KeepAlive(v)
	return r
}

func valueGet(v ref, p string) ref

// Set sets the JavaScript property p of value v to ValueOf(x).
// It panics if v is not a JavaScript object.
func (v Value) Set(p string, x any) {
	if vType := v.Type(); !vType.isObject() {
		panic(&ValueError{"Value.Set", vType})
	}
	xv := ValueOf(x)
	valueSet(v.ref, p, xv.ref)
	runtime.KeepAlive(v)
	runtime.KeepAlive(xv)
}

func valueSet(v ref, p string, x ref)

// Delete deletes the JavaScript property p of value v.
// It panics if v is not a JavaScript object.
func (v Value) Delete(p string) {
	if vType := v.Type(); !vType.isObject() {
		panic(&ValueError{"Value.Delete", vType})
	}
	valueDelete(v.ref, p)
	runtime.KeepAlive(v)
}

func valueDelete(v ref, p string)

// Index returns JavaScript index i of value v.
// It panics if v is not a JavaScript object.
func (v Value) Index(i int) Value {
	if vType := v.Type(); !vType.isObject() {
		panic(&ValueError{"Value.Index", vType})
	}
	r := makeValue(valueIndex(v.ref, i))
	runtime.KeepAlive(v)
	return r
}

func valueIndex(v ref, i int) ref

// SetIndex sets the JavaScript index i of value v to ValueOf(x).
// It panics if v is not a JavaScript object.
func (v Value) SetIndex(i int, x any) {
	if vType := v.Type(); !vType.isObject() {
		panic(&ValueError{"Value.SetIndex", vType})
	}
	xv := ValueOf(x)
	valueSetIndex(v.ref, i, xv.ref)
	runtime.KeepAlive(v)
	runtime.KeepAlive(xv)
}

func valueSetIndex(v ref, i int, x ref)

func makeArgs(args []any) ([]Value, []ref) {
	argVals := make([]Value, len(args))
	argRefs := make([]ref, len(args))
	for i, arg := range args {
		v := ValueOf(arg)
		argVals[i] = v
		argRefs[i] = v.ref
	}
	return argVals, argRefs
}

// Length returns the JavaScript property "length" of v.
// It panics if v is not a JavaScript object.
func (v Value) Length() int {
	if vType := v.Type(); !vType.isObject() {
		panic(&ValueError{"Value.SetIndex", vType})
	}
	r := valueLength(v.ref)
	runtime.KeepAlive(v)
	return r
}

func valueLength(v ref) int

// Call does a JavaScript call to the method m of value v with the given arguments.
// It panics if v has no method m.
// The arguments get mapped to JavaScript values according to the ValueOf function.
func (v Value) Call(m string, args ...any) Value {
	argVals, argRefs := makeArgs(args)
	res, ok := valueCall(v.ref, m, argRefs)
	runtime.KeepAlive(v)
	runtime.KeepAlive(argVals)
	if !ok {
		if vType := v.Type(); !vType.isObject() { // check here to avoid overhead in success case
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
// It panics if v is not a JavaScript function.
// The arguments get mapped to JavaScript values according to the ValueOf function.
func (v Value) Invoke(args ...any) Value {
	argVals, argRefs := makeArgs(args)
	res, ok := valueInvoke(v.ref, argRefs)
	runtime.KeepAlive(v)
	runtime.KeepAlive(argVals)
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
// It panics if v is not a JavaScript function.
// The arguments get mapped to JavaScript values according to the ValueOf function.
func (v Value) New(args ...any) Value {
	argVals, argRefs := makeArgs(args)
	res, ok := valueNew(v.ref, argRefs)
	runtime.KeepAlive(v)
	runtime.KeepAlive(argVals)
	if !ok {
		if vType := v.Type(); vType != TypeFunction { // check here to avoid overhead in success case
			panic(&ValueError{"Value.Invoke", vType})
		}
		panic(Error{makeValue(res)})
	}
	return makeValue(res)
}

func valueNew(v ref, args []ref) (ref, bool)

func (v Value) isNumber() bool {
	return v.ref == valueZero.ref ||
		v.ref == valueNaN.ref ||
		(v.ref != valueUndefined.ref && (v.ref>>32)&nanHead != nanHead)
}

func (v Value) float(method string) float64 {
	if !v.isNumber() {
		panic(&ValueError{method, v.Type()})
	}
	if v.ref == valueZero.ref {
		return 0
	}
	return *(*float64)(unsafe.Pointer(&v.ref))
}

// Float returns the value v as a float64.
// It panics if v is not a JavaScript number.
func (v Value) Float() float64 {
	return v.float("Value.Float")
}

// Int returns the value v truncated to an int.
// It panics if v is not a JavaScript number.
func (v Value) Int() int {
	return int(v.float("Value.Int"))
}

// Bool returns the value v as a bool.
// It panics if v is not a JavaScript boolean.
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

// Truthy returns the JavaScript "truthiness" of the value v. In JavaScript,
// false, 0, "", null, undefined, and NaN are "falsy", and everything else is
// "truthy". See https://developer.mozilla.org/en-US/docs/Glossary/Truthy.
func (v Value) Truthy() bool {
	switch v.Type() {
	case TypeUndefined, TypeNull:
		return false
	case TypeBoolean:
		return v.Bool()
	case TypeNumber:
		return v.ref != valueNaN.ref && v.ref != valueZero.ref
	case TypeString:
		return v.String() != ""
	case TypeSymbol, TypeFunction, TypeObject:
		return true
	default:
		panic("bad type")
	}
}

// String returns the value v as a string.
// String is a special case because of Go's String method convention. Unlike the other getters,
// it does not panic if v's Type is not TypeString. Instead, it returns a string of the form "<T>"
// or "<T: V>" where T is v's type and V is a string representation of v's value.
func (v Value) String() string {
	switch v.Type() {
	case TypeString:
		return jsString(v)
	case TypeUndefined:
		return "<undefined>"
	case TypeNull:
		return "<null>"
	case TypeBoolean:
		return "<boolean: " + jsString(v) + ">"
	case TypeNumber:
		return "<number: " + jsString(v) + ">"
	case TypeSymbol:
		return "<symbol>"
	case TypeObject:
		return "<object>"
	case TypeFunction:
		return "<function>"
	default:
		panic("bad type")
	}
}

func jsString(v Value) string {
	str, length := valuePrepareString(v.ref)
	runtime.KeepAlive(v)
	b := make([]byte, length)
	valueLoadString(str, b)
	finalizeRef(str)
	return string(b)
}

func valuePrepareString(v ref) (ref, int)

func valueLoadString(v ref, b []byte)

// InstanceOf reports whether v is an instance of type t according to JavaScript's instanceof operator.
func (v Value) InstanceOf(t Value) bool {
	r := valueInstanceOf(v.ref, t.ref)
	runtime.KeepAlive(v)
	runtime.KeepAlive(t)
	return r
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

// CopyBytesToGo copies bytes from src to dst.
// It panics if src is not an Uint8Array or Uint8ClampedArray.
// It returns the number of bytes copied, which will be the minimum of the lengths of src and dst.
func CopyBytesToGo(dst []byte, src Value) int {
	n, ok := copyBytesToGo(dst, src.ref)
	runtime.KeepAlive(src)
	if !ok {
		panic("syscall/js: CopyBytesToGo: expected src to be an Uint8Array or Uint8ClampedArray")
	}
	return n
}

func copyBytesToGo(dst []byte, src ref) (int, bool)

// CopyBytesToJS copies bytes from src to dst.
// It panics if dst is not an Uint8Array or Uint8ClampedArray.
// It returns the number of bytes copied, which will be the minimum of the lengths of src and dst.
func CopyBytesToJS(dst Value, src []byte) int {
	n, ok := copyBytesToJS(dst.ref, src)
	runtime.KeepAlive(dst)
	if !ok {
		panic("syscall/js: CopyBytesToJS: expected dst to be an Uint8Array or Uint8ClampedArray")
	}
	return n
}

func copyBytesToJS(dst ref, src []byte) (int, bool)
