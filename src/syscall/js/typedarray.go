// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package js

import (
	"sync"
	"unsafe"
)

var (
	int8Array    = Global().Get("Int8Array")
	int16Array   = Global().Get("Int16Array")
	int32Array   = Global().Get("Int32Array")
	uint8Array   = Global().Get("Uint8Array")
	uint16Array  = Global().Get("Uint16Array")
	uint32Array  = Global().Get("Uint32Array")
	float32Array = Global().Get("Float32Array")
	float64Array = Global().Get("Float64Array")
)

var _ Wrapper = TypedArray{} // TypedArray must implement Wrapper

// TypedArray represents a JavaScript typed array.
type TypedArray struct {
	Value
}

// Release frees up resources allocated for the typed array.
// The typed array and its buffer must not be accessed after calling Release.
func (a TypedArray) Release() {
	openTypedArraysMutex.Lock()
	delete(openTypedArrays, a)
	openTypedArraysMutex.Unlock()
}

var (
	openTypedArraysMutex sync.Mutex
	openTypedArrays      = make(map[TypedArray]interface{})
)

// TypedArrayOf returns a JavaScript typed array backed by the slice's underlying array.
//
// The supported types are []int8, []int16, []int32, []uint8, []uint16, []uint32, []float32 and []float64.
// Passing an unsupported value causes a panic.
//
// TypedArray.Release must be called to free up resources when the typed array will not be used any more.
func TypedArrayOf(slice interface{}) TypedArray {
	a := TypedArray{typedArrayOf(slice)}
	openTypedArraysMutex.Lock()
	openTypedArrays[a] = slice
	openTypedArraysMutex.Unlock()
	return a
}

func typedArrayOf(slice interface{}) Value {
	switch slice := slice.(type) {
	case []int8:
		if len(slice) == 0 {
			return int8Array.New(memory.Get("buffer"), 0, 0)
		}
		return int8Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	case []int16:
		if len(slice) == 0 {
			return int16Array.New(memory.Get("buffer"), 0, 0)
		}
		return int16Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	case []int32:
		if len(slice) == 0 {
			return int32Array.New(memory.Get("buffer"), 0, 0)
		}
		return int32Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	case []uint8:
		if len(slice) == 0 {
			return uint8Array.New(memory.Get("buffer"), 0, 0)
		}
		return uint8Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	case []uint16:
		if len(slice) == 0 {
			return uint16Array.New(memory.Get("buffer"), 0, 0)
		}
		return uint16Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	case []uint32:
		if len(slice) == 0 {
			return uint32Array.New(memory.Get("buffer"), 0, 0)
		}
		return uint32Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	case []float32:
		if len(slice) == 0 {
			return float32Array.New(memory.Get("buffer"), 0, 0)
		}
		return float32Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	case []float64:
		if len(slice) == 0 {
			return float64Array.New(memory.Get("buffer"), 0, 0)
		}
		return float64Array.New(memory.Get("buffer"), unsafe.Pointer(&slice[0]), len(slice))
	default:
		panic("TypedArrayOf: not a supported slice")
	}
}
