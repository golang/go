// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package js

import "sync"

var (
	funcsMu    sync.Mutex
	funcs             = make(map[uint32]func(Value, []Value) interface{})
	nextFuncID uint32 = 1
)

// Func is a wrapped Go function to be called by JavaScript.
type Func struct {
	Value // the JavaScript function that invokes the Go function
	id    uint32
}

// FuncOf returns a function to be used by JavaScript.
//
// The Go function fn is called with the value of JavaScript's "this" keyword and the
// arguments of the invocation. The return value of the invocation is
// the result of the Go function mapped back to JavaScript according to ValueOf.
//
// Invoking the wrapped Go function from JavaScript will
// pause the event loop and spawn a new goroutine.
// Other wrapped functions which are triggered during a call from Go to JavaScript
// get executed on the same goroutine.
//
// As a consequence, if one wrapped function blocks, JavaScript's event loop
// is blocked until that function returns. Hence, calling any async JavaScript
// API, which requires the event loop, like fetch (http.Client), will cause an
// immediate deadlock. Therefore a blocking function should explicitly start a
// new goroutine.
//
// Func.Release must be called to free up resources when the function will not be invoked any more.
func FuncOf(fn func(this Value, args []Value) interface{}) Func {
	funcsMu.Lock()
	id := nextFuncID
	nextFuncID++
	funcs[id] = fn
	funcsMu.Unlock()
	return Func{
		id:    id,
		Value: jsGo.Call("_makeFuncWrapper", id),
	}
}

// Release frees up resources allocated for the function.
// The function must not be invoked after calling Release.
// It is allowed to call Release while the function is still running.
func (c Func) Release() {
	funcsMu.Lock()
	delete(funcs, c.id)
	funcsMu.Unlock()
}

// setEventHandler is defined in the runtime package.
func setEventHandler(fn func())

func init() {
	setEventHandler(handleEvent)
}

func handleEvent() {
	cb := jsGo.Get("_pendingEvent")
	if cb.IsNull() {
		return
	}
	jsGo.Set("_pendingEvent", Null())

	id := uint32(cb.Get("id").Int())
	if id == 0 { // zero indicates deadlock
		select {}
	}
	funcsMu.Lock()
	f, ok := funcs[id]
	funcsMu.Unlock()
	if !ok {
		Global().Get("console").Call("error", "call to released function")
		return
	}

	this := cb.Get("this")
	argsObj := cb.Get("args")
	args := make([]Value, argsObj.Length())
	for i := range args {
		args[i] = argsObj.Index(i)
	}
	result := f(this, args)
	cb.Set("result", result)
}
