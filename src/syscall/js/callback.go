// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package js

import "sync"

var (
	pendingCallbacks        = Global().Get("Array").New()
	makeCallbackHelper      = Global().Get("Go").Get("_makeCallbackHelper")
	makeEventCallbackHelper = Global().Get("Go").Get("_makeEventCallbackHelper")
)

var (
	callbacksMu    sync.Mutex
	callbacks             = make(map[uint32]func([]Value))
	nextCallbackID uint32 = 1
)

// Callback is a Go function that got wrapped for use as a JavaScript callback.
type Callback struct {
	Value // the JavaScript function that queues the callback for execution
	id    uint32
}

// NewCallback returns a wrapped callback function.
//
// Invoking the callback in JavaScript will queue the Go function fn for execution.
// This execution happens asynchronously on a special goroutine that handles all callbacks and preserves
// the order in which the callbacks got called.
// As a consequence, if one callback blocks this goroutine, other callbacks will not be processed.
// A blocking callback should therefore explicitly start a new goroutine.
//
// Callback.Release must be called to free up resources when the callback will not be used any more.
func NewCallback(fn func(args []Value)) Callback {
	callbackLoopOnce.Do(func() {
		go callbackLoop()
	})

	callbacksMu.Lock()
	id := nextCallbackID
	nextCallbackID++
	callbacks[id] = fn
	callbacksMu.Unlock()
	return Callback{
		Value: makeCallbackHelper.Invoke(id, pendingCallbacks, jsGo),
		id:    id,
	}
}

type EventCallbackFlag int

const (
	// PreventDefault can be used with NewEventCallback to call event.preventDefault synchronously.
	PreventDefault EventCallbackFlag = 1 << iota
	// StopPropagation can be used with NewEventCallback to call event.stopPropagation synchronously.
	StopPropagation
	// StopImmediatePropagation can be used with NewEventCallback to call event.stopImmediatePropagation synchronously.
	StopImmediatePropagation
)

// NewEventCallback returns a wrapped callback function, just like NewCallback, but the callback expects to have
// exactly one argument, the event. Depending on flags, it will synchronously call event.preventDefault,
// event.stopPropagation and/or event.stopImmediatePropagation before queuing the Go function fn for execution.
func NewEventCallback(flags EventCallbackFlag, fn func(event Value)) Callback {
	c := NewCallback(func(args []Value) {
		fn(args[0])
	})
	return Callback{
		Value: makeEventCallbackHelper.Invoke(
			flags&PreventDefault != 0,
			flags&StopPropagation != 0,
			flags&StopImmediatePropagation != 0,
			c,
		),
		id: c.id,
	}
}

// Release frees up resources allocated for the callback.
// The callback must not be invoked after calling Release.
func (c Callback) Release() {
	callbacksMu.Lock()
	delete(callbacks, c.id)
	callbacksMu.Unlock()
}

var callbackLoopOnce sync.Once

func callbackLoop() {
	for !jsGo.Get("_callbackShutdown").Bool() {
		sleepUntilCallback()
		for {
			cb := pendingCallbacks.Call("shift")
			if cb == Undefined() {
				break
			}

			id := uint32(cb.Get("id").Int())
			callbacksMu.Lock()
			f, ok := callbacks[id]
			callbacksMu.Unlock()
			if !ok {
				Global().Get("console").Call("error", "call to closed callback")
				continue
			}

			argsObj := cb.Get("args")
			args := make([]Value, argsObj.Length())
			for i := range args {
				args[i] = argsObj.Index(i)
			}
			f(args)
		}
	}
}

// sleepUntilCallback is defined in the runtime package
func sleepUntilCallback()
