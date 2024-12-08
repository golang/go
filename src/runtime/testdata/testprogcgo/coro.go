// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.rangefunc && !windows

package main

/*
#include <stdint.h> // for uintptr_t

void go_callback_coro(uintptr_t handle);

static void call_go(uintptr_t handle) {
	go_callback_coro(handle);
}
*/
import "C"

import (
	"fmt"
	"iter"
	"runtime/cgo"
)

func init() {
	register("CoroCgoIterCallback", func() {
		println("expect: OK")
		CoroCgo(callerExhaust, iterCallback)
	})
	register("CoroCgoIterCallbackYield", func() {
		println("expect: OS thread locking must match")
		CoroCgo(callerExhaust, iterCallbackYield)
	})
	register("CoroCgoCallback", func() {
		println("expect: OK")
		CoroCgo(callerExhaustCallback, iterSimple)
	})
	register("CoroCgoCallbackIterNested", func() {
		println("expect: OK")
		CoroCgo(callerExhaustCallback, iterNested)
	})
	register("CoroCgoCallbackIterCallback", func() {
		println("expect: OK")
		CoroCgo(callerExhaustCallback, iterCallback)
	})
	register("CoroCgoCallbackIterCallbackYield", func() {
		println("expect: OS thread locking must match")
		CoroCgo(callerExhaustCallback, iterCallbackYield)
	})
	register("CoroCgoCallbackAfterPull", func() {
		println("expect: OS thread locking must match")
		CoroCgo(callerCallbackAfterPull, iterSimple)
	})
	register("CoroCgoStopCallback", func() {
		println("expect: OK")
		CoroCgo(callerStopCallback, iterSimple)
	})
	register("CoroCgoStopCallbackIterNested", func() {
		println("expect: OK")
		CoroCgo(callerStopCallback, iterNested)
	})
}

var toCall func()

//export go_callback_coro
func go_callback_coro(handle C.uintptr_t) {
	h := cgo.Handle(handle)
	h.Value().(func())()
	h.Delete()
}

func callFromC(f func()) {
	C.call_go(C.uintptr_t(cgo.NewHandle(f)))
}

func CoroCgo(driver func(iter.Seq[int]) error, seq iter.Seq[int]) {
	if err := driver(seq); err != nil {
		println("error:", err.Error())
		return
	}
	println("OK")
}

func callerExhaust(i iter.Seq[int]) error {
	next, _ := iter.Pull(i)
	for {
		v, ok := next()
		if !ok {
			break
		}
		if v != 5 {
			return fmt.Errorf("bad iterator: wanted value %d, got %d", 5, v)
		}
	}
	return nil
}

func callerExhaustCallback(i iter.Seq[int]) (err error) {
	callFromC(func() {
		next, _ := iter.Pull(i)
		for {
			v, ok := next()
			if !ok {
				break
			}
			if v != 5 {
				err = fmt.Errorf("bad iterator: wanted value %d, got %d", 5, v)
			}
		}
	})
	return err
}

func callerStopCallback(i iter.Seq[int]) (err error) {
	callFromC(func() {
		next, stop := iter.Pull(i)
		v, _ := next()
		stop()
		if v != 5 {
			err = fmt.Errorf("bad iterator: wanted value %d, got %d", 5, v)
		}
	})
	return err
}

func callerCallbackAfterPull(i iter.Seq[int]) (err error) {
	next, _ := iter.Pull(i)
	callFromC(func() {
		for {
			v, ok := next()
			if !ok {
				break
			}
			if v != 5 {
				err = fmt.Errorf("bad iterator: wanted value %d, got %d", 5, v)
			}
		}
	})
	return err
}

func iterSimple(yield func(int) bool) {
	for range 3 {
		if !yield(5) {
			return
		}
	}
}

func iterNested(yield func(int) bool) {
	next, stop := iter.Pull(iterSimple)
	for {
		v, ok := next()
		if ok {
			if !yield(v) {
				stop()
			}
		} else {
			return
		}
	}
}

func iterCallback(yield func(int) bool) {
	for range 3 {
		callFromC(func() {})
		if !yield(5) {
			return
		}
	}
}

func iterCallbackYield(yield func(int) bool) {
	for range 3 {
		var ok bool
		callFromC(func() {
			ok = yield(5)
		})
		if !ok {
			return
		}
	}
}
