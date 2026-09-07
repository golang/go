// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm32

package runtime_test

import (
	"internal/abi"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

type wasm32Adder interface {
	Add(int) int
}

type wasm32AddOne struct{}

//go:noinline
func (wasm32AddOne) Add(v int) int { return v + 1 }

//go:noinline
func wasm32GrowAndYield(depth int, fn func(int) int) int {
	var frame [256]byte
	frame[0] = byte(depth)
	if depth == 0 {
		runtime.Gosched()
		result := fn(40)
		runtime.KeepAlive(&frame)
		return result
	}
	result := wasm32GrowAndYield(depth-1, fn)
	runtime.KeepAlive(&frame)
	return result
}

//go:noinline
func wasm32RecoverAndContinue(fn func() int) (result int) {
	defer func() {
		if recover() == nil {
			result = -1
			return
		}
		runtime.Gosched()
		result = fn()
	}()
	panic("resume")
}

//go:noinline
func wasm32IndirectTarget(v int) int { return v + 2 }

func TestWasm32ContinuationState(t *testing.T) {
	var nilFn func()
	if got := reflect.ValueOf(nilFn).Pointer(); got != 0 {
		t.Fatalf("nil function PC = %#x, want 0", got)
	}
	if got := reflect.ValueOf(nilFn).UnsafePointer(); got != nil {
		t.Fatalf("nil function pointer = %p, want nil", got)
	}
	fn := wasm32IndirectTarget
	pc := abi.FuncPCABIInternal(fn)
	if got := reflect.ValueOf(fn).Pointer(); got != pc {
		t.Fatalf("reflect function PC = %#x, want %#x", got, pc)
	}
	f := runtime.FuncForPC(pc)
	if f == nil {
		t.Fatalf("FuncForPC(%#x) returned nil", pc)
	}
	if got := f.Name(); !strings.HasSuffix(got, ".wasm32IndirectTarget") {
		t.Fatalf("FuncForPC(%#x).Name() = %q, want wasm32IndirectTarget", pc, got)
	}

	done := make(chan int, 1)
	go func() {
		var adder wasm32Adder = wasm32AddOne{}
		result := wasm32GrowAndYield(256, adder.Add)
		result += wasm32RecoverAndContinue(func() int { return 1 })
		done <- result
	}()

	if got := <-done; got != 42 {
		t.Fatalf("continuation result = %d, want 42", got)
	}
}
