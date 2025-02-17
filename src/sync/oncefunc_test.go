// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"bytes"
	"math"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"testing"
	_ "unsafe"
)

// We assume that the Once.Do tests have already covered parallelism.

func TestOnceFunc(t *testing.T) {
	calls := 0
	f := sync.OnceFunc(func() { calls++ })
	allocs := testing.AllocsPerRun(10, f)
	if calls != 1 {
		t.Errorf("want calls==1, got %d", calls)
	}
	if allocs != 0 {
		t.Errorf("want 0 allocations per call, got %v", allocs)
	}
}

func TestOnceValue(t *testing.T) {
	calls := 0
	f := sync.OnceValue(func() int {
		calls++
		return calls
	})
	allocs := testing.AllocsPerRun(10, func() { f() })
	value := f()
	if calls != 1 {
		t.Errorf("want calls==1, got %d", calls)
	}
	if value != 1 {
		t.Errorf("want value==1, got %d", value)
	}
	if allocs != 0 {
		t.Errorf("want 0 allocations per call, got %v", allocs)
	}
}

func TestOnceValues(t *testing.T) {
	calls := 0
	f := sync.OnceValues(func() (int, int) {
		calls++
		return calls, calls + 1
	})
	allocs := testing.AllocsPerRun(10, func() { f() })
	v1, v2 := f()
	if calls != 1 {
		t.Errorf("want calls==1, got %d", calls)
	}
	if v1 != 1 || v2 != 2 {
		t.Errorf("want v1==1 and v2==2, got %d and %d", v1, v2)
	}
	if allocs != 0 {
		t.Errorf("want 0 allocations per call, got %v", allocs)
	}
}

func testOncePanicX(t *testing.T, calls *int, f func()) {
	testOncePanicWith(t, calls, f, func(label string, p any) {
		if p != "x" {
			t.Fatalf("%s: want panic %v, got %v", label, "x", p)
		}
	})
}

func testOncePanicWith(t *testing.T, calls *int, f func(), check func(label string, p any)) {
	// Check that the each call to f panics with the same value, but the
	// underlying function is only called once.
	for _, label := range []string{"first time", "second time"} {
		var p any
		panicked := true
		func() {
			defer func() {
				p = recover()
			}()
			f()
			panicked = false
		}()
		if !panicked {
			t.Fatalf("%s: f did not panic", label)
		}
		check(label, p)
	}
	if *calls != 1 {
		t.Errorf("want calls==1, got %d", *calls)
	}
}

func TestOnceFuncPanic(t *testing.T) {
	calls := 0
	f := sync.OnceFunc(func() {
		calls++
		panic("x")
	})
	testOncePanicX(t, &calls, f)
}

func TestOnceValuePanic(t *testing.T) {
	calls := 0
	f := sync.OnceValue(func() int {
		calls++
		panic("x")
	})
	testOncePanicX(t, &calls, func() { f() })
}

func TestOnceValuesPanic(t *testing.T) {
	calls := 0
	f := sync.OnceValues(func() (int, int) {
		calls++
		panic("x")
	})
	testOncePanicX(t, &calls, func() { f() })
}

func TestOnceFuncPanicNil(t *testing.T) {
	calls := 0
	f := sync.OnceFunc(func() {
		calls++
		panic(nil)
	})
	testOncePanicWith(t, &calls, f, func(label string, p any) {
		switch p.(type) {
		case nil, *runtime.PanicNilError:
			return
		}
		t.Fatalf("%s: want nil panic, got %v", label, p)
	})
}

func TestOnceFuncGoexit(t *testing.T) {
	// If f calls Goexit, the results are unspecified. But check that f doesn't
	// get called twice.
	calls := 0
	f := sync.OnceFunc(func() {
		calls++
		runtime.Goexit()
	})
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { recover() }()
			f()
		}()
		wg.Wait()
	}
	if calls != 1 {
		t.Errorf("want calls==1, got %d", calls)
	}
}

func TestOnceFuncPanicTraceback(t *testing.T) {
	// Test that on the first invocation of a OnceFunc, the stack trace goes all
	// the way to the origin of the panic.
	f := sync.OnceFunc(onceFuncPanic)

	defer func() {
		if p := recover(); p != "x" {
			t.Fatalf("want panic %v, got %v", "x", p)
		}
		stack := debug.Stack()
		want := "sync_test.onceFuncPanic"
		if !bytes.Contains(stack, []byte(want)) {
			t.Fatalf("want stack containing %v, got:\n%s", want, string(stack))
		}
	}()
	f()
}

func onceFuncPanic() {
	panic("x")
}

func TestOnceXGC(t *testing.T) {
	fns := map[string]func([]byte) func(){
		"OnceFunc": func(buf []byte) func() {
			return sync.OnceFunc(func() { buf[0] = 1 })
		},
		"OnceValue": func(buf []byte) func() {
			f := sync.OnceValue(func() any { buf[0] = 1; return nil })
			return func() { f() }
		},
		"OnceValues": func(buf []byte) func() {
			f := sync.OnceValues(func() (any, any) { buf[0] = 1; return nil, nil })
			return func() { f() }
		},
	}
	for n, fn := range fns {
		t.Run(n, func(t *testing.T) {
			buf := make([]byte, 1024)
			var gc atomic.Bool
			runtime.AddCleanup(&buf[0], func(g *atomic.Bool) { g.Store(true) }, &gc)
			f := fn(buf)
			gcwaitfin()
			if gc.Load() != false {
				t.Fatal("wrapped function garbage collected too early")
			}
			f()
			gcwaitfin()
			if gc.Load() != true {
				// Even if f is still alive, the function passed to Once(Func|Value|Values)
				// is not kept alive after the first call to f.
				t.Fatal("wrapped function should be garbage collected, but still live")
			}
			f()
		})
	}
}

// gcwaitfin performs garbage collection and waits for all finalizers to run.
func gcwaitfin() {
	runtime.GC()
	runtime_blockUntilEmptyFinalizerQueue(math.MaxInt64)
}

//go:linkname runtime_blockUntilEmptyFinalizerQueue runtime.blockUntilEmptyFinalizerQueue
func runtime_blockUntilEmptyFinalizerQueue(int64) bool

var (
	onceFunc = sync.OnceFunc(func() {})

	onceFuncOnce sync.Once
)

func doOnceFunc() {
	onceFuncOnce.Do(func() {})
}

func BenchmarkOnceFunc(b *testing.B) {
	b.Run("v=Once", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// The baseline is direct use of sync.Once.
			doOnceFunc()
		}
	})
	b.Run("v=Global", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// As of 3/2023, the compiler doesn't recognize that onceFunc is
			// never mutated and is a closure that could be inlined.
			// Too bad, because this is how OnceFunc will usually be used.
			onceFunc()
		}
	})
	b.Run("v=Local", func(b *testing.B) {
		b.ReportAllocs()
		// As of 3/2023, the compiler *does* recognize this local binding as an
		// inlinable closure. This is the best case for OnceFunc, but probably
		// not typical usage.
		f := sync.OnceFunc(func() {})
		for i := 0; i < b.N; i++ {
			f()
		}
	})
}

var (
	onceValue = sync.OnceValue(func() int { return 42 })

	onceValueOnce  sync.Once
	onceValueValue int
)

func doOnceValue() int {
	onceValueOnce.Do(func() {
		onceValueValue = 42
	})
	return onceValueValue
}

func BenchmarkOnceValue(b *testing.B) {
	// See BenchmarkOnceFunc
	b.Run("v=Once", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if want, got := 42, doOnceValue(); want != got {
				b.Fatalf("want %d, got %d", want, got)
			}
		}
	})
	b.Run("v=Global", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if want, got := 42, onceValue(); want != got {
				b.Fatalf("want %d, got %d", want, got)
			}
		}
	})
	b.Run("v=Local", func(b *testing.B) {
		b.ReportAllocs()
		onceValue := sync.OnceValue(func() int { return 42 })
		for i := 0; i < b.N; i++ {
			if want, got := 42, onceValue(); want != got {
				b.Fatalf("want %d, got %d", want, got)
			}
		}
	})
}
