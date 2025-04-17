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
	of := func() { calls++ }
	f := sync.OnceFunc(of)
	allocs := testing.AllocsPerRun(10, f)
	if calls != 1 {
		t.Errorf("want calls==1, got %d", calls)
	}
	if allocs != 0 {
		t.Errorf("want 0 allocations per call to f, got %v", allocs)
	}
	allocs = testing.AllocsPerRun(10, func() {
		f = sync.OnceFunc(of)
	})
	if allocs > 2 {
		t.Errorf("want at most 2 allocations per call to OnceFunc, got %v", allocs)
	}
}

func TestOnceValue(t *testing.T) {
	calls := 0
	of := func() int {
		calls++
		return calls
	}
	f := sync.OnceValue(of)
	allocs := testing.AllocsPerRun(10, func() { f() })
	value := f()
	if calls != 1 {
		t.Errorf("want calls==1, got %d", calls)
	}
	if value != 1 {
		t.Errorf("want value==1, got %d", value)
	}
	if allocs != 0 {
		t.Errorf("want 0 allocations per call to f, got %v", allocs)
	}
	allocs = testing.AllocsPerRun(10, func() {
		f = sync.OnceValue(of)
	})
	if allocs > 2 {
		t.Errorf("want at most 2 allocations per call to OnceValue, got %v", allocs)
	}
}

func TestOnceValues(t *testing.T) {
	calls := 0
	of := func() (int, int) {
		calls++
		return calls, calls + 1
	}
	f := sync.OnceValues(of)
	allocs := testing.AllocsPerRun(10, func() { f() })
	v1, v2 := f()
	if calls != 1 {
		t.Errorf("want calls==1, got %d", calls)
	}
	if v1 != 1 || v2 != 2 {
		t.Errorf("want v1==1 and v2==2, got %d and %d", v1, v2)
	}
	if allocs != 0 {
		t.Errorf("want 0 allocations per call to f, got %v", allocs)
	}
	allocs = testing.AllocsPerRun(10, func() {
		f = sync.OnceValues(of)
	})
	if allocs > 2 {
		t.Errorf("want at most 2 allocations per call to OnceValues, got %v", allocs)
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
		"OnceFunc panic": func(buf []byte) func() {
			return sync.OnceFunc(func() { buf[0] = 1; panic("test panic") })
		},
		"OnceValue panic": func(buf []byte) func() {
			f := sync.OnceValue(func() any { buf[0] = 1; panic("test panic") })
			return func() { f() }
		},
		"OnceValues panic": func(buf []byte) func() {
			f := sync.OnceValues(func() (any, any) { buf[0] = 1; panic("test panic") })
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
			func() {
				defer func() { recover() }()
				f()
			}()
			gcwaitfin()
			if gc.Load() != true {
				// Even if f is still alive, the function passed to Once(Func|Value|Values)
				// is not kept alive after the first call to f.
				t.Fatal("wrapped function should be garbage collected, but still live")
			}
			func() {
				defer func() { recover() }()
				f()
			}()
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

	onceFuncFunc func()
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
	b.Run("v=Make", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			onceFuncFunc = sync.OnceFunc(func() {})
		}
	})
}

var (
	onceValue = sync.OnceValue(func() int { return 42 })

	onceValueOnce  sync.Once
	onceValueValue int

	onceValueFunc func() int
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
	b.Run("v=Make", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			onceValueFunc = sync.OnceValue(func() int { return 42 })
		}
	})
}

const (
	onceValuesWant1 = 42
	onceValuesWant2 = true
)

var (
	onceValues = sync.OnceValues(func() (int, bool) {
		return onceValuesWant1, onceValuesWant2
	})

	onceValuesOnce   sync.Once
	onceValuesValue1 int
	onceValuesValue2 bool

	onceValuesFunc func() (int, bool)
)

func doOnceValues() (int, bool) {
	onceValuesOnce.Do(func() {
		onceValuesValue1 = onceValuesWant1
		onceValuesValue2 = onceValuesWant2
	})
	return onceValuesValue1, onceValuesValue2
}

func BenchmarkOnceValues(b *testing.B) {
	// See BenchmarkOnceFunc
	b.Run("v=Once", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if got1, got2 := doOnceValues(); got1 != onceValuesWant1 {
				b.Fatalf("value 1: got %d, want %d", got1, onceValuesWant1)
			} else if got2 != onceValuesWant2 {
				b.Fatalf("value 2: got %v, want %v", got2, onceValuesWant2)
			}
		}
	})
	b.Run("v=Global", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if got1, got2 := onceValues(); got1 != onceValuesWant1 {
				b.Fatalf("value 1: got %d, want %d", got1, onceValuesWant1)
			} else if got2 != onceValuesWant2 {
				b.Fatalf("value 2: got %v, want %v", got2, onceValuesWant2)
			}
		}
	})
	b.Run("v=Local", func(b *testing.B) {
		b.ReportAllocs()
		onceValues := sync.OnceValues(func() (int, bool) {
			return onceValuesWant1, onceValuesWant2
		})
		for i := 0; i < b.N; i++ {
			if got1, got2 := onceValues(); got1 != onceValuesWant1 {
				b.Fatalf("value 1: got %d, want %d", got1, onceValuesWant1)
			} else if got2 != onceValuesWant2 {
				b.Fatalf("value 2: got %v, want %v", got2, onceValuesWant2)
			}
		}
	})
	b.Run("v=Make", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			onceValuesFunc = sync.OnceValues(func() (int, bool) {
				return onceValuesWant1, onceValuesWant2
			})
		}
	})
}
