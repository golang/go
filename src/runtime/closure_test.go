// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/testenv"
	"testing"
)

var s int

func BenchmarkCallClosure(b *testing.B) {
	for i := 0; i < b.N; i++ {
		s += func(ii int) int { return 2 * ii }(i)
	}
}

func BenchmarkCallClosure1(b *testing.B) {
	for i := 0; i < b.N; i++ {
		j := i
		s += func(ii int) int { return 2*ii + j }(i)
	}
}

var ss *int

func BenchmarkCallClosure2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		j := i
		s += func() int {
			ss = &j
			return 2
		}()
	}
}

func addr1(x int) *int {
	return func() *int { return &x }()
}

func BenchmarkCallClosure3(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ss = addr1(i)
	}
}

func addr2() (x int, p *int) {
	return 0, func() *int { return &x }()
}

func BenchmarkCallClosure4(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, ss = addr2()
	}
}

var (
	genericClosureSink       func()
	genericNestedClosureSink func() func()
	genericIntClosureSink    func() int
	genericDictClosureSink   func() any
)

// Do not inline the factories: the returned closures must escape independently
// of whether their callers would otherwise inline them.

//go:noinline
func genericEmptyClosure[T any]() func() {
	return func() {}
}

//go:noinline
func genericNestedClosure[T any]() func() func() {
	return func() func() { return func() {} }
}

//go:noinline
func genericIntClosure[T any](n int) func() int {
	return func() int { return n }
}

//go:noinline
func genericSiblingClosures[T any]() (func(), func() any) {
	return func() {}, func() any {
		var zero T
		return zero
	}
}

func TestGenericClosureAllocation(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)

	type namedInt int
	nested := genericNestedClosure[int]()
	for _, test := range []struct {
		name string
		f    func()
		want float64
	}{
		{"Empty", func() { genericClosureSink = genericEmptyClosure[int]() }, 0},
		{"NestedOuter", func() { genericNestedClosureSink = genericNestedClosure[int]() }, 0},
		{"NestedInner", func() { genericClosureSink = nested() }, 0},
		// Only the second sibling needs a dictionary for the conversion to any.
		{"Siblings", func() { genericClosureSink, genericDictClosureSink = genericSiblingClosures[namedInt]() }, 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := testing.AllocsPerRun(100, test.f); got != test.want {
				t.Errorf("got %v allocations, want %v", got, test.want)
			}
		})
	}

	empty, dict := genericSiblingClosures[namedInt]()
	empty()
	if got := dict(); got != namedInt(0) {
		t.Fatalf("dictionary-dependent sibling returned %T(%v), want namedInt(0)", got, got)
	}

	// Removing the dictionary must not disturb the remaining int capture.
	first, second := genericIntClosure[int](17), genericIntClosure[string](23)
	if got := first(); got != 17 {
		t.Errorf("first closure returned %d, want 17", got)
	}
	if got := second(); got != 23 {
		t.Errorf("second closure returned %d, want 23", got)
	}
}

func BenchmarkGenericClosure(b *testing.B) {
	b.Run("Empty", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			genericClosureSink = genericEmptyClosure[int]()
		}
	})
	b.Run("NestedEmpty", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			genericNestedClosureSink = genericNestedClosure[int]()
			genericClosureSink = genericNestedClosureSink()
		}
	})
	b.Run("Int", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			genericIntClosureSink = genericIntClosure[int](42)
		}
	})
	b.Run("Siblings", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			genericClosureSink, genericDictClosureSink = genericSiblingClosures[int]()
		}
	})
}
