// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"testing"
)

//go:noinline
func issue74383Direct(x int) (*int, int) {
	switch x := any(x).(type) {
	case *int:
		return x, 0
	case int:
		return nil, x
	default:
		return nil, 0
	}
}

func issue74383Switch(x any) (*int, int) {
	switch x := x.(type) {
	case *int:
		return x, 0
	case int:
		return nil, x
	default:
		return nil, 0
	}
}

//go:noinline
func issue74383Inlined(x int) (*int, int) {
	return issue74383Switch(x)
}

//go:noinline
func issue74383Control(x int) (*int, int) { return nil, x }

var issue74383Pointer *int
var issue74383Integer int
var issue74383Interface any

func TestIssue74383Allocations(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	for _, test := range []struct {
		name string
		f    func(int) (*int, int)
	}{
		{"direct", issue74383Direct},
		{"inlined", issue74383Inlined},
	} {
		t.Run(test.name, func(t *testing.T) {
			x := 1000 // Avoid the runtime's cache of small boxed integers.
			allocs := testing.AllocsPerRun(100, func() {
				x++
				issue74383Pointer, issue74383Integer = test.f(x)
			})
			if allocs != 0 {
				t.Errorf("got %v allocations, want 0", allocs)
			}
		})
	}
}

func TestIssue74383Values(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, x := range []int{-maxInt - 1, -1000, -1, 0, 1, 255, 256, 1000, maxInt} {
		for _, f := range []func(int) (*int, int){issue74383Direct, issue74383Inlined} {
			if p, got := f(x); p != nil || got != x {
				t.Errorf("input %d: got (%v, %d), want (nil, %d)", x, p, got, x)
			}
		}
	}
	var p *int
	if got, _ := issue74383Switch(p); got != nil {
		t.Errorf("typed nil: got %v", got)
	}
	if got, n := issue74383Switch(nil); got != nil || n != 0 {
		t.Errorf("nil interface: got (%v, %d)", got, n)
	}
	classify := func(x any) string {
		switch x.(type) {
		case nil:
			return "nil interface"
		case *int:
			return "pointer"
		default:
			return "other"
		}
	}
	if got := classify(p); got != "pointer" {
		t.Errorf("typed nil selected %q", got)
	}
	if got := classify(nil); got != "nil interface" {
		t.Errorf("nil interface selected %q", got)
	}
	x := 1000
	if got, _ := issue74383Switch(&x); got != &x {
		t.Errorf("pointer: got %v, want %v", got, &x)
	}
	type namedInt int
	if got, n := issue74383Switch(namedInt(x)); got != nil || n != 0 {
		t.Errorf("named int: got (%v, %d)", got, n)
	}
}

//go:noinline
func issue74383MatchingPointer(x int) *int {
	switch y := any(&x).(type) {
	case int:
		return nil
	case *int:
		return y
	}
	return nil
}

type issue74383NamedPointer *int

//go:noinline
func issue74383MatchingNamedPointer(x int) issue74383NamedPointer {
	switch y := any(issue74383NamedPointer(&x)).(type) {
	case *int:
		return nil
	case issue74383NamedPointer:
		return y
	}
	return nil
}

type issue74383Value int

func (x issue74383Value) Value() int { return int(x) }

//go:noinline
func issue74383MatchingInterface(x int) any {
	switch y := any(issue74383Value(x)).(type) {
	case *int:
		return y
	case interface{ Value() int }:
		return y
	}
	return nil
}

//go:noinline
func issue74383MatchingMultiple(x int) any {
	switch y := any(x).(type) {
	case int, string:
		return y
	default:
		return nil
	}
}

//go:noinline
func issue74383MatchingDefault(x int) any {
	switch y := any(x).(type) {
	case *int:
		return y
	default:
		return y
	}
}

//go:noinline
func issue74383Generic[T any](x T) any {
	switch y := any(x).(type) {
	case *T:
		return y
	case T:
		return y
	default:
		return nil
	}
}

//go:noinline
func issue74383GenericCase[T any](x int) any {
	switch y := any(x).(type) {
	case T:
		return y
	default:
		return nil
	}
}

func TestIssue74383EscapingCases(t *testing.T) {
	p := issue74383MatchingPointer(1000)
	q := issue74383MatchingPointer(2000)
	if p == nil || q == nil || p == q || *p != 1000 || *q != 2000 {
		t.Fatalf("matching pointer cases: got %v, %v", p, q)
	}
	*p = 3000
	if *q != 2000 {
		t.Errorf("pointer values alias: got %d", *q)
	}
	np := issue74383MatchingNamedPointer(1000)
	nq := issue74383MatchingNamedPointer(2000)
	if np == nil || nq == nil || np == nq || *np != 1000 || *nq != 2000 {
		t.Fatalf("matching named pointer cases: got %v, %v", np, nq)
	}
	for _, f := range []func(int) any{issue74383MatchingInterface, issue74383MatchingMultiple, issue74383MatchingDefault} {
		issue74383Interface = f(1000)
		switch x := issue74383Interface.(type) {
		case int:
			if x != 1000 {
				t.Errorf("matching case: got %d", x)
			}
		case issue74383Value:
			if x.Value() != 1000 {
				t.Errorf("matching interface: got %d", x.Value())
			}
		default:
			t.Errorf("matching case: got %T", x)
		}
	}
	if got := issue74383Generic(issue74383Value(1000)); got != issue74383Value(1000) {
		t.Errorf("generic value: got %v", got)
	}
	if got := issue74383Generic(p); got != p {
		t.Errorf("generic pointer: got %v, want %v", got, p)
	}
	issue74383Interface = issue74383GenericCase[int](1000)
	if issue74383Interface != 1000 {
		t.Errorf("matching generic case: got %v", issue74383Interface)
	}
	if got := issue74383GenericCase[*int](1000); got != nil {
		t.Errorf("non-matching generic case: got %v", got)
	}
}

func TestIssue74383EscapingAllocations(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	for _, test := range []struct {
		name string
		f    func(int) any
	}{
		{"pointer", func(x int) any { return issue74383MatchingPointer(x) }},
		{"named pointer", func(x int) any { return issue74383MatchingNamedPointer(x) }},
		{"interface", issue74383MatchingInterface},
		{"multiple", issue74383MatchingMultiple},
		{"default", issue74383MatchingDefault},
		{"generic", issue74383Generic[int]},
		{"generic case", issue74383GenericCase[int]},
	} {
		t.Run(test.name, func(t *testing.T) {
			x := 1000
			allocs := testing.AllocsPerRun(100, func() {
				x++
				issue74383Interface = test.f(x)
			})
			if allocs == 0 {
				t.Error("got no allocations for an escaping value")
			}
		})
	}
}

//go:noinline
func issue74383Effects(calls *int, fail bool) (*int, int) {
	value := func() int {
		*calls++
		if fail {
			panic("issue74383")
		}
		return 1000
	}
	switch x := any(value()).(type) {
	case *int:
		return x, 0
	case int:
		return nil, x
	}
	return nil, 0
}

func TestIssue74383Effects(t *testing.T) {
	calls := 0
	if p, n := issue74383Effects(&calls, false); p != nil || n != 1000 || calls != 1 {
		t.Fatalf("got (%v, %d), calls %d", p, n, calls)
	}
	defer func() {
		if got := recover(); got != "issue74383" || calls != 2 {
			t.Errorf("panic %v, calls %d", got, calls)
		}
	}()
	issue74383Effects(&calls, true)
}

func BenchmarkIssue74383(b *testing.B) {
	for _, test := range []struct {
		name string
		f    func(int) (*int, int)
	}{
		{"direct", issue74383Direct},
		{"inlined", issue74383Inlined},
		{"control", issue74383Control},
	} {
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				issue74383Pointer, issue74383Integer = test.f(1000 + (i & 1023))
			}
		})
	}
}
