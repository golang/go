// errorcheck -0 -d=escapealias=1

//go:build goexperiment.runtimefreegc

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test recognizing certain patterns of usage,
// currently focused on whether a slice is aliased.

package escapealias

import "runtime"

// Basic examples.
//
// Some of these directly overlap with later tests below, but are presented at the start
// to help show the big picture (before going into more variations).

var alias []int

func basic1() {
	// A simple append with no aliasing of s.
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	_ = s
}

func basic2() []int {
	// The slice can escape.
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	return s
}

func basic3() {
	// A simple example of s being aliased.
	// We give up when we see the aliasing.
	var s []int
	alias = s
	s = append(s, 0)
	_ = s
}

func basic4() {
	// The analysis is conservative, giving up on
	// IR nodes it doesn't understand. It does not
	// yet understand comparisons, for example.
	var s []int
	_ = s == nil
	s = append(s, 0)
	_ = s
}

func basic5() {
	// We also give up if s is assigned to another variable.
	var s []int
	s2 := s
	s2 = append(s2, 0)
	_ = s2
}

func basic6() {
	// A self-assigning append does not create an alias,
	// so s is still unaliased when we reach the second append here.
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	s = append(s, 0) // ERROR "append using non-aliased slice"
	_ = s
}

func basic7() {
	// An append can be unaliased if it happens before aliasing.
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	alias = s
	s = append(s, 0)
	_ = s
}

func basic8() {
	// Aliasing anywhere in a loop means we give up for the whole loop body,
	// even if the aliasing is after the append in the loop body.
	var s []int
	for range 10 {
		s = append(s, 0)
		alias = s
	}
	_ = s
}

func basic9() {
	// Aliases after a loop do not affect whether this is aliasing in the loop.
	var s []int
	for range 10 {
		s = append(s, 0) // ERROR "append using non-aliased slice"
	}
	alias = s
	_ = s
}

func basic10() {
	// We track the depth at which a slice is declared vs. aliased,
	// which helps for example with nested loops.
	// In this example, the aliasing occurs after both loops are done.
	var s []int
	for range 10 {
		for range 10 {
			s = append(s, 0) // ERROR "append using non-aliased slice"
		}
	}
	alias = s
}

func basic11() {
	// In contrast, here the aliasing occurs in the outer loop body.
	var s []int
	for range 10 {
		for range 10 {
			s = append(s, 0)
		}
		alias = s
	}
}

// Some variations on single appends.

func singleAppend1() []int {
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	return s
}

func singleAppend2() {
	var s []int
	alias = s
	s = append(s, 0)
}

func singleAppend3() {
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	alias = s
}

func singleAppend4() {
	var s []int
	p := &s
	_ = p
	s = append(s, 0)
}

func singleAppend5(s []int) {
	s = append(s, 0)
}

func singleAppend6() {
	var s []int
	alias, _ = s, 0
	s = append(s, 0)
}

// Examples with variations on slice declarations.

func sliceDeclaration1() {
	s := []int{}
	s = append(s, 0) // ERROR "append using non-aliased slice"
}

func sliceDeclaration2() {
	s := []int{1, 2, 3}
	s = append(s, 0) // ERROR "append using non-aliased slice"
}

func sliceDeclaration3() {
	s := make([]int, 3)
	s = append(s, 0) // ERROR "append using non-aliased slice"
}

func sliceDeclaration4() {
	s := []int{}
	alias = s
	s = append(s, 0)
}

func sliceDeclaration5() {
	s := []int{1, 2, 3}
	alias = s
	s = append(s, 0)
}

func sliceDeclaration6() {
	s := make([]int, 3)
	alias = s
	s = append(s, 0)
}

func sliceDeclaration7() {
	s, x := []int{}, 0
	s = append(s, x) // ERROR "append using non-aliased slice"
}

// Basic loops. First, a single loop.

func loops1a() {
	var s []int
	for i := range 10 {
		s = append(s, i) // ERROR "append using non-aliased slice"
	}
}

func loops1b() {
	var s []int
	for i := range 10 {
		alias = s
		s = append(s, i)
	}
}

func loops1c() {
	var s []int
	for i := range 10 {
		s = append(s, i)
		alias = s
	}
}

func loops1d() {
	var s []int
	for i := range 10 {
		s = append(s, i) // ERROR "append using non-aliased slice"
	}
	alias = s
}

func loops1e() {
	var s []int
	for i := range use(s) {
		s = append(s, i)
	}
}

func loops1f() {
	var s []int
	for i := range use(s) {
		s = append(s, i)
	}
	s = append(s, 0)
}

// Nested loops with s declared outside the loops.

func loops2a() {
	var s []int
	for range 10 {
		for i := range 10 {
			s = append(s, i) // ERROR "append using non-aliased slice"
		}
	}
}

func loops2b() {
	var s []int
	for range 10 {
		alias = s
		for i := range 10 {
			s = append(s, i)
		}
	}
}

func loops2c() {
	var s []int
	for range 10 {
		for i := range 10 {
			s = append(s, i)
		}
		alias = s
	}
}

func loops2d() {
	var s []int
	for range 10 {
		for i := range 10 {
			s = append(s, i) // ERROR "append using non-aliased slice"
		}
	}
	alias = s
}

func loops2e() {
	var s []int
	for range use(s) {
		for i := range 10 {
			s = append(s, i)
		}
		s = append(s, 0)
	}
	s = append(s, 0)
}

func loops2f() {
	var s []int
	for range 10 {
		for i := range use(s) {
			s = append(s, i)
		}
		s = append(s, 0)
	}
	s = append(s, 0)
}

// Nested loops with s declared inside the first loop.

func loops3a() {
	for range 10 {
		var s []int
		for i := range 10 {
			s = append(s, i) // ERROR "append using non-aliased slice"
		}
	}
}

func loops3b() {
	for range 10 {
		var s []int
		for i := range 10 {
			alias = s
			s = append(s, i)
		}
	}
}

func loops3c() {
	for range 10 {
		var s []int
		for i := range 10 {
			s = append(s, i)
			alias = s
		}
	}
}

func loops3d() {
	for range 10 {
		var s []int
		for i := range 10 {
			s = append(s, i) // ERROR "append using non-aliased slice"
		}
		alias = s
	}
}

func loops3e() {
	for range 10 {
		var s []int
		for i := range use(s) {
			s = append(s, i)
		}
		s = append(s, 0)
	}
}

// Loops using OFOR instead of ORANGE.

func loops4a() {
	var s []int
	for i := 0; i < 10; i++ {
		s = append(s, i) // ERROR "append using non-aliased slice"
	}
}

func loops4b() {
	var s []int
	for i := 0; i < 10; i++ {
		alias = s
		s = append(s, i)
	}
}

func loops4c() {
	var s []int
	for i := 0; i < 10; i++ {
		s = append(s, i)
		alias = s
	}
}

func loops4d() {
	var s []int
	for i := 0; i < 10; i++ {
		s = append(s, i) // ERROR "append using non-aliased slice"
	}
	alias = s
}

// Loops with some initialization variations.

func loopsInit1() {
	var i int
	for s := []int{}; i < 10; i++ {
		s = append(s, i) // ERROR "append using non-aliased slice"
	}
}

func loopsInit2() {
	var i int
	for s := []int{}; i < 10; i++ {
		s = append(s, i)
		alias = s
	}
}

func loopsInit3() {
	var i int
	for s := []int{}; i < 10; i++ {
		for range 10 {
			s = append(s, i) // ERROR "append using non-aliased slice"
		}
	}
}

func loopsInit5() {
	var i int
	for s := []int{}; i < 10; i++ {
		for range 10 {
			s = append(s, i)
			alias = s
		}
	}
}

func loopsInit5b() {
	var i int
	for s := []int{}; i < 10; i++ {
		for range 10 {
			s = append(s, i)
		}
		alias = s
	}
}

func loopsInit6() {
	for range 10 {
		var i int
		for s := []int{}; i < 10; i++ {
			s = append(s, i) // ERROR "append using non-aliased slice"
		}
	}
}

func loopsInit7() {
	for range 10 {
		var i int
		for s := []int{}; i < 10; i++ {
			s = append(s, i)
			alias = s
		}
	}
}

// Some initialization variations with use of s in the for or range.

func loopsInit8() {
	var s []int
	for use(s) == 0 {
		s = append(s, 0)
	}
}

func loopsInit9() {
	for s := []int{}; use(s) == 0; {
		s = append(s, 0)
	}
}

func loopsInit10() {
	for s := []int{}; ; use(s) {
		s = append(s, 0)
	}
}

func loopsInit11() {
	var s [][]int
	for _, s2 := range s {
		s = append(s, s2)
	}
}

// Examples of calling functions that get inlined,
// starting with a simple pass-through function.

// TODO(thepudds): we handle many of these starting in https://go.dev/cl/712422

func inlineReturn(param []int) []int {
	return param
}

func inline1a() {
	var s []int
	s = inlineReturn(s)
	s = append(s, 0)
}

func inline1b() {
	var s []int
	for range 10 {
		s = inlineReturn(s)
		s = append(s, 0)
	}
}

func inline1c() {
	var s []int
	for range 10 {
		s = inlineReturn(s)
		alias = s
		s = append(s, 0)
	}
}

func inline1d() {
	var s []int
	for range 10 {
		s = inlineReturn(s)
		s = append(s, 0)
		alias = s
	}
}

// Examples with an inlined function that uses append.

func inlineAppend(param []int) []int {
	param = append(param, 0)
	// TODO(thepudds): could in theory also handle a direct 'return append(param, 0)'
	return param
}

func inline2a() {
	var s []int
	s = inlineAppend(s)
	s = append(s, 0)
}

func inline2b() {
	var s []int
	for range 10 {
		s = inlineAppend(s)
		s = append(s, 0)
	}
}

func inline2c() {
	var s []int
	for range 10 {
		s = inlineAppend(s)
		alias = s
		s = append(s, 0)
	}
}

func inline2d() {
	var s []int
	for range 10 {
		s = inlineAppend(s)
		s = append(s, 0)
		alias = s
	}
}

// Examples calling non-inlined functions that do and do not escape.

var sink interface{}

//go:noinline
func use(s []int) int { return 0 } // s content does not escape

//go:noinline
func escape(s []int) int { sink = s; return 0 } // s content escapes

func call1() {
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	use(s)
}

// TODO(thepudds): OK to disallow this for now, but would be nice to allow this given use(s) is non-escaping.
func call2() {
	var s []int
	use(s)
	s = append(s, 0)
}

func call3() {
	var s []int
	s = append(s, use(s))
}

func call4() {
	var s []int
	for i := range 10 {
		s = append(s, i)
		use(s)
	}
}

func callEscape1() {
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	escape(s)
}

func callEscape2() {
	var s []int
	escape(s)
	s = append(s, 0)
}

func callEscape3() {
	var s []int
	s = append(s, escape(s))
}

func callEscape4() {
	var s []int
	for i := range 10 {
		s = append(s, i)
		escape(s)
	}
}

// Examples of some additional expressions we understand.

func expr1() {
	var s []int
	_ = len(s)
	_ = cap(s)
	s = append(s, 0) // ERROR "append using non-aliased slice"
}

// Examples of some expressions or statements we do not understand.
// Some of these we could handle in the future, but some likely not.

func notUnderstood1() {
	var s []int
	s = append(s[:], 0)
}

func notUnderstood2() {
	// Note: we must be careful if we analyze slice expressions.
	// See related comment about slice expressions in (*aliasAnalysis).analyze.
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	s = s[1:]        // s no longer points to the base of the heap object.
	s = append(s, 0)
}

func notUnderstood3() {
	// The first append is currently the heart of slices.Grow.
	var s []int
	n := 1000
	s = append(s[:cap(s)], make([]int, n)...)[:len(s)]
	s = append(s, 0)
}

func notUnderstood4() []int {
	// A return statement could be allowed to use the slice in a loop
	// because we cannot revisit the append once we return.
	var s []int
	for i := range 10 {
		s = append(s, 0)
		if i > 5 {
			return s
		}
	}
	return s
}

func notUnderstood5() {
	// AddCleanup is an example function call that we do not understand.
	// See related comment about specials in (*aliasAnalysis).analyze.
	var s []int
	runtime.AddCleanup(&s, func(int) {}, 0)
	s = append(s, 0)
}

// Examples with closures.

func closure1() {
	var s []int // declared outside the closure
	f := func() {
		for i := range 10 {
			s = append(s, i)
		}
	}
	_ = f // avoid calling f, which would just get inlined
}

// TODO(thepudds): it's probably ok that we currently allow this. Could conservatively
// disallow if needed.
func closure2() {
	f := func() {
		var s []int // declared inside the closure
		for i := range 10 {
			s = append(s, i) // ERROR "append using non-aliased slice"
		}
	}
	_ = f // avoid calling f, which would just get inlined
}

// Examples with goto and labels.

func goto1() {
	var s []int
label:
	s = append(s, 0)
	alias = s
	goto label
}

func goto2() {
	var s []int
	s = append(s, 0) // ERROR "append using non-aliased slice"
	alias = s
label:
	goto label
}

func goto3() {
	var s []int
label:
	for i := range 10 {
		s = append(s, i)
	}
	goto label
}

func break1() {
	var s []int
label:
	for i := range 10 {
		s = append(s, i)
		break label
	}
}

// Examples with iterators.

func collect[E any](seq Seq[E]) []E {
	var result []E
	for v := range seq {
		result = append(result, v)
	}
	return result
}

func count(yield func(int) bool) {
	for i := range 10 {
		if !yield(i) {
			return
		}
	}
}

func iteratorUse1() {
	var s []int
	s = collect(count)
	_ = s
}

func iteratorUse2() {
	var s []int
	s = collect(count)
	s = append(s, 0)
}

type Seq[E any] func(yield func(E) bool)
