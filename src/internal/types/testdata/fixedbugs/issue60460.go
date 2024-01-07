// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Simplified (representative) test case.

func _() {
	f(R1{})
}

func f[T any](R[T]) {}

type R[T any] interface {
	m(R[T])
}

type R1 struct{}

func (R1) m(R[int]) {}

// Test case from issue.

func _() {
	r := newTestRules()
	NewSet(r)
	r2 := newTestRules2()
	NewSet(r2)
}

type Set[T any] struct {
	rules Rules[T]
}

func NewSet[T any](rules Rules[T]) Set[T] {
	return Set[T]{
		rules: rules,
	}
}

func (s Set[T]) Copy() Set[T] {
	return NewSet(s.rules)
}

type Rules[T any] interface {
	Hash(T) int
	Equivalent(T, T) bool
	SameRules(Rules[T]) bool
}

type testRules struct{}

func newTestRules() Rules[int] {
	return testRules{}
}

func (r testRules) Hash(val int) int {
	return val % 16
}

func (r testRules) Equivalent(val1 int, val2 int) bool {
	return val1 == val2
}

func (r testRules) SameRules(other Rules[int]) bool {
	_, ok := other.(testRules)
	return ok
}

type testRules2 struct{}

func newTestRules2() Rules[string] {
	return testRules2{}
}

func (r testRules2) Hash(val string) int {
	return 16
}

func (r testRules2) Equivalent(val1 string, val2 string) bool {
	return val1 == val2
}

func (r testRules2) SameRules(other Rules[string]) bool {
	_, ok := other.(testRules2)
	return ok
}
