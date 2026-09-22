// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"fmt"
	"slices"
	"testing"
)

// TestScopeObjects tests that Scope.Objects yields elements in sorted name order, handles empty scopes, and supports early break.
func TestScopeObjects(t *testing.T) {
	s := NewScope(nil, nopos, nopos, "test")

	// Empty scope
	var gotNames []string
	for obj := range s.Objects() {
		gotNames = append(gotNames, obj.Name())
	}
	if len(gotNames) != 0 {
		t.Errorf("empty scope: got %v, want empty", gotNames)
	}

	// Insert objects out of order
	names := []string{"e", "c", "a", "b", "d"}
	for _, name := range names {
		v := NewVar(nopos, nil, name, Typ[Int])
		if alt := s.Insert(v); alt != nil {
			t.Fatalf("Insert(%s) failed", name)
		}
	}

	wantNames := []string{"a", "b", "c", "d", "e"}
	if !slices.Equal(s.Names(), wantNames) {
		t.Errorf("Names() = %v, want %v", s.Names(), wantNames)
	}

	// Scope.Objects yields elements in sorted name order
	gotNames = nil
	for obj := range s.Objects() {
		gotNames = append(gotNames, obj.Name())
		if obj != s.Lookup(obj.Name()) {
			t.Errorf("Objects() yielded %v, want Lookup result %v", obj, s.Lookup(obj.Name()))
		}
	}
	if !slices.Equal(gotNames, wantNames) {
		t.Errorf("Objects() yielded %v, want %v", gotNames, wantNames)
	}

	// Break early from iteration
	count := 0
	for range s.Objects() {
		count++
		if count == 2 {
			break
		}
	}
	if count != 2 {
		t.Errorf("early break: iterated %d times, want 2", count)
	}
}

// TestScopeMutationReflectsChanges tests that Scope.Insert clears the name cache so subsequent calls reflect mutations.
func TestScopeMutationReflectsChanges(t *testing.T) {
	s := NewScope(nil, nopos, nopos, "test")

	v1 := NewVar(nopos, nil, "b", Typ[Int])
	s.Insert(v1)

	// Prime cache
	if want := []string{"b"}; !slices.Equal(s.Names(), want) {
		t.Fatalf("Names() = %v, want %v", s.Names(), want)
	}

	// Insert before existing name
	v0 := NewVar(nopos, nil, "a", Typ[Int])
	s.Insert(v0)

	// Names() must reflect new object
	wantNames := []string{"a", "b"}
	if !slices.Equal(s.Names(), wantNames) {
		t.Errorf("after inserting 'a': Names() = %v, want %v", s.Names(), wantNames)
	}

	// Objects() must reflect new object
	var gotNames []string
	for obj := range s.Objects() {
		gotNames = append(gotNames, obj.Name())
	}
	if !slices.Equal(gotNames, wantNames) {
		t.Errorf("after inserting 'a': Objects() = %v, want %v", gotNames, wantNames)
	}

	// Insert after existing names
	v2 := NewVar(nopos, nil, "c", Typ[Int])
	s.Insert(v2)

	wantNames = []string{"a", "b", "c"}
	if !slices.Equal(s.Names(), wantNames) {
		t.Errorf("after inserting 'c': Names() = %v, want %v", s.Names(), wantNames)
	}
	gotNames = nil
	for obj := range s.Objects() {
		gotNames = append(gotNames, obj.Name())
	}
	if !slices.Equal(gotNames, wantNames) {
		t.Errorf("after inserting 'c': Objects() = %v, want %v", gotNames, wantNames)
	}

	// Duplicate insert should not mutate or break cache
	dup := NewVar(nopos, nil, "b", Typ[String])
	if alt := s.Insert(dup); alt != v1 {
		t.Errorf("Insert duplicate: got %v, want %v", alt, v1)
	}
	if !slices.Equal(s.Names(), wantNames) {
		t.Errorf("after duplicate Insert: Names() = %v, want %v", s.Names(), wantNames)
	}
}

// TestScopeNoAllocations tests that repeated calls to Scope.Names
// and Scope.Objects do not allocate once cached.
func TestScopeNoAllocations(t *testing.T) {
	s := NewScope(nil, nopos, nopos, "test")
	for i := range 10 {
		s.Insert(NewVar(nopos, nil, fmt.Sprintf("v%d", i), Typ[Int]))
	}

	// Prime the cache.
	_ = s.Names()

	// Scope.Names
	namesAllocs := testing.AllocsPerRun(100, func() { _ = s.Names() })
	if namesAllocs > 0 {
		t.Errorf("repeated s.Names() allocated %f times, want 0", namesAllocs)
	}

	// Scope.Objects allocates the iterator closure, and nothing else,
	// on top of what Names allocates.
	if allocs := testing.AllocsPerRun(100, func() {
		s.Objects()(func(Object) bool { return true })
	}); allocs > namesAllocs+1 {
		t.Errorf("repeated s.Objects() iteration allocated %f times, want at most %f", allocs, namesAllocs+1)
	}
}
