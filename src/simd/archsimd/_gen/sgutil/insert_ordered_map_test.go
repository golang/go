// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import (
	"slices"
	"testing"
)

func TestInsertMapBasic(t *testing.T) {
	im := &InsertMap[string, int]{}

	// Test Contains and Get on empty map
	if im.Contains("A") {
		t.Error("empty map should not contain key A")
	}
	if val := im.Get("A"); val != 0 {
		t.Errorf("empty map Get(A) should return 0, got %d", val)
	}

	// Put elements
	im.Put("A", 1)
	im.Put("B", 2)
	im.Put("C", 3)

	// Test Contains
	if !im.Contains("A") || !im.Contains("B") || !im.Contains("C") {
		t.Error("map should contain keys A, B, and C")
	}
	if im.Contains("D") {
		t.Error("map should not contain key D")
	}

	// Test Get
	if val := im.Get("A"); val != 1 {
		t.Errorf("Get(A) expected 1, got %d", val)
	}
	if val := im.Get("B"); val != 2 {
		t.Errorf("Get(B) expected 2, got %d", val)
	}
	if val := im.Get("C"); val != 3 {
		t.Errorf("Get(C) expected 3, got %d", val)
	}
	if val := im.Get("D"); val != 0 {
		t.Errorf("Get(D) expected 0, got %d", val)
	}

	// Test Compare
	if cmp := im.Compare("A", "B"); cmp != -1 {
		t.Errorf("Compare(A, B) expected -1, got %d", cmp)
	}
	if cmp := im.Compare("B", "A"); cmp != 1 {
		t.Errorf("Compare(B, A) expected 1, got %d", cmp)
	}
	if cmp := im.Compare("A", "A"); cmp != 0 {
		t.Errorf("Compare(A, A) expected 0, got %d", cmp)
	}
	if cmp := im.Compare("A", "D"); cmp != -1 {
		t.Errorf("Compare(A, D) expected -1, got %d", cmp)
	}
	if cmp := im.Compare("D", "A"); cmp != 1 {
		t.Errorf("Compare(D, A) expected 1, got %d", cmp)
	}
	if cmp := im.Compare("D", "E"); cmp != 0 {
		t.Errorf("Compare(D, E) expected 0, got %d", cmp)
	}
}

func TestInsertMapUpdate(t *testing.T) {
	im := &InsertMap[string, int]{}

	im.Put("A", 1)
	im.Put("B", 2)

	// Update existing key
	im.Put("A", 10)

	if val := im.Get("A"); val != 10 {
		t.Errorf("Get(A) after update expected 10, got %d", val)
	}

	// Check if A is still ordered before B
	if cmp := im.Compare("A", "B"); cmp != -1 {
		t.Errorf("Compare(A, B) after update expected -1, got %d", cmp)
	}

	// Verify internal slice size (should be 2, not 3)
	if len(im.v) != 2 {
		t.Errorf("expected internal slice length 2, got %d. Slice content: %v", len(im.v), im.v)
	}
}

func TestInsertMapIterators(t *testing.T) {
	im := &InsertMap[string, int]{}
	im.Put("A", 1)
	im.Put("B", 2)
	im.Put("C", 3)

	// Test Keys iterator
	var keys []string
	for k := range im.Keys() {
		keys = append(keys, k)
	}
	expectedKeys := []string{"A", "B", "C"}
	if !slices.Equal(keys, expectedKeys) {
		t.Errorf("Keys() got %v, expected %v", keys, expectedKeys)
	}

	// Test Values iterator
	var values []int
	for v := range im.Values() {
		values = append(values, v)
	}
	expectedValues := []int{1, 2, 3}
	if !slices.Equal(values, expectedValues) {
		t.Errorf("Values() got %v, expected %v", values, expectedValues)
	}

	// Test All iterator
	var allKeys []string
	var allValues []int
	for k, v := range im.All() {
		allKeys = append(allKeys, k)
		allValues = append(allValues, v)
	}
	if !slices.Equal(allKeys, expectedKeys) {
		t.Errorf("All() keys got %v, expected %v", allKeys, expectedKeys)
	}
	if !slices.Equal(allValues, expectedValues) {
		t.Errorf("All() values got %v, expected %v", allValues, expectedValues)
	}
}
