// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"reflect"
	"slices"
	"testing"
)

func ExampleClosure_All_tuple() {
	v := mustParse(`
- !sum [1, 2]
- !sum [3, 4]
`)
	printYaml(slices.Collect(v.All()))

	// Output:
	// - [1, 3]
	// - [1, 4]
	// - [2, 3]
	// - [2, 4]
}

func ExampleClosure_All_def() {
	v := mustParse(`
a: !sum [1, 2]
b: !sum [3, 4]
c: 5
`)
	printYaml(slices.Collect(v.All()))

	// Output:
	// - {a: 1, b: 3, c: 5}
	// - {a: 1, b: 4, c: 5}
	// - {a: 2, b: 3, c: 5}
	// - {a: 2, b: 4, c: 5}
}

func checkDecode[T any](t *testing.T, got *Value, want T) {
	var gotT T
	if err := got.Decode(&gotT); err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if !reflect.DeepEqual(&gotT, &want) {
		t.Fatalf("got:\n%s\nwant:\n%s", prettyYaml(gotT), prettyYaml(want))
	}
}
