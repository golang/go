// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"strings"
	"testing"

	"simd/archsimd/_gen/unify"
)

// TestAddUnifies checks that the loader's emitted ADD defs unify with the SVE op
// and type definitions (in the parent simdgen directory) to yield concrete Go
// API mappings.
func TestAddUnifies(t *testing.T) {
	defs := parse(t, addUnpred).emitAll()
	if len(defs) == 0 {
		t.Fatal("emitAll produced no defs")
	}
	inputs := []unify.Closure{unify.NewSum(defs...)}
	for _, path := range []string{"../go_sve.yaml", "../types.yaml", "../categories.yaml"} {
		cl, err := unify.ReadFile(path, unify.ReadOpts{})
		if err != nil {
			t.Fatalf("ReadFile %s: %v", path, err)
		}
		inputs = append(inputs, cl)
	}
	unified, err := unify.Unify(inputs...)
	if err != nil {
		t.Fatalf("Unify: %v", err)
	}
	var sawInt8s, sawUint8s bool
	for v := range unified.All() {
		if !v.Exact() {
			continue
		}
		s := v.String()
		if !strings.Contains(s, "Add") {
			continue
		}
		sawInt8s = sawInt8s || strings.Contains(s, "Int8s")
		sawUint8s = sawUint8s || strings.Contains(s, "Uint8s")
	}
	if !sawInt8s || !sawUint8s {
		t.Errorf("want Int8s and Uint8s Add mappings, got int8s=%v uint8s=%v", sawInt8s, sawUint8s)
	}
}
