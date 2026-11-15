// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

func TestCloneWithOverlaysLeavesOrigUnmutated(t *testing.T) {
	const src = `package overlayorig

func Plain(x int) int {
	return x
}

type Box struct {
	Name string
}

func (b Box) Count() int {
	return len(b.Name)
}

func Take(v const []int) int {
	return len(v)
}
`
	lowered, opts, consts := checkLoweredForOverlay(t, src)
	if len(opts)+len(consts) == 0 {
		t.Fatal("want overlay facts")
	}
	plain, ok := lowered.Scope().Lookup("Plain").(*types2.Func)
	if !ok {
		t.Fatal("lowered Plain is not a func")
	}
	count := lookupMethod(t, lowered, "Box", "Count")
	if plain.LogosReadonly() || count.LogosReadonly() {
		t.Fatal("lowered package already carries readonly")
	}
	fns := []facts.FnFact{
		{Object: "Plain", Readonly: true},
		{Object: "Box.Count", Readonly: true},
	}
	cloned, err := CloneWithOverlaysAndReadonly(lowered, opts, consts, fns, nil, nil)
	if err != nil {
		t.Fatalf("CloneWithOverlaysAndReadonly failed: %v", err)
	}
	if cloned == lowered {
		t.Fatal("overlay path returned the input package")
	}
	if plain.LogosReadonly() || count.LogosReadonly() {
		t.Fatal("overlay clone mutated the shared input package")
	}
	cplain, ok := cloned.Scope().Lookup("Plain").(*types2.Func)
	if !ok || !cplain.LogosReadonly() {
		t.Fatal("clone lost the Plain readonly fact")
	}
	if got := lookupMethod(t, cloned, "Box", "Count"); !got.LogosReadonly() {
		t.Fatal("clone lost the Box.Count readonly fact")
	}
}
