// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"cmd/compile/internal/types"
	"reflect"
	"slices"
	"sort"
	"testing"
)

func TestSymLessAndCmp(t *testing.T) {
	var (
		local = types.NewPkg("", "")
		abc   = types.NewPkg("abc", "")
		uvw   = types.NewPkg("uvw", "")
		xyz   = types.NewPkg("xyz", "")
		gr    = types.NewPkg("gr", "")
	)

	data := []*types.Sym{
		abc.Lookup("b"),
		local.Lookup("B"),
		local.Lookup("C"),
		uvw.Lookup("c"),
		local.Lookup("C"),
		gr.Lookup("φ"),
		local.Lookup("Φ"),
		xyz.Lookup("b"),
		abc.Lookup("a"),
		local.Lookup("B"),
	}
	data2 := slices.Clone(data)
	want := []*types.Sym{
		local.Lookup("B"),
		local.Lookup("B"),
		local.Lookup("C"),
		local.Lookup("C"),
		local.Lookup("Φ"),
		abc.Lookup("a"),
		abc.Lookup("b"),
		xyz.Lookup("b"),
		uvw.Lookup("c"),
		gr.Lookup("φ"),
	}
	if len(data) != len(want) {
		t.Fatal("want and data must match")
	}
	if reflect.DeepEqual(data, want) {
		t.Fatal("data must be shuffled")
	}
	sort.Slice(data, func(i, j int) bool { return data[i].Less(data[j]) })
	if !reflect.DeepEqual(data, want) {
		t.Logf("want: %#v", want)
		t.Logf("data: %#v", data)
		t.Errorf("sorting failed")
	}
	slices.SortFunc(data2, func(a, b *types.Sym) int {
		return a.Compare(b)
	})
	if !reflect.DeepEqual(data2, want) {
		t.Logf("want: %#v", want)
		t.Logf("data2: %#v", data2)
		t.Errorf("sorting failed")
	}
}
