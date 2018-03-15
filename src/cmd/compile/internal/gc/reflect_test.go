// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"reflect"
	"testing"
)

func TestSortingBySigLT(t *testing.T) {
	data := []*Sig{
		&Sig{name: "b", pkg: &types.Pkg{Path: "abc"}},
		&Sig{name: "B", pkg: nil},
		&Sig{name: "C", pkg: nil},
		&Sig{name: "c", pkg: &types.Pkg{Path: "uvw"}},
		&Sig{name: "C", pkg: nil},
		&Sig{name: "φ", pkg: &types.Pkg{Path: "gr"}},
		&Sig{name: "Φ", pkg: nil},
		&Sig{name: "b", pkg: &types.Pkg{Path: "xyz"}},
		&Sig{name: "a", pkg: &types.Pkg{Path: "abc"}},
		&Sig{name: "B", pkg: nil},
	}
	want := []*Sig{
		&Sig{name: "B", pkg: nil},
		&Sig{name: "B", pkg: nil},
		&Sig{name: "C", pkg: nil},
		&Sig{name: "C", pkg: nil},
		&Sig{name: "Φ", pkg: nil},
		&Sig{name: "a", pkg: &types.Pkg{Path: "abc"}},
		&Sig{name: "b", pkg: &types.Pkg{Path: "abc"}},
		&Sig{name: "b", pkg: &types.Pkg{Path: "xyz"}},
		&Sig{name: "c", pkg: &types.Pkg{Path: "uvw"}},
		&Sig{name: "φ", pkg: &types.Pkg{Path: "gr"}},
	}
	if len(data) != len(want) {
		t.Fatal("want and data must match")
	}
	if reflect.DeepEqual(data, want) {
		t.Fatal("data must be shuffled")
	}
	obj.SortSlice(data, func(i, j int) bool { return siglt(data[i], data[j]) })
	if !reflect.DeepEqual(data, want) {
		t.Logf("want: %#v", want)
		t.Logf("data: %#v", data)
		t.Errorf("sorting failed")
	}
}
