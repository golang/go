// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import "testing"

func fpeq(fp1, fp2 FuncProps) bool {
	if fp1.Flags != fp2.Flags {
		return false
	}
	if len(fp1.ParamFlags) != len(fp2.ParamFlags) {
		return false
	}
	for i := range fp1.ParamFlags {
		if fp1.ParamFlags[i] != fp2.ParamFlags[i] {
			return false
		}
	}
	if len(fp1.ResultFlags) != len(fp2.ResultFlags) {
		return false
	}
	for i := range fp1.ResultFlags {
		if fp1.ResultFlags[i] != fp2.ResultFlags[i] {
			return false
		}
	}
	return true
}

func TestSerDeser(t *testing.T) {
	testcases := []FuncProps{
		FuncProps{},
		FuncProps{
			Flags: 0xfffff,
		},
		FuncProps{
			Flags:       1,
			ResultFlags: []ResultPropBits{ResultAlwaysSameConstant},
		},
		FuncProps{
			Flags:       1,
			ParamFlags:  []ParamPropBits{0x99, 0xaa, 0xfffff},
			ResultFlags: []ResultPropBits{0xfeedface},
		},
	}

	for k, tc := range testcases {
		s := tc.SerializeToString()
		fp := DeserializeFromString(s)
		got := fp.String()
		want := tc.String()
		if !fpeq(*fp, tc) {
			t.Errorf("eq check failed for test %d: got:\n%s\nwant:\n%s\n", k, got, want)
		}
	}

	var nilt *FuncProps
	ns := nilt.SerializeToString()
	nfp := DeserializeFromString(ns)
	if len(ns) != 0 || nfp != nil {
		t.Errorf("nil serialize/deserialize failed")
	}
}
