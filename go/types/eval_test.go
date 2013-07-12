// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for Eval.

package types

import "testing"

func testEval(t *testing.T, pkg *Package, scope *Scope, str string, typ Type, typStr, valStr string) {
	gotTyp, gotVal, err := Eval(str, pkg, scope)
	if err != nil {
		t.Errorf("Eval(%s) failed: %s", str, err)
		return
	}
	if gotTyp == nil {
		t.Errorf("Eval(%s) got nil type but no error", str)
		return
	}

	// compare types
	if typ != nil {
		// we have a type, check identity
		if !IsIdentical(gotTyp, typ) {
			t.Errorf("Eval(%s) got type %s, want %s", str, gotTyp, typ)
			return
		}
	} else {
		// we have a string, compare type string
		gotStr := gotTyp.String()
		if gotStr != typStr {
			t.Errorf("Eval(%s) got type %s, want %s", str, gotStr, typStr)
			return
		}
	}

	// compare values
	gotStr := ""
	if gotVal != nil {
		gotStr = gotVal.String()
	}
	if gotStr != valStr {
		t.Errorf("Eval(%s) got value %s, want %s", str, gotStr, valStr)
	}
}

func TestEvalBasic(t *testing.T) {
	for _, typ := range Typ[Bool : String+1] {
		testEval(t, nil, nil, typ.name, typ, "", "")
	}
}

func TestEvalComposite(t *testing.T) {
	for _, test := range testTypes {
		testEval(t, nil, nil, test.src, nil, test.str, "")
	}
}

// TODO(gri) expand
