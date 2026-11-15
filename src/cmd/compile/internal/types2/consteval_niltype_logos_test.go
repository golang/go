// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	"strings"
	"testing"
)

func TestLogosConstComputeNilTypeFailsClosed(t *testing.T) {
	check := logosTestChecker()
	var msgs []string
	check.conf.Error = func(err error) { msgs = append(msgs, err.Error()) }
	f, err := syntax.Parse(syntax.NewFileBase("niltype.lgo"), strings.NewReader("package p\nvar _ = []int{1}\n"), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	vd, ok := f.DeclList[0].(*syntax.VarDecl)
	if !ok {
		t.Fatalf("want *VarDecl, got %T", f.DeclList[0])
	}
	obj := NewConst(nopos, nil, "X", nil, constant.MakeUnknown())
	x := &operand{mode_: constant_}
	if !check.logosConstCompute(obj, vd.Values, x) {
		t.Fatalf("logosConstCompute = false, want true")
	}
	if obj.typ != Typ[Invalid] {
		t.Fatalf("obj.typ = %v, want Invalid", obj.typ)
	}
	if len(msgs) != 1 {
		t.Fatalf("msgs = %q, want one error", msgs)
	}
	if !strings.Contains(msgs[0], "[correctness::const_eval]") {
		t.Fatalf("error = %q, want const_eval rule", msgs[0])
	}
}
