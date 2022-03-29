// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

// This file contains utility routines and harness infrastructure used
// by the ABI tests in "abiutils_test.go".

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"strings"
	"testing"
	"text/scanner"
)

func mkParamResultField(t *types.Type, s *types.Sym, which ir.Class) *types.Field {
	field := types.NewField(src.NoXPos, s, t)
	n := typecheck.NewName(s)
	n.Class = which
	field.Nname = n
	n.SetType(t)
	return field
}

// mkstruct is a helper routine to create a struct type with fields
// of the types specified in 'fieldtypes'.
func mkstruct(fieldtypes []*types.Type) *types.Type {
	fields := make([]*types.Field, len(fieldtypes))
	for k, t := range fieldtypes {
		if t == nil {
			panic("bad -- field has no type")
		}
		f := types.NewField(src.NoXPos, nil, t)
		fields[k] = f
	}
	s := types.NewStruct(types.LocalPkg, fields)
	return s
}

func mkFuncType(rcvr *types.Type, ins []*types.Type, outs []*types.Type) *types.Type {
	q := typecheck.Lookup("?")
	inf := []*types.Field{}
	for _, it := range ins {
		inf = append(inf, mkParamResultField(it, q, ir.PPARAM))
	}
	outf := []*types.Field{}
	for _, ot := range outs {
		outf = append(outf, mkParamResultField(ot, q, ir.PPARAMOUT))
	}
	var rf *types.Field
	if rcvr != nil {
		rf = mkParamResultField(rcvr, q, ir.PPARAM)
	}
	return types.NewSignature(types.LocalPkg, rf, nil, inf, outf)
}

type expectedDump struct {
	dump string
	file string
	line int
}

func tokenize(src string) []string {
	var s scanner.Scanner
	s.Init(strings.NewReader(src))
	res := []string{}
	for tok := s.Scan(); tok != scanner.EOF; tok = s.Scan() {
		res = append(res, s.TokenText())
	}
	return res
}

func verifyParamResultOffset(t *testing.T, f *types.Field, r abi.ABIParamAssignment, which string, idx int) int {
	n := ir.AsNode(f.Nname).(*ir.Name)
	if n.FrameOffset() != int64(r.Offset()) {
		t.Errorf("%s %d: got offset %d wanted %d t=%v",
			which, idx, r.Offset(), n.Offset_, f.Type)
		return 1
	}
	return 0
}

func makeExpectedDump(e string) expectedDump {
	return expectedDump{dump: e}
}

func difftokens(atoks []string, etoks []string) string {
	if len(atoks) != len(etoks) {
		return fmt.Sprintf("expected %d tokens got %d",
			len(etoks), len(atoks))
	}
	for i := 0; i < len(etoks); i++ {
		if etoks[i] == atoks[i] {
			continue
		}

		return fmt.Sprintf("diff at token %d: expected %q got %q",
			i, etoks[i], atoks[i])
	}
	return ""
}

func nrtest(t *testing.T, ft *types.Type, expected int) {
	types.CalcSize(ft)
	got := configAMD64.NumParamRegs(ft)
	if got != expected {
		t.Errorf("]\nexpected num regs = %d, got %d, type %v", expected, got, ft)
	}
}

func abitest(t *testing.T, ft *types.Type, exp expectedDump) {

	types.CalcSize(ft)

	// Analyze with full set of registers.
	regRes := configAMD64.ABIAnalyze(ft, false)
	regResString := strings.TrimSpace(regRes.String())

	// Check results.
	reason := difftokens(tokenize(regResString), tokenize(exp.dump))
	if reason != "" {
		t.Errorf("\nexpected:\n%s\ngot:\n%s\nreason: %s",
			strings.TrimSpace(exp.dump), regResString, reason)
	}

}
