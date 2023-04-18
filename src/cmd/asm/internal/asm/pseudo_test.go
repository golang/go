// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"strings"
	"testing"

	"cmd/asm/internal/lex"
)

func tokenize(s string) [][]lex.Token {
	res := [][]lex.Token{}
	if len(s) == 0 {
		return res
	}
	for _, o := range strings.Split(s, ",") {
		res = append(res, lex.Tokenize(o))
	}
	return res
}

func TestErroneous(t *testing.T) {

	type errtest struct {
		pseudo   string
		operands string
		expected string
	}

	nonRuntimeTests := []errtest{
		{"TEXT", "", "expect two or three operands for TEXT"},
		{"TEXT", "%", "expect two or three operands for TEXT"},
		{"TEXT", "1, 1", "TEXT symbol \"<erroneous symbol>\" must be a symbol(SB)"},
		{"TEXT", "$\"foo\", 0, $1", "TEXT symbol \"<erroneous symbol>\" must be a symbol(SB)"},
		{"TEXT", "$0É:0, 0, $1", "expected end of operand, found É"}, // Issue #12467.
		{"TEXT", "$:0:(SB, 0, $1", "expected '(', found 0"},          // Issue 12468.
		{"TEXT", "@B(SB),0,$0", "expected '(', found B"},             // Issue 23580.
		{"TEXT", "foo<ABIInternal>(SB),0", "ABI selector only permitted when compiling runtime, reference was to \"foo\""},
		{"FUNCDATA", "", "expect two operands for FUNCDATA"},
		{"FUNCDATA", "(SB ", "expect two operands for FUNCDATA"},
		{"DATA", "", "expect two operands for DATA"},
		{"DATA", "0", "expect two operands for DATA"},
		{"DATA", "(0), 1", "expect /size for DATA argument"},
		{"DATA", "@B(SB)/4,0", "expected '(', found B"}, // Issue 23580.
		{"DATA", "·A(SB)/4,0", "DATA value must be an immediate constant or address"},
		{"DATA", "·B(SB)/4,$0", ""},
		{"DATA", "·C(SB)/5,$0", "bad int size for DATA argument: 5"},
		{"DATA", "·D(SB)/5,$0.0", "bad float size for DATA argument: 5"},
		{"DATA", "·E(SB)/4,$·A(SB)", "bad addr size for DATA argument: 4"},
		{"DATA", "·F(SB)/8,$·A(SB)", ""},
		{"DATA", "·G(SB)/5,$\"abcde\"", ""},
		{"GLOBL", "", "expect two or three operands for GLOBL"},
		{"GLOBL", "0,1", "GLOBL symbol \"<erroneous symbol>\" must be a symbol(SB)"},
		{"GLOBL", "@B(SB), 0", "expected '(', found B"}, // Issue 23580.
		{"PCDATA", "", "expect two operands for PCDATA"},
		{"PCDATA", "1", "expect two operands for PCDATA"},
	}

	runtimeTests := []errtest{
		{"TEXT", "foo<ABIInternal>(SB),0", "TEXT \"foo\": ABIInternal requires NOSPLIT"},
	}

	testcats := []struct {
		compilingRuntime bool
		tests            []errtest
	}{
		{
			compilingRuntime: false,
			tests:            nonRuntimeTests,
		},
		{
			compilingRuntime: true,
			tests:            runtimeTests,
		},
	}

	// Note these errors should be independent of the architecture.
	// Just run the test with amd64.
	parser := newParser("amd64")
	var buf strings.Builder
	parser.errorWriter = &buf

	for _, cat := range testcats {
		for _, test := range cat.tests {
			parser.compilingRuntime = cat.compilingRuntime
			parser.errorCount = 0
			parser.lineNum++
			if !parser.pseudo(test.pseudo, tokenize(test.operands)) {
				t.Fatalf("Wrong pseudo-instruction: %s", test.pseudo)
			}
			errorLine := buf.String()
			if test.expected != errorLine {
				t.Errorf("Unexpected error %q; expected %q", errorLine, test.expected)
			}
			buf.Reset()
		}
	}

}
