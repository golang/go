// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"bytes"
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

	tests := []struct {
		pseudo   string
		operands string
		expected string
	}{
		{"TEXT", "", "expect two or three operands for TEXT"},
		{"TEXT", "%", "expect two or three operands for TEXT"},
		{"TEXT", "1, 1", "TEXT symbol \"<erroneous symbol>\" must be a symbol(SB)"},
		{"TEXT", "$\"foo\", 0, $1", "TEXT symbol \"<erroneous symbol>\" must be a symbol(SB)"},
		{"TEXT", "$0É:0, 0, $1", "expected end of operand, found É"}, // Issue #12467.
		{"TEXT", "$:0:(SB, 0, $1", "expected '(', found 0"},          // Issue 12468.
		{"FUNCDATA", "", "expect two operands for FUNCDATA"},
		{"FUNCDATA", "(SB ", "expect two operands for FUNCDATA"},
		{"DATA", "", "expect two operands for DATA"},
		{"DATA", "0", "expect two operands for DATA"},
		{"DATA", "(0), 1", "expect /size for DATA argument"},
		{"GLOBL", "", "expect two or three operands for GLOBL"},
		{"GLOBL", "0,1", "GLOBL symbol \"<erroneous symbol>\" must be a symbol(SB)"},
		{"PCDATA", "", "expect two operands for PCDATA"},
		{"PCDATA", "1", "expect two operands for PCDATA"},
	}

	// Note these errors should be independent of the architecture.
	// Just run the test with amd64.
	parser := newParser("amd64")
	var buf bytes.Buffer
	parser.errorWriter = &buf

	for _, test := range tests {
		parser.errorCount = 0
		parser.lineNum++
		parser.histLineNum++
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
