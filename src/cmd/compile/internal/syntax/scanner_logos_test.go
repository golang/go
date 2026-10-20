// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

// logosScannerMode is the scanner.init mode bit that selects loGo/s
// lexing (M1-S1). The plan states the scanner receives the mode through
// scanner.init's mode uint field and gates '?' on s.mode&LogosMode, so
// these attacks drive the scanner directly with uint(LogosMode).
const logosScannerMode = uint(LogosMode)

// TestLogosQuestionAfterName checks that in LogosMode `x?` scans as
// _Name followed by _Question (M1-S1 Attack, contract C1-C2).
func TestLogosQuestionAfterName(t *testing.T) {
	var errs []string
	var s scanner
	s.init(strings.NewReader("x?"), func(_, _ uint, msg string) {
		errs = append(errs, msg)
	}, logosScannerMode)

	s.next()
	if s.tok != _Name || s.lit != "x" {
		t.Fatalf("got %s %q; want _Name %q", s.tok, s.lit, "x")
	}
	s.next()
	if s.tok != _Question {
		t.Fatalf("got %s; want _Question", s.tok)
	}
	// `?` sets nlsemi, so trailing EOF translates to ';' first.
	s.next()
	if s.tok != _Semi || s.lit != "EOF" {
		t.Fatalf("got %s %q; want _Semi %q", s.tok, s.lit, "EOF")
	}
	s.next()
	if s.tok != _EOF {
		t.Fatalf("got %s; want _EOF", s.tok)
	}
	if len(errs) != 0 {
		t.Fatalf("got errors %q; want none", errs)
	}
}

// TestLogosQuestionNlsemi checks that in LogosMode `?` sets nlsemi like
// an operand end: `f()?\n` inserts `;` after `?` (M1-S1 Attack).
func TestLogosQuestionNlsemi(t *testing.T) {
	for _, test := range []struct {
		src    string
		tokens []token
	}{
		{"x?\n", []token{_Name, _Question, _Semi, _EOF}},
		{"f()?\n", []token{_Name, _Lparen, _Rparen, _Question, _Semi, _EOF}},
	} {
		var errs []string
		var s scanner
		s.init(strings.NewReader(test.src), func(_, _ uint, msg string) {
			errs = append(errs, msg)
		}, logosScannerMode)

		for i, want := range test.tokens {
			s.next()
			if s.tok != want {
				t.Errorf("%q token %d: got %s; want %s", test.src, i, s.tok, want)
			}
			if want == _Semi && s.lit != "newline" {
				t.Errorf("%q token %d: got semi %q; want %q", test.src, i, s.lit, "newline")
			}
		}
		if len(errs) != 0 {
			t.Errorf("%q: got errors %q; want none", test.src, errs)
		}
	}
}

// TestPlainQuestionRejected pins plain-mode behavior: `?` still reports
// exactly `invalid character U+003F '?'` at the position of `?`
// (M1-S1 Attack; scanner.go default case). Regression pin: must keep
// passing after the M1-S1 build.
func TestPlainQuestionRejected(t *testing.T) {
	const want = "invalid character U+003F '?'"
	for _, test := range []struct {
		src       string
		line, col uint // 0-based, relative to linebase/colbase
	}{
		{"?", 0, 0},
		{"x?", 0, 1},
		{"f()?", 0, 3},
	} {
		var s scanner
		var line, col uint
		var err string
		s.init(strings.NewReader(test.src), func(l, c uint, msg string) {
			if err == "" {
				line, col = l-linebase, c-colbase
				err = msg
			}
		}, 0)

		for {
			s.next()
			if s.tok == _EOF {
				break
			}
		}

		if err != want {
			t.Errorf("%q: got err = %q; want %q", test.src, err, want)
		}
		if line != test.line {
			t.Errorf("%q: got line = %d; want %d", test.src, line, test.line)
		}
		if col != test.col {
			t.Errorf("%q: got col = %d; want %d", test.src, col, test.col)
		}
	}
}

// TestQuestionString checks that _Question String() is "?" (M1-S1 Build:
// token_string.go regeneration appends `?` for the `// ?` linecomment).
func TestQuestionString(t *testing.T) {
	if got := _Question.String(); got != "?" {
		t.Errorf("got _Question.String() = %q; want %q", got, "?")
	}
}

// TestKeywordRangeUnchanged spot-checks that the keyword range
// _Break.._Var is unchanged: `break` and `var` still scan as keywords
// in plain and Logos modes (M1-S1 Build: _Question appends after _Var).
func TestKeywordRangeUnchanged(t *testing.T) {
	for _, mode := range []uint{0, logosScannerMode} {
		for _, test := range []struct {
			src  string
			want token
		}{
			{"break", _Break},
			{"var", _Var},
		} {
			var errs []string
			var s scanner
			s.init(strings.NewReader(test.src), func(_, _ uint, msg string) {
				errs = append(errs, msg)
			}, mode)

			s.next()
			if s.tok != test.want {
				t.Errorf("mode %d: %q: got %s; want %s", mode, test.src, s.tok, test.want)
			}
			if len(errs) != 0 {
				t.Errorf("mode %d: %q: got errors %q; want none", mode, test.src, errs)
			}
		}
	}
	if !(_Break < _Var && _Var < _Question && _Question < _QuestionDot && _QuestionDot < _Coalesce && _Coalesce < tokenCount) {
		t.Errorf("want _Break < _Var < _Question < _QuestionDot < _Coalesce < tokenCount, got %d < %d < %d < %d < %d < %d",
			_Break, _Var, _Question, _QuestionDot, _Coalesce, tokenCount)
	}
}

// TestTokenCountUnder64 checks the total token count stays under 64 so
// tokens still fit in the uint64 set (M1-S1 Build: tokens.go assert).
func TestTokenCountUnder64(t *testing.T) {
	if tokenCount > 64 {
		t.Errorf("got tokenCount = %d; want <= 64", tokenCount)
	}
}

// TestLogosCoalesceScan checks that in LogosMode `??` with no gap scans
// as one _Coalesce token, while a gap keeps two _Question tokens
// (M3-S2 Attack, SPEC Sec 5.1, T-104.3).
func TestLogosCoalesceScan(t *testing.T) {
	for _, test := range []struct {
		src    string
		tokens []token
	}{
		{"a??b", []token{_Name, _Coalesce, _Name, _Semi, _EOF}},
		{"a ?? b", []token{_Name, _Coalesce, _Name, _Semi, _EOF}},
		{"a? ?b", []token{_Name, _Question, _Question, _Name, _Semi, _EOF}},
		{"a ? ? b", []token{_Name, _Question, _Question, _Name, _Semi, _EOF}},
		{"a?\n?b", []token{_Name, _Question, _Semi, _Question, _Name, _Semi, _EOF}},
	} {
		var errs []string
		var s scanner
		s.init(strings.NewReader(test.src), func(_, _ uint, msg string) {
			errs = append(errs, msg)
		}, logosScannerMode)

		for i, want := range test.tokens {
			s.next()
			if s.tok != want {
				t.Errorf("%q token %d: got %s; want %s", test.src, i, s.tok, want)
			}
			// A `;` mid-stream comes from an operand-final token before a
			// real newline; only the trailing `;` at EOF is literal "EOF".
			if want == _Semi && i == len(test.tokens)-2 && s.lit != "EOF" {
				t.Errorf("%q token %d: got semi %q; want %q", test.src, i, s.lit, "EOF")
			}
		}
		if len(errs) != 0 {
			t.Errorf("%q: got errors %q; want none", test.src, errs)
		}
	}
}

// TestLogosQuestionDotScan checks that in LogosMode `?.` with no gap
// scans as one _QuestionDot token, a gap keeps _Question plus _Dot, and
// `?.5` stays _Question plus a number (M3-S2 Attack, SPEC Sec 5.1).
func TestLogosQuestionDotScan(t *testing.T) {
	for _, test := range []struct {
		src    string
		tokens []token
	}{
		{"a?.b", []token{_Name, _QuestionDot, _Name, _Semi, _EOF}},
		{"a ?. b", []token{_Name, _QuestionDot, _Name, _Semi, _EOF}},
		{"a ? . b", []token{_Name, _Question, _Dot, _Name, _Semi, _EOF}},
		{"a?\t.b", []token{_Name, _Question, _Dot, _Name, _Semi, _EOF}},
		{"a?\n.b", []token{_Name, _Question, _Semi, _Dot, _Name, _Semi, _EOF}},
		{"a?.5", []token{_Name, _Question, _Literal, _Semi, _EOF}},
	} {
		var errs []string
		var s scanner
		s.init(strings.NewReader(test.src), func(_, _ uint, msg string) {
			errs = append(errs, msg)
		}, logosScannerMode)

		for i, want := range test.tokens {
			s.next()
			if s.tok != want {
				t.Errorf("%q token %d: got %s; want %s", test.src, i, s.tok, want)
			}
			// Only the trailing `;` at EOF is literal "EOF"; a `;` mid-
			// stream after a newline-final operand stays literal "newline".
			if want == _Semi && i == len(test.tokens)-2 && s.lit != "EOF" {
				t.Errorf("%q token %d: got semi %q; want %q", test.src, i, s.lit, "EOF")
			}
		}
		if len(errs) != 0 {
			t.Errorf("%q: got errors %q; want none", test.src, errs)
		}
	}
}

// TestLogosOptionNlsemi checks that neither `?.` nor `??` ends an
// operand: `a?.\n b` and `a ??\n b` scan with no `;`, while `x?\n y`
// keeps the M1-S1 `;` after `?` (M3-S2 Attack, nlsemi contract).
func TestLogosOptionNlsemi(t *testing.T) {
	for _, test := range []struct {
		src    string
		tokens []token
	}{
		{"a?.\n b", []token{_Name, _QuestionDot, _Name, _Semi, _EOF}},
		{"a ??\n b", []token{_Name, _Coalesce, _Name, _Semi, _EOF}},
		{"x?\n y", []token{_Name, _Question, _Semi, _Name, _Semi, _EOF}},
	} {
		var errs []string
		var s scanner
		s.init(strings.NewReader(test.src), func(_, _ uint, msg string) {
			errs = append(errs, msg)
		}, logosScannerMode)

		for i, want := range test.tokens {
			s.next()
			if s.tok != want {
				t.Errorf("%q token %d: got %s; want %s", test.src, i, s.tok, want)
			}
			// A mid-input `;` comes from a newline; the trailing
			// one (second to last, before EOF) comes from EOF.
			// Either way no `;` follows `?.` or `??` itself.
			if want == _Semi {
				wantLit := "newline"
				if i == len(test.tokens)-2 {
					wantLit = "EOF"
				}
				if s.lit != wantLit {
					t.Errorf("%q token %d: got semi %q; want %q", test.src, i, s.lit, wantLit)
				}
			}
		}
		if len(errs) != 0 {
			t.Errorf("%q: got errors %q; want none", test.src, errs)
		}
	}
}

// TestOptionTokenStrings checks that _QuestionDot String() is "?." and
// _Coalesce String() is "??" (M3-S2 Build: token_string.go regeneration
// appends `?.` and `??` for the `// ?.` and `// ??` linecomments).
func TestOptionTokenStrings(t *testing.T) {
	if got := _QuestionDot.String(); got != "?." {
		t.Errorf("got _QuestionDot.String() = %q; want %q", got, "?.")
	}
	if got := _Coalesce.String(); got != "??" {
		t.Errorf("got _Coalesce.String() = %q; want %q", got, "??")
	}
}

// TestPlainOptionTailsRejected pins plain-mode behavior: `?.` and `??`
// still report exactly `invalid character U+003F '?'` at the position of
// the first `?`, like a lone `?` (M3-S2 Build; scanner.go default case).
func TestPlainOptionTailsRejected(t *testing.T) {
	const want = "invalid character U+003F '?'"
	for _, test := range []struct {
		src       string
		line, col uint // 0-based, relative to linebase/colbase
	}{
		{"a?.b", 0, 1},
		{"a??b", 0, 1},
	} {
		var s scanner
		var line, col uint
		var err string
		s.init(strings.NewReader(test.src), func(l, c uint, msg string) {
			if err == "" {
				line, col = l-linebase, c-colbase
				err = msg
			}
		}, 0)

		for {
			s.next()
			if s.tok == _EOF {
				break
			}
		}

		if err != want {
			t.Errorf("%q: got err = %q; want %q", test.src, err, want)
		}
		if line != test.line {
			t.Errorf("%q: got line = %d; want %d", test.src, line, test.line)
		}
		if col != test.col {
			t.Errorf("%q: got col = %d; want %d", test.src, col, test.col)
		}
	}
}
