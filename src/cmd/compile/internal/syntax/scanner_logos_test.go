// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	goscan "go/scanner"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	gotoken "go/token"
)

const logosScannerMode = uint(LogosMode)

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

func TestPlainQuestionRejected(t *testing.T) {
	const want = "invalid character U+003F '?'"
	for _, test := range []struct {
		src       string
		line, col uint
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

func TestQuestionString(t *testing.T) {
	if got := _Question.String(); got != "?" {
		t.Errorf("got _Question.String() = %q; want %q", got, "?")
	}
}

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

func TestTokenCountUnder64(t *testing.T) {
	if tokenCount > 64 {
		t.Errorf("got tokenCount = %d; want <= 64", tokenCount)
	}
}

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
			if want == _Semi && i == len(test.tokens)-2 && s.lit != "EOF" {
				t.Errorf("%q token %d: got semi %q; want %q", test.src, i, s.lit, "EOF")
			}
		}
		if len(errs) != 0 {
			t.Errorf("%q: got errors %q; want none", test.src, errs)
		}
	}
}

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
			if want == _Semi && i == len(test.tokens)-2 && s.lit != "EOF" {
				t.Errorf("%q token %d: got semi %q; want %q", test.src, i, s.lit, "EOF")
			}
		}
		if len(errs) != 0 {
			t.Errorf("%q: got errors %q; want none", test.src, errs)
		}
	}
}

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

func TestOptionTokenStrings(t *testing.T) {
	if got := _QuestionDot.String(); got != "?." {
		t.Errorf("got _QuestionDot.String() = %q; want %q", got, "?.")
	}
	if got := _Coalesce.String(); got != "??" {
		t.Errorf("got _Coalesce.String() = %q; want %q", got, "??")
	}
}

func TestPlainOptionTailsRejected(t *testing.T) {
	const want = "invalid character U+003F '?'"
	for _, test := range []struct {
		src       string
		line, col uint
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

func TestLogosNoNewGlobalKeywords(t *testing.T) {
	upstream := []string{
		"break", "case", "chan", "const", "continue",
		"default", "defer", "else", "fallthrough", "for",
		"func", "go", "goto", "if", "import",
		"interface", "map", "package", "range", "return",
		"select", "struct", "switch", "type", "var",
	}
	for _, kw := range upstream {
		if !gotoken.IsKeyword(kw) {
			t.Fatalf("go/token does not list %q as a keyword", kw)
		}
	}
	contextual := []string{"errdefer", "yield", "scope", "group", "limit"}
	for _, w := range contextual {
		if gotoken.IsKeyword(w) {
			t.Fatalf("go/token lists %q as a keyword", w)
		}
	}
	for _, mode := range []uint{0, logosScannerMode} {
		for _, kw := range upstream {
			var errs []string
			var s scanner
			s.init(strings.NewReader(kw), func(_, _ uint, msg string) {
				errs = append(errs, msg)
			}, mode)
			s.next()
			if s.tok == _Name || s.tok.String() != kw {
				t.Errorf("mode %d: %q: got %s; want keyword %q", mode, kw, s.tok, kw)
			}
			if len(errs) != 0 {
				t.Errorf("mode %d: %q: got errors %q; want none", mode, kw, errs)
			}
		}
		for _, w := range contextual {
			var errs []string
			var s scanner
			s.init(strings.NewReader(w), func(_, _ uint, msg string) {
				errs = append(errs, msg)
			}, mode)
			s.next()
			if s.tok != _Name || s.lit != w {
				t.Errorf("mode %d: %q: got %s %q; want _Name %q", mode, w, s.tok, s.lit, w)
			}
			if len(errs) != 0 {
				t.Errorf("mode %d: %q: got errors %q; want none", mode, w, errs)
			}
		}
	}
	for _, w := range contextual {
		for _, mode := range []Mode{CheckBranches, CheckBranches | LogosMode} {
			var msgs []string
			f, _ := Parse(nil, strings.NewReader("package p\nvar v = "+w+"\n"), func(err error) {
				msgs = append(msgs, err.Error())
			}, nil, mode)
			if len(msgs) != 0 {
				t.Errorf("mode %d: var v = %s: got errors %q; want none", uint(mode), w, msgs)
				continue
			}
			if got := countNodes[*YieldStmt](f) + countNodes[*ErrdeferStmt](f) + countNodes[*ScopeDeferStmt](f) + countNodes[*GroupStmt](f); got != 0 {
				t.Errorf("mode %d: var v = %s: found %d keyword nodes, want 0", uint(mode), w, got)
			}
		}
	}
}
func stdlibIdents(seen map[string]struct{}) error {
	root := filepath.Join(runtime.GOROOT(), "src")
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || filepath.Ext(path) != ".go" {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		fset := gotoken.NewFileSet()
		f := fset.AddFile(path, -1, len(data))
		var s goscan.Scanner
		s.Init(f, data, nil, goscan.ScanComments)
		for {
			_, tok, lit := s.Scan()
			if tok == gotoken.EOF {
				return nil
			}
			if tok == gotoken.IDENT {
				seen[lit] = struct{}{}
			}
		}
	})
}
func secondLetterWords(seen map[string]struct{}, a rune) {
	for b := 'a'; b <= 'z'; b++ {
		seen[string([]rune{a, b})] = struct{}{}
	}
}
func thirdLetterWords(seen map[string]struct{}, a, b rune) {
	for c := 'a'; c <= 'z'; c++ {
		seen[string([]rune{a, b, c})] = struct{}{}
	}
}
func shortWords(seen map[string]struct{}) {
	for c := 'a'; c <= 'z'; c++ {
		seen[string(rune(c))] = struct{}{}
	}
	for a := 'a'; a <= 'z'; a++ {
		secondLetterWords(seen, a)
	}
	for a := 'a'; a <= 'z'; a++ {
		for b := 'a'; b <= 'z'; b++ {
			thirdLetterWords(seen, a, b)
		}
	}
	for _, w := range []string{"async", "await", "match", "catch", "finally", "class", "extends", "public", "private", "protected", "static", "final", "throws", "enum", "assert", "lambda", "where", "with", "loop", "repeat", "when", "unless", "until", "become", "mut", "dyn", "union", "trait", "impl", "letm", "fnm"} {
		seen[w] = struct{}{}
	}
}
func scanWordKind(w string, mode uint) (token, string, []string) {
	var errs []string
	var s scanner
	s.init(strings.NewReader(w), func(_, _ uint, msg string) {
		errs = append(errs, msg)
	}, mode)
	s.next()
	return s.tok, s.lit, errs
}
func checkWordKind(t *testing.T, w string) {
	t.Helper()
	plainTok, plainLit, plainErrs := scanWordKind(w, 0)
	logosTok, logosLit, logosErrs := scanWordKind(w, logosScannerMode)
	if len(plainErrs) != 0 || len(logosErrs) != 0 {
		t.Errorf("%q: got errors %q %q; want none", w, plainErrs, logosErrs)
		return
	}
	if plainTok != logosTok {
		t.Errorf("%q: plain %s != logos %s", w, plainTok, logosTok)
		return
	}
	if gotoken.IsKeyword(w) {
		if plainTok == _Name || logosTok == _Name {
			t.Errorf("%q: got _Name; want keyword", w)
		}
		if plainTok.String() != w || logosTok.String() != w {
			t.Errorf("%q: got %s %s; want keyword %q", w, plainTok, logosTok, w)
		}
		return
	}
	if plainTok != _Name || logosTok != _Name {
		t.Errorf("%q: got %s %s; want _Name", w, plainTok, logosTok)
		return
	}
	if plainLit != w || logosLit != w {
		t.Errorf("%q: got %q %q; want %q", w, plainLit, logosLit, w)
	}
}
func TestLogosNoNewGlobalKeywordsProperty(t *testing.T) {
	seen := map[string]struct{}{}
	if err := stdlibIdents(seen); err != nil {
		t.Fatalf("collect stdlib idents: %v", err)
	}
	shortWords(seen)
	for _, w := range slices.Sorted(maps.Keys(seen)) {
		checkWordKind(t, w)
	}
}
