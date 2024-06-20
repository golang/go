// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constraint

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

var exprStringTests = []struct {
	x   Expr
	out string
}{
	{
		x:   tag("abc"),
		out: "abc",
	},
	{
		x:   not(tag("abc")),
		out: "!abc",
	},
	{
		x:   not(and(tag("abc"), tag("def"))),
		out: "!(abc && def)",
	},
	{
		x:   and(tag("abc"), or(tag("def"), tag("ghi"))),
		out: "abc && (def || ghi)",
	},
	{
		x:   or(and(tag("abc"), tag("def")), tag("ghi")),
		out: "(abc && def) || ghi",
	},
}

func TestExprString(t *testing.T) {
	for i, tt := range exprStringTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			s := tt.x.String()
			if s != tt.out {
				t.Errorf("String() mismatch:\nhave %s\nwant %s", s, tt.out)
			}
		})
	}
}

var lexTests = []struct {
	in  string
	out string
}{
	{"", ""},
	{"x", "x"},
	{"x.y", "x.y"},
	{"x_y", "x_y"},
	{"αx", "αx"},
	{"αx²", "αx err: invalid syntax at ²"},
	{"go1.2", "go1.2"},
	{"x y", "x y"},
	{"x!y", "x ! y"},
	{"&&||!()xy yx ", "&& || ! ( ) xy yx"},
	{"x~", "x err: invalid syntax at ~"},
	{"x ~", "x err: invalid syntax at ~"},
	{"x &", "x err: invalid syntax at &"},
	{"x &y", "x err: invalid syntax at &"},
}

func TestLex(t *testing.T) {
	for i, tt := range lexTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			p := &exprParser{s: tt.in}
			out := ""
			for {
				tok, err := lexHelp(p)
				if tok == "" && err == nil {
					break
				}
				if out != "" {
					out += " "
				}
				if err != nil {
					out += "err: " + err.Error()
					break
				}
				out += tok
			}
			if out != tt.out {
				t.Errorf("lex(%q):\nhave %s\nwant %s", tt.in, out, tt.out)
			}
		})
	}
}

func lexHelp(p *exprParser) (tok string, err error) {
	defer func() {
		if e := recover(); e != nil {
			if e, ok := e.(*SyntaxError); ok {
				err = e
				return
			}
			panic(e)
		}
	}()

	p.lex()
	return p.tok, nil
}

var parseExprTests = []struct {
	in string
	x  Expr
}{
	{"x", tag("x")},
	{"x&&y", and(tag("x"), tag("y"))},
	{"x||y", or(tag("x"), tag("y"))},
	{"(x)", tag("x")},
	{"x||y&&z", or(tag("x"), and(tag("y"), tag("z")))},
	{"x&&y||z", or(and(tag("x"), tag("y")), tag("z"))},
	{"x&&(y||z)", and(tag("x"), or(tag("y"), tag("z")))},
	{"(x||y)&&z", and(or(tag("x"), tag("y")), tag("z"))},
	{"!(x&&y)", not(and(tag("x"), tag("y")))},
}

func TestParseExpr(t *testing.T) {
	for i, tt := range parseExprTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			x, err := parseExpr(tt.in)
			if err != nil {
				t.Fatal(err)
			}
			if x.String() != tt.x.String() {
				t.Errorf("parseExpr(%q):\nhave %s\nwant %s", tt.in, x, tt.x)
			}
		})
	}
}

var parseExprErrorTests = []struct {
	in  string
	err error
}{
	{"x && ", &SyntaxError{Offset: 5, Err: "unexpected end of expression"}},
	{"x && (", &SyntaxError{Offset: 6, Err: "missing close paren"}},
	{"x && ||", &SyntaxError{Offset: 5, Err: "unexpected token ||"}},
	{"x && !", &SyntaxError{Offset: 6, Err: "unexpected end of expression"}},
	{"x && !!", &SyntaxError{Offset: 6, Err: "double negation not allowed"}},
	{"x !", &SyntaxError{Offset: 2, Err: "unexpected token !"}},
	{"x && (y", &SyntaxError{Offset: 5, Err: "missing close paren"}},
}

func TestParseError(t *testing.T) {
	for i, tt := range parseExprErrorTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			x, err := parseExpr(tt.in)
			if err == nil {
				t.Fatalf("parseExpr(%q) = %v, want error", tt.in, x)
			}
			if !reflect.DeepEqual(err, tt.err) {
				t.Fatalf("parseExpr(%q): wrong error:\nhave %#v\nwant %#v", tt.in, err, tt.err)
			}
		})
	}
}

var exprEvalTests = []struct {
	in   string
	ok   bool
	tags string
}{
	{"x", false, "x"},
	{"x && y", false, "x y"},
	{"x || y", false, "x y"},
	{"!x && yes", true, "x yes"},
	{"yes || y", true, "y yes"},
}

func TestExprEval(t *testing.T) {
	for i, tt := range exprEvalTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			x, err := parseExpr(tt.in)
			if err != nil {
				t.Fatal(err)
			}
			tags := make(map[string]bool)
			wantTags := make(map[string]bool)
			for _, tag := range strings.Fields(tt.tags) {
				wantTags[tag] = true
			}
			hasTag := func(tag string) bool {
				tags[tag] = true
				return tag == "yes"
			}
			ok := x.Eval(hasTag)
			if ok != tt.ok || !reflect.DeepEqual(tags, wantTags) {
				t.Errorf("Eval(%#q):\nhave ok=%v, tags=%v\nwant ok=%v, tags=%v",
					tt.in, ok, tags, tt.ok, wantTags)
			}
		})
	}
}

var parsePlusBuildExprTests = []struct {
	in string
	x  Expr
}{
	{"x", tag("x")},
	{"x,y", and(tag("x"), tag("y"))},
	{"x y", or(tag("x"), tag("y"))},
	{"x y,z", or(tag("x"), and(tag("y"), tag("z")))},
	{"x,y z", or(and(tag("x"), tag("y")), tag("z"))},
	{"x,!y !z", or(and(tag("x"), not(tag("y"))), not(tag("z")))},
	{"!! x", or(tag("ignore"), tag("x"))},
	{"!!x", tag("ignore")},
	{"!x", not(tag("x"))},
	{"!", tag("ignore")},
	{"", tag("ignore")},
}

func TestParsePlusBuildExpr(t *testing.T) {
	for i, tt := range parsePlusBuildExprTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			x, _ := parsePlusBuildExpr(tt.in)
			if x.String() != tt.x.String() {
				t.Errorf("parsePlusBuildExpr(%q):\nhave %v\nwant %v", tt.in, x, tt.x)
			}
		})
	}
}

var constraintTests = []struct {
	in  string
	x   Expr
	err string
}{
	{"//+build !", tag("ignore"), ""},
	{"//+build", tag("ignore"), ""},
	{"//+build x y", or(tag("x"), tag("y")), ""},
	{"// +build x y \n", or(tag("x"), tag("y")), ""},
	{"// +build x y \n ", nil, "not a build constraint"},
	{"// +build x y \nmore", nil, "not a build constraint"},
	{" //+build x y", nil, "not a build constraint"},

	{"//go:build x && y", and(tag("x"), tag("y")), ""},
	{"//go:build x && y\n", and(tag("x"), tag("y")), ""},
	{"//go:build x && y\n ", nil, "not a build constraint"},
	{"//go:build x && y\nmore", nil, "not a build constraint"},
	{" //go:build x && y", nil, "not a build constraint"},
	{"//go:build\n", nil, "unexpected end of expression"},
}

func TestParse(t *testing.T) {
	for i, tt := range constraintTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			x, err := Parse(tt.in)
			if err != nil {
				if tt.err == "" {
					t.Errorf("Constraint(%q): unexpected error: %v", tt.in, err)
				} else if !strings.Contains(err.Error(), tt.err) {
					t.Errorf("Constraint(%q): error %v, want %v", tt.in, err, tt.err)
				}
				return
			}
			if tt.err != "" {
				t.Errorf("Constraint(%q) = %v, want error %v", tt.in, x, tt.err)
				return
			}
			if x.String() != tt.x.String() {
				t.Errorf("Constraint(%q):\nhave %v\nwant %v", tt.in, x, tt.x)
			}
		})
	}
}

var plusBuildLinesTests = []struct {
	in  string
	out []string
	err error
}{
	{"x", []string{"x"}, nil},
	{"x && !y", []string{"x,!y"}, nil},
	{"x || y", []string{"x y"}, nil},
	{"x && (y || z)", []string{"x", "y z"}, nil},
	{"!(x && y)", []string{"!x !y"}, nil},
	{"x || (y && z)", []string{"x y,z"}, nil},
	{"w && (x || (y && z))", []string{"w", "x y,z"}, nil},
	{"v || (w && (x || (y && z)))", nil, errComplex},
}

func TestPlusBuildLines(t *testing.T) {
	for i, tt := range plusBuildLinesTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			x, err := parseExpr(tt.in)
			if err != nil {
				t.Fatal(err)
			}
			lines, err := PlusBuildLines(x)
			if err != nil {
				if tt.err == nil {
					t.Errorf("PlusBuildLines(%q): unexpected error: %v", tt.in, err)
				} else if tt.err != err {
					t.Errorf("PlusBuildLines(%q): error %v, want %v", tt.in, err, tt.err)
				}
				return
			}
			if tt.err != nil {
				t.Errorf("PlusBuildLines(%q) = %v, want error %v", tt.in, lines, tt.err)
				return
			}
			var want []string
			for _, line := range tt.out {
				want = append(want, "// +build "+line)
			}
			if !reflect.DeepEqual(lines, want) {
				t.Errorf("PlusBuildLines(%q):\nhave %q\nwant %q", tt.in, lines, want)
			}
		})
	}
}

func TestSizeLimits(t *testing.T) {
	for _, tc := range []struct {
		name string
		expr string
	}{
		{
			name: "go:build or limit",
			expr: "//go:build " + strings.Repeat("a || ", maxSize+2),
		},
		{
			name: "go:build and limit",
			expr: "//go:build " + strings.Repeat("a && ", maxSize+2),
		},
		{
			name: "go:build and depth limit",
			expr: "//go:build " + strings.Repeat("(a &&", maxSize+2),
		},
		{
			name: "go:build or depth limit",
			expr: "//go:build " + strings.Repeat("(a ||", maxSize+2),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse(tc.expr)
			if err == nil {
				t.Error("expression did not trigger limit")
			} else if syntaxErr, ok := err.(*SyntaxError); !ok || syntaxErr.Err != "build expression too large" {
				if !ok {
					t.Errorf("unexpected error: %v", err)
				} else {
					t.Errorf("unexpected syntax error: %s", syntaxErr.Err)
				}
			}
		})
	}
}

func TestPlusSizeLimits(t *testing.T) {
	maxOldSize := 100
	for _, tc := range []struct {
		name string
		expr string
	}{
		{
			name: "+build or limit",
			expr: "// +build " + strings.Repeat("a ", maxOldSize+2),
		},
		{
			name: "+build and limit",
			expr: "// +build " + strings.Repeat("a,", maxOldSize+2),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse(tc.expr)
			if err == nil {
				t.Error("expression did not trigger limit")
			} else if err != errComplex {
				t.Errorf("unexpected error: got %q, want %q", err, errComplex)
			}
		})
	}
}
