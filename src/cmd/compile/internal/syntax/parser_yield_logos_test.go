// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

func firstYield(t *testing.T, f *File) *YieldStmt {
	t.Helper()
	var found *YieldStmt
	Inspect(f, func(n Node) bool {
		if y, ok := n.(*YieldStmt); ok && found == nil {
			found = y
		}
		return true
	})
	if found == nil {
		t.Fatal("no yield statement")
	}
	return found
}

func yieldShape(e Expr) string {
	switch e := e.(type) {
	case *Name:
		return e.Value
	case *BasicLit:
		return e.Value
	case *ParenExpr:
		return "P(" + yieldShape(e.X) + ")"
	case *Operation:
		if e.Y == nil {
			return "(" + e.Op.String() + " " + yieldShape(e.X) + ")"
		}
		return "(" + e.Op.String() + " " + yieldShape(e.X) + " " + yieldShape(e.Y) + ")"
	case *ListExpr:
		shapes := make([]string, len(e.ElemList))
		for i, elem := range e.ElemList {
			shapes[i] = yieldShape(elem)
		}
		return strings.Join(shapes, ",")
	case *SelectorExpr:
		return "sel(" + yieldShape(e.X) + "." + e.Sel.Value + ")"
	case *IndexExpr:
		return "idx(" + yieldShape(e.X) + ";" + yieldShape(e.Index) + ")"
	case *CallExpr:
		args := make([]string, len(e.ArgList))
		for i, arg := range e.ArgList {
			args[i] = yieldShape(arg)
		}
		return "call(" + yieldShape(e.Fun) + ";" + strings.Join(args, ",") + ")"
	}
	return "?"
}

func TestLogosYieldStartsAnywhere(t *testing.T) {
	starts := []string{
		"yield 1",
		"yield x",
		"yield func() {}",
		"yield -x",
		"yield +x",
		"yield *p",
		"yield &v",
		"yield ^x",
		"yield !ok",
		"yield const []byte(s)",
		"yield switch x { default => 1 }",
		"yield map[int]int{}[0]",
		"yield struct{}{}",
		"yield []int{1}[0]",
		"yield [1]int{1}[0]",
		"yield [...]int{1}[0]",
		"yield [2][2]int{{1, 2}, {3, 4}}[1][0]",
		"yield [2]*int{}[0]",
		"yield []*int{}[0]",
		"yield x, y",
		"yield (1) + x",
		"yield (a + b) * 2",
		"yield -a + b",
		"yield (a).f",
		"yield (a)[0]",
		"yield (a).(int)",
		"yield (f)(x) + 1",
		"yield -a, b",
		"yield (a), b",
		"yield *p, -b",
	}
	for _, stmt := range starts {
		t.Run(stmt, func(t *testing.T) {
			src := "package p\nfunc f() {\n" + stmt + "\n}\n"
			if got := countNodes[*YieldStmt](parseLogos(t, src)); got != 1 {
				t.Fatalf("%s: found %d yield statements, want 1", stmt, got)
			}
		})
	}
}

func TestLogosYieldOperandGrouping(t *testing.T) {
	tests := []struct {
		stmt string
		want string
	}{
		{"yield -a * 2", "(* (- a) 2)"},
		{"yield -a + b", "(+ (- a) b)"},
		{"yield -a * b + c", "(+ (* (- a) b) c)"},
		{"yield (a + b) * 2", "(* P((+ a b)) 2)"},
		{"yield *p + 1", "(+ (* p) 1)"},
		{"yield &x == nil", "(== (& x) nil)"},
		{"yield -(a) * 2", "(* (- P(a)) 2)"},
		{"yield !ok", "(! ok)"},
		{"yield (a).n", "sel(P(a).n)"},
		{"yield (xs)[1]", "idx(P(xs);1)"},
		{"yield (fn)(a) + 1", "(+ call(P(fn);a) 1)"},
		{"yield (a), b", "P(a),b"},
		{"yield -a, b", "(- a),b"},
		{"yield *p, -b", "(* p),(- b)"},
	}
	for _, test := range tests {
		t.Run(test.stmt, func(t *testing.T) {
			src := "package p\nfunc f() {\n" + test.stmt + "\n}\n"
			y := firstYield(t, parseLogos(t, src))
			if got := yieldShape(y.Value); got != test.want {
				t.Fatalf("%s: operand %s, want %s", test.stmt, got, test.want)
			}
		})
	}
}

func TestLogosYieldKeepsGoStatements(t *testing.T) {
	forms := []string{
		"yield (1)",
		"yield (fn)(a)",
		"yield(2)",
		"yield (a).f()",
		"yield <-ch",
		"yield <- v",
		"yield <-ch + 1",
		"yield[0] = 1",
		"yield[0]++",
		"yield[0]()",
		"yield[T](x)",
		"yield[int, string](x)",
		"yield[0][1] = 2",
		"yield.f()",
		"yield.f = 1",
		"yield := 1",
		"yield = 1",
		"yield++",
		"yield--",
		"yield += 1",
		"yield, x = 1, 2",
		"yield, x := 1, 2",
		"yield",
		"yield:\nfor {\nbreak yield\n}",
	}
	for _, stmt := range forms {
		t.Run(stmt, func(t *testing.T) {
			src := "package p\nfunc f() {\n" + stmt + "\n}\n"
			if got := countNodes[*YieldStmt](parseLogos(t, src)); got != 0 {
				t.Fatalf("%s: found %d yield statements, want 0", stmt, got)
			}
		})
	}
}
