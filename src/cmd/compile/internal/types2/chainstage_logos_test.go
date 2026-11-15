// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	. "cmd/compile/internal/types2"
	"fmt"
	"strings"
	"testing"
)

type stageImporter map[string]*Package

func (m stageImporter) Import(path string) (*Package, error) {
	if pkg, ok := m[path]; ok {
		return pkg, nil
	}
	return nil, fmt.Errorf("unknown import %q", path)
}

func stageContextPackage() *Package {
	pkg := NewPackage("context", "context")
	name := NewTypeName(syntax.Pos{}, pkg, "Context", nil)
	NewNamed(name, NewInterfaceType(nil, nil), nil)
	pkg.Scope().Insert(name)
	pkg.MarkComplete()
	return pkg
}

type stageCheck struct {
	pkg    *Package
	info   *Info
	file   *syntax.File
	errors []string
}

func checkStageSource(t *testing.T, src string) *stageCheck {
	t.Helper()
	file, err := syntax.Parse(syntax.NewFileBase("stage.lgo"), strings.NewReader(src), nil, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	res := &stageCheck{file: file, info: &Info{Types: make(map[syntax.Expr]TypeAndValue)}}
	conf := &Config{
		Importer: stageImporter{"context": stageContextPackage()},
		Logos:    true,
		Error:    func(err error) { res.errors = append(res.errors, err.Error()) },
	}
	res.pkg, _ = conf.Check("p", []*syntax.File{file}, res.info)
	return res
}

func (c *stageCheck) callsNamed(name string) []*syntax.CallExpr {
	var calls []*syntax.CallExpr
	syntax.Inspect(c.file, func(n syntax.Node) bool {
		if call, ok := n.(*syntax.CallExpr); ok {
			if sel, ok := call.Fun.(*syntax.SelectorExpr); ok && sel.Sel.Value == name {
				calls = append(calls, call)
			}
		}
		return true
	})
	return calls
}

func (c *stageCheck) stage(t *testing.T, name string) (Type, bool) {
	t.Helper()
	calls := c.callsNamed(name)
	if len(calls) != 1 {
		t.Fatalf("source has %d %s calls; want 1", len(calls), name)
	}
	return LogosGroupStage(c.pkg, c.info, calls[0])
}

func stageProgram(decls, recv string) string {
	return "package p\n\nimport \"context\"\n\n" + decls + "\nfunc f(ctx context.Context, r " + recv + ") {\n\t_, _ = r.group(ctx, limit = 2).unordered().map(x => x)\n}\n"
}

func TestLogosGroupStageReceiverKinds(t *testing.T) {
	for _, tc := range []struct{ name, decls, recv string }{
		{"slice", "", "[]int"},
		{"array", "", "[3]int"},
		{"sequence", "", "func(yield func(int) bool)"},
		{"receive channel", "", "<-chan int"},
		{"bidirectional channel", "", "chan int"},
		{"named slice", "type T []int\n", "T"},
		{"named array", "type T [3]int\n", "T"},
		{"named sequence", "type T func(yield func(int) bool)\n", "T"},
		{"named receive channel", "type T <-chan int\n", "T"},
		{"named bidirectional channel", "type T chan int\n", "T"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := checkStageSource(t, stageProgram(tc.decls, tc.recv))
			if len(c.errors) != 0 {
				t.Fatalf("type errors: %q", c.errors)
			}
			for _, name := range []string{"group", "unordered"} {
				elem, ok := c.stage(t, name)
				if !ok || !Identical(elem, Typ[Int]) {
					t.Errorf("%s: elem = %v, ok = %v; want int, true", name, elem, ok)
				}
			}
		})
	}
}

func TestLogosGroupStageRejectsReceivers(t *testing.T) {
	for _, tc := range []struct{ name, decls, recv string }{
		{"map", "", "map[string]int"},
		{"send channel", "", "chan<- int"},
		{"integer", "", "int"},
		{"string", "", "string"},
		{"struct", "type T struct{}\n", "T"},
		{"function without yield", "", "func(int) bool"},
		{"function with yield result", "", "func(yield func(int) bool) bool"},
		{"yield with two values", "", "func(yield func(int, int) bool)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := checkStageSource(t, stageProgram(tc.decls, tc.recv))
			joined := strings.Join(c.errors, "\n")
			if !strings.Contains(joined, "group undefined") {
				t.Errorf("errors %q do not carry the undefined method", c.errors)
			}
			if strings.Contains(joined, "[correctness::named_argument]") {
				t.Errorf("errors %q carry correctness::named_argument for limit = N", c.errors)
			}
			if elem, ok := c.stage(t, "group"); ok {
				t.Errorf("group accepted the receiver with elem %v", elem)
			}
			if elem, ok := c.stage(t, "unordered"); ok {
				t.Errorf("unordered accepted after a rejected group with elem %v", elem)
			}
		})
	}
}

func TestLogosGroupStageKeepsGoMethods(t *testing.T) {
	for _, tc := range []struct{ name, decls, recv string }{
		{"struct", "type T struct{}\n\nfunc (T) group(ctx context.Context, n int) int { return n }\n", "T"},
		{"slice", "type T []int\n\nfunc (T) group(ctx context.Context, n int) int { return n }\n", "T"},
		{"channel", "type T chan int\n\nfunc (T) group(ctx context.Context, n int) int { return n }\n", "T"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := "package p\n\nimport \"context\"\n\n" + tc.decls + "\nfunc f(ctx context.Context, r " + tc.recv + ") {\n\t_ = r.group(ctx, 2)\n}\n"
			c := checkStageSource(t, src)
			if len(c.errors) != 0 {
				t.Fatalf("type errors: %q", c.errors)
			}
			if elem, ok := c.stage(t, "group"); ok {
				t.Errorf("group call of a method was a stage with elem %v", elem)
			}
		})
	}
}

func TestLogosGroupStageUnorderedFollowsGroupOnly(t *testing.T) {
	src := "package p\n\nimport \"context\"\n\nfunc f(ctx context.Context, xs []int) {\n\t_, _ = xs.group(ctx, limit = 2).map(x => x).unordered()\n}\n"
	c := checkStageSource(t, src)
	if elem, ok := c.stage(t, "unordered"); ok {
		t.Errorf("unordered after map was a stage with elem %v", elem)
	}
}
