package ssa_test

// This file defines tests of source-level debugging utilities.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"testing"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
)

func TestObjValueLookup(t *testing.T) {
	imp := importer.New(new(importer.Context)) // (uses GCImporter)
	f, err := parser.ParseFile(imp.Fset, "testdata/objlookup.go", nil, parser.DeclarationErrors|parser.ParseComments)
	if err != nil {
		t.Errorf("parse error: %s", err)
		return
	}

	// Maps each var Ident (represented "name:linenum") to the
	// kind of ssa.Value we expect (represented "Constant", "&Alloc").
	expectations := make(map[string]string)

	// Find all annotations of form x::BinOp, &y::Alloc, etc.
	re := regexp.MustCompile(`(\b|&)?(\w*)::(\w*)\b`)
	for _, c := range f.Comments {
		text := c.Text()
		pos := imp.Fset.Position(c.Pos())
		for _, m := range re.FindAllStringSubmatch(text, -1) {
			key := fmt.Sprintf("%s:%d", m[2], pos.Line)
			value := m[1] + m[3]
			expectations[key] = value
		}
	}

	info, err := imp.CreateSourcePackage("main", []*ast.File{f})
	if err != nil {
		t.Error(err.Error())
		return
	}

	prog := ssa.NewProgram(imp.Fset, ssa.DebugInfo /*|ssa.LogFunctions*/)
	prog.CreatePackages(imp)
	pkg := prog.Package(info.Pkg)
	pkg.Build()

	// Gather all idents and objects in file.
	objs := make(map[types.Object]bool)
	var ids []*ast.Ident
	ast.Inspect(f, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			ids = append(ids, id)
			if obj := info.ObjectOf(id); obj != nil {
				objs[obj] = true
			}
		}
		return true
	})

	// Check invariants for func and const objects.
	for obj := range objs {
		switch obj := obj.(type) {
		case *types.Func:
			if obj.Name() == "interfaceMethod" {
				continue // TODO(adonovan): not yet implemented.
			}
			checkFuncValue(t, prog, obj)

		case *types.Const:
			checkConstValue(t, prog, obj)
		}
	}

	// Check invariants for var objects.
	// The result varies based on the specific Ident.
	for _, id := range ids {
		if obj, ok := info.ObjectOf(id).(*types.Var); ok {
			ref, _ := importer.PathEnclosingInterval(f, id.Pos(), id.Pos())
			pos := imp.Fset.Position(id.Pos())
			exp := expectations[fmt.Sprintf("%s:%d", id.Name, pos.Line)]
			if exp == "" {
				t.Errorf("%s: no expectation for var ident %s ", pos, id.Name)
				continue
			}
			wantAddr := false
			if exp[0] == '&' {
				wantAddr = true
				exp = exp[1:]
			}
			checkVarValue(t, prog, ref, obj, exp, wantAddr)
		}
	}
}

func checkFuncValue(t *testing.T, prog *ssa.Program, obj *types.Func) {
	v := prog.FuncValue(obj)
	// fmt.Printf("FuncValue(%s) = %s\n", obj, v) // debugging
	if v == nil {
		t.Errorf("FuncValue(%s) == nil", obj)
		return
	}
	// v must be an *ssa.Function or *ssa.Builtin.
	v2, _ := v.(interface {
		Object() types.Object
	})
	if v2 == nil {
		t.Errorf("FuncValue(%s) = %s %T; has no Object() method",
			obj, v.Name(), v)
		return
	}
	if vobj := v2.Object(); vobj != obj {
		t.Errorf("FuncValue(%s).Object() == %s; value was %s",
			obj, vobj, v.Name())
		return
	}
	if !types.IsIdentical(v.Type(), obj.Type()) {
		t.Errorf("FuncValue(%s).Type() == %s", obj, v.Type())
		return
	}
}

func checkConstValue(t *testing.T, prog *ssa.Program, obj *types.Const) {
	c := prog.ConstValue(obj)
	// fmt.Printf("ConstValue(%s) = %s\n", obj, c) // debugging
	if c == nil {
		t.Errorf("ConstValue(%s) == nil", obj)
		return
	}
	if !types.IsIdentical(c.Type(), obj.Type()) {
		t.Errorf("ConstValue(%s).Type() == %s", obj, c.Type())
		return
	}
	if obj.Name() != "nil" {
		if !exact.Compare(c.Value, token.EQL, obj.Val()) {
			t.Errorf("ConstValue(%s).Value (%s) != %s",
				obj, c.Value, obj.Val())
			return
		}
	}
}

func checkVarValue(t *testing.T, prog *ssa.Program, ref []ast.Node, obj *types.Var, expKind string, wantAddr bool) {
	// The prefix of all assertions messages.
	prefix := fmt.Sprintf("VarValue(%s @ L%d)",
		obj, prog.Fset.Position(ref[0].Pos()).Line)

	v := prog.VarValue(obj, ref)

	// Kind is the concrete type of the ssa Value.
	gotKind := "nil"
	if v != nil {
		gotKind = fmt.Sprintf("%T", v)[len("*ssa."):]
	}

	// fmt.Printf("%s = %v (kind %q; expect %q) addr=%t\n", prefix, v, gotKind, expKind, wantAddr) // debugging

	// Check the kinds match.
	// "nil" indicates expected failure (e.g. optimized away).
	if expKind != gotKind {
		t.Errorf("%s concrete type == %s, want %s", prefix, gotKind, expKind)
	}

	// Check the types match.
	// If wantAddr, the expected type is the object's address.
	if v != nil {
		expType := obj.Type()
		if wantAddr {
			expType = types.NewPointer(expType)
		}
		if !types.IsIdentical(v.Type(), expType) {
			t.Errorf("%s.Type() == %s, want %s", prefix, v.Type(), expType)
		}
	}
}
