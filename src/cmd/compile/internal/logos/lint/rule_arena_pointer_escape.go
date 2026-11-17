// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

var arenaPointerEscape = Rule{
	ID:       "correctness::arena_pointer_escape",
	Group:    Correctness,
	Default:  Deny,
	Go:       true,
	LintOnly: true,
	Check:    checkUnsafePointer,
}

func checkUnsafePointer(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.CallExpr:
				if convertsType(n) && hasBasicKind(TypeOf(n), types2.UnsafePointer) &&
					hasBasicKind(TypeOf(n.ArgList[0]), types2.Uintptr) && !safeUintptr(pass, n.ArgList[0]) {
					pass.Report(Diagnostic{
						Pos:     syntax.StartPos(n),
						End:     pass.CallEnd(f, n),
						Message: "possible misuse of unsafe.Pointer: uintptr does not keep its referent alive; convert directly in the same expression",
					})
				}
			case *syntax.Operation:
				if n.Y != nil {
					break
				}
				var typ types2.Type
				switch n.Op {
				case syntax.Mul:
					typ = TypeOf(n)
				case syntax.And:
					typ = TypeOf(n.X)
				}
				if name := reflectTypeName(typ); name == "SliceHeader" || name == "StringHeader" {
					pass.Reportf(n, "possible misuse of reflect.%s value; keep a pointer to the original slice or string", name)
				}
			}
			return true
		})
	}
}

func safeUintptr(pass *Pass, expr syntax.Expr) bool {
	switch x := syntax.Unparen(expr).(type) {
	case *syntax.SelectorExpr:
		if x.Sel.Value == "Data" {
			if ptr, ok := underOrNil(TypeOf(x.X)).(*types2.Pointer); ok {
				name := reflectTypeName(ptr.Elem())
				if name == "SliceHeader" || name == "StringHeader" {
					return true
				}
			}
		}
	case *syntax.CallExpr:
		fn := calleeFunc(pass, x)
		if fn != nil && fn.Pkg() != nil && fn.Pkg().Path() == "reflect" &&
			(fn.Name() == "Pointer" || fn.Name() == "UnsafeAddr") {
			sig := fn.Type().(*types2.Signature)
			if recv := sig.Recv(); recv != nil && reflectTypeName(recv.Type()) == "Value" {
				return true
			}
		}
	}
	return pointerArithmetic(expr) == arithmeticPointer
}

type arithmeticKind uint8

const (
	arithmeticScalar arithmeticKind = iota
	arithmeticPointer
	arithmeticInvalid
)

func pointerArithmetic(expr syntax.Expr) arithmeticKind {
	switch x := syntax.Unparen(expr).(type) {
	case *syntax.CallExpr:
		if convertsType(x) {
			if hasBasicKind(TypeOf(x), types2.Uintptr) && hasBasicKind(TypeOf(x.ArgList[0]), types2.UnsafePointer) {
				return arithmeticPointer
			}
			if pointerArithmetic(x.ArgList[0]) != arithmeticScalar {
				return arithmeticInvalid
			}
		}
	case *syntax.Operation:
		left := pointerArithmetic(x.X)
		right := arithmeticScalar
		if x.Y != nil {
			right = pointerArithmetic(x.Y)
		}
		if left == arithmeticScalar && right == arithmeticScalar {
			return arithmeticScalar
		}
		if x.Y != nil {
			switch x.Op {
			case syntax.Add:
				if left == arithmeticScalar && right == arithmeticPointer {
					return arithmeticPointer
				}
				fallthrough
			case syntax.Sub, syntax.AndNot:
				if left == arithmeticPointer && right == arithmeticScalar {
					return arithmeticPointer
				}
			}
		}
		return arithmeticInvalid
	}
	return arithmeticScalar
}

func reflectTypeName(typ types2.Type) string {
	named, ok := types2.Unalias(typ).(*types2.Named)
	if !ok || named.Obj().Pkg() == nil || named.Obj().Pkg().Path() != "reflect" {
		return ""
	}
	return named.Obj().Name()
}
