// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// atomicPlainInt64 is portability::atomic_plain_int64 (SPEC Sec 1.2): a
// sync/atomic call whose 64-bit operation takes the address of a plain int64
// or uint64 value. A 32-bit target cannot promise 8-byte alignment for such
// a value, so the call can fault; the autofix replaces the declared type
// with atomic.Int64, atomic.Uint64, which aligns itself. Language neutral,
// and a warning.
var atomicPlainInt64 = Rule{
	ID:    "portability::atomic_plain_int64",
	Group: Portability,
	Go:    true,
	Check: checkAtomicPlainInt64,
}

// atomic64Funcs maps each sync/atomic 64-bit function to the wrapper type
// that replaces a plain value.
var atomic64Funcs = map[string]string{
	"LoadInt64":            "atomic.Int64",
	"StoreInt64":           "atomic.Int64",
	"AddInt64":             "atomic.Int64",
	"SwapInt64":            "atomic.Int64",
	"CompareAndSwapInt64":  "atomic.Int64",
	"AndInt64":             "atomic.Int64",
	"OrInt64":              "atomic.Int64",
	"LoadUint64":           "atomic.Uint64",
	"StoreUint64":          "atomic.Uint64",
	"AddUint64":            "atomic.Uint64",
	"SwapUint64":           "atomic.Uint64",
	"CompareAndSwapUint64": "atomic.Uint64",
	"AndUint64":            "atomic.Uint64",
	"OrUint64":             "atomic.Uint64",
}

func checkAtomicPlainInt64(pass *Pass) {
	declTypes := declarationTypes(pass)
	atomicObjs := logosAtomicObjects(pass)
	reported := make(map[types2.Object]bool)
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			call, ok := n.(*syntax.CallExpr)
			if !ok || len(call.ArgList) == 0 {
				return true
			}
			wrapper, fnName, ok := atomic64Call(pass, call)
			if !ok {
				return true
			}
			obj := plainInt64Address(pass, call.ArgList[0])
			if obj == nil || reported[obj] {
				return true
			}
			if atomicObjs[obj] {
				return true
			}
			typeExpr, ok := declTypes[obj]
			if !ok {
				return true
			}
			reported[obj] = true
			pass.Report(Diagnostic{
				Pos: typeExpr.Pos(),
				End: syntax.EndPos(typeExpr),
				Message: fmt.Sprintf("sync/atomic %s on the address of the plain %s %s: declare it %s",
					fnName, obj.Type().String(), obj.Name(), wrapper),
				Fixes: []Fix{{
					Description: "declare " + obj.Name() + " as " + wrapper,
					Replacement: wrapper,
				}},
			})
			return true
		})
	}
}

func logosAtomicObjects(pass *Pass) map[types2.Object]bool {
	out := make(map[types2.Object]bool)
	dirs := scanDirectives(pass.in)
	for _, a := range dirs.Atomics {
		out[a.Object] = true
	}
	return out
}

// declarationTypes maps each variable and field object to the type
// expression of its declaration.
func declarationTypes(pass *Pass) map[types2.Object]syntax.Expr {
	out := make(map[types2.Object]syntax.Expr)
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch d := n.(type) {
			case *syntax.VarDecl:
				for _, name := range d.NameList {
					if obj := pass.in.defObject(name); obj != nil {
						if d.Type != nil {
							out[obj] = d.Type
						} else if d.Values != nil {
							out[obj] = d.Values
						}
					}
				}
			case *syntax.Field:
				if d.Name != nil {
					if obj := pass.in.defObject(d.Name); obj != nil {
						out[obj] = d.Type
					}
				}
			}
			return true
		})
	}
	return out
}

// atomic64Call reports whether call is a 64-bit sync/atomic function and
// returns the wrapper type that replaces a plain value and the function
// spelling.
func atomic64Call(pass *Pass, call *syntax.CallExpr) (string, string, bool) {
	if !atomicQualifiedCall(pass, call) {
		return "", "", false
	}
	sel := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	wrapper, ok := atomic64Funcs[sel.Sel.Value]
	if !ok {
		return "", "", false
	}
	return wrapper, "atomic." + sel.Sel.Value, true
}

// plainInt64Address returns the variable whose address arg takes.
func plainInt64Address(pass *Pass, arg syntax.Expr) types2.Object {
	op, ok := syntax.Unparen(arg).(*syntax.Operation)
	if !ok || op.Op != syntax.And || op.Y != nil {
		return nil
	}
	var obj types2.Object
	switch x := syntax.Unparen(op.X).(type) {
	case *syntax.Name:
		obj = pass.in.useObject(x)
	case *syntax.SelectorExpr:
		if pass.in.Info != nil {
			if sel := pass.in.Info.Selections[x]; sel != nil && sel.Kind() == types2.FieldVal {
				obj = sel.Obj()
			}
		}
	default:
		return nil
	}
	if obj == nil || !isPlainInt64(obj.Type()) {
		return nil
	}
	return obj
}

func isPlainInt64(t types2.Type) bool {
	return hasBasicKind(t, types2.Int64) || hasBasicKind(t, types2.Uint64)
}
