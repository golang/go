// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// This file defines helper functions useful for satisfying toolstash
// -cmp when compared against the legacy frontend behavior, but can be
// removed after that's no longer a concern.

// quirksMode controls whether behavior specific to satisfying
// toolstash -cmp is used.
func quirksMode() bool {
	return base.Debug.UnifiedQuirks != 0
}

// posBasesOf returns all of the position bases in the source files,
// as seen in a straightforward traversal.
//
// This is necessary to ensure position bases (and thus file names)
// get registered in the same order as noder would visit them.
func posBasesOf(noders []*noder) []*syntax.PosBase {
	seen := make(map[*syntax.PosBase]bool)
	var bases []*syntax.PosBase

	for _, p := range noders {
		syntax.Crawl(p.file, func(n syntax.Node) bool {
			if b := n.Pos().Base(); !seen[b] {
				bases = append(bases, b)
				seen[b] = true
			}
			return false
		})
	}

	return bases
}

// importedObjsOf returns the imported objects (i.e., referenced
// objects not declared by curpkg) from the parsed source files, in
// the order that typecheck used to load their definitions.
//
// This is needed because loading the definitions for imported objects
// can also add file names.
func importedObjsOf(curpkg *types2.Package, info *types2.Info, noders []*noder) []types2.Object {
	// This code is complex because it matches the precise order that
	// typecheck recursively and repeatedly traverses the IR. It's meant
	// to be thrown away eventually anyway.

	seen := make(map[types2.Object]bool)
	var objs []types2.Object

	var phase int

	decls := make(map[types2.Object]syntax.Decl)
	assoc := func(decl syntax.Decl, names ...*syntax.Name) {
		for _, name := range names {
			obj, ok := info.Defs[name]
			assert(ok)
			decls[obj] = decl
		}
	}

	for _, p := range noders {
		syntax.Crawl(p.file, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.ConstDecl:
				assoc(n, n.NameList...)
			case *syntax.FuncDecl:
				assoc(n, n.Name)
			case *syntax.TypeDecl:
				assoc(n, n.Name)
			case *syntax.VarDecl:
				assoc(n, n.NameList...)
			case *syntax.BlockStmt:
				return true
			}
			return false
		})
	}

	var visited map[syntax.Decl]bool

	var resolveDecl func(n syntax.Decl)
	var resolveNode func(n syntax.Node, top bool)

	resolveDecl = func(n syntax.Decl) {
		if visited[n] {
			return
		}
		visited[n] = true

		switch n := n.(type) {
		case *syntax.ConstDecl:
			resolveNode(n.Type, true)
			resolveNode(n.Values, true)

		case *syntax.FuncDecl:
			if n.Recv != nil {
				resolveNode(n.Recv, true)
			}
			resolveNode(n.Type, true)

		case *syntax.TypeDecl:
			resolveNode(n.Type, true)

		case *syntax.VarDecl:
			if n.Type != nil {
				resolveNode(n.Type, true)
			} else {
				resolveNode(n.Values, true)
			}
		}
	}

	resolveObj := func(pos syntax.Pos, obj types2.Object) {
		switch obj.Pkg() {
		case nil:
			// builtin; nothing to do

		case curpkg:
			if decl, ok := decls[obj]; ok {
				resolveDecl(decl)
			}

		default:
			if obj.Parent() == obj.Pkg().Scope() && !seen[obj] {
				seen[obj] = true
				objs = append(objs, obj)
			}
		}
	}

	checkdefat := func(pos syntax.Pos, n *syntax.Name) {
		if n.Value == "_" {
			return
		}
		obj, ok := info.Uses[n]
		if !ok {
			obj, ok = info.Defs[n]
			if !ok {
				return
			}
		}
		if obj == nil {
			return
		}
		resolveObj(pos, obj)
	}
	checkdef := func(n *syntax.Name) { checkdefat(n.Pos(), n) }

	var later []syntax.Node

	resolveNode = func(n syntax.Node, top bool) {
		if n == nil {
			return
		}
		syntax.Crawl(n, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.Name:
				checkdef(n)

			case *syntax.SelectorExpr:
				if name, ok := n.X.(*syntax.Name); ok {
					if _, isPkg := info.Uses[name].(*types2.PkgName); isPkg {
						checkdefat(n.X.Pos(), n.Sel)
						return true
					}
				}

			case *syntax.AssignStmt:
				resolveNode(n.Rhs, top)
				resolveNode(n.Lhs, top)
				return true

			case *syntax.VarDecl:
				resolveNode(n.Values, top)

			case *syntax.FuncLit:
				if top {
					resolveNode(n.Type, top)
					later = append(later, n.Body)
					return true
				}

			case *syntax.BlockStmt:
				if phase >= 3 {
					for _, stmt := range n.List {
						resolveNode(stmt, false)
					}
				}
				return true
			}

			return false
		})
	}

	for phase = 1; phase <= 5; phase++ {
		visited = map[syntax.Decl]bool{}

		for _, p := range noders {
			for _, decl := range p.file.DeclList {
				switch decl := decl.(type) {
				case *syntax.ConstDecl:
					resolveDecl(decl)

				case *syntax.FuncDecl:
					resolveDecl(decl)
					if phase >= 3 && decl.Body != nil {
						resolveNode(decl.Body, true)
					}

				case *syntax.TypeDecl:
					if !decl.Alias || phase >= 2 {
						resolveDecl(decl)
					}

				case *syntax.VarDecl:
					if phase >= 2 {
						resolveNode(decl.Values, true)
						resolveDecl(decl)
					}
				}
			}

			if phase >= 5 {
				syntax.Crawl(p.file, func(n syntax.Node) bool {
					if name, ok := n.(*syntax.Name); ok {
						if obj, ok := info.Uses[name]; ok {
							resolveObj(name.Pos(), obj)
						}
					}
					return false
				})
			}
		}

		for i := 0; i < len(later); i++ {
			resolveNode(later[i], true)
		}
		later = nil
	}

	return objs
}

// typeExprEndPos returns the position that noder would leave base.Pos
// after parsing the given type expression.
func typeExprEndPos(expr0 syntax.Expr) syntax.Pos {
	for {
		switch expr := expr0.(type) {
		case *syntax.Name:
			return expr.Pos()
		case *syntax.SelectorExpr:
			return expr.X.Pos()

		case *syntax.ParenExpr:
			expr0 = expr.X

		case *syntax.Operation:
			assert(expr.Op == syntax.Mul)
			assert(expr.Y == nil)
			expr0 = expr.X

		case *syntax.ArrayType:
			expr0 = expr.Elem
		case *syntax.ChanType:
			expr0 = expr.Elem
		case *syntax.DotsType:
			expr0 = expr.Elem
		case *syntax.MapType:
			expr0 = expr.Value
		case *syntax.SliceType:
			expr0 = expr.Elem

		case *syntax.StructType:
			return expr.Pos()

		case *syntax.InterfaceType:
			expr0 = lastFieldType(expr.MethodList)
			if expr0 == nil {
				return expr.Pos()
			}

		case *syntax.FuncType:
			expr0 = lastFieldType(expr.ResultList)
			if expr0 == nil {
				expr0 = lastFieldType(expr.ParamList)
				if expr0 == nil {
					return expr.Pos()
				}
			}

		case *syntax.IndexExpr: // explicit type instantiation
			targs := unpackListExpr(expr.Index)
			expr0 = targs[len(targs)-1]

		default:
			panic(fmt.Sprintf("%s: unexpected type expression %v", expr.Pos(), syntax.String(expr)))
		}
	}
}

func lastFieldType(fields []*syntax.Field) syntax.Expr {
	if len(fields) == 0 {
		return nil
	}
	return fields[len(fields)-1].Type
}

// sumPos returns the position that noder.sum would produce for
// constant expression x.
func sumPos(x syntax.Expr) syntax.Pos {
	orig := x
	for {
		switch x1 := x.(type) {
		case *syntax.BasicLit:
			assert(x1.Kind == syntax.StringLit)
			return x1.Pos()
		case *syntax.Operation:
			assert(x1.Op == syntax.Add && x1.Y != nil)
			if r, ok := x1.Y.(*syntax.BasicLit); ok {
				assert(r.Kind == syntax.StringLit)
				x = x1.X
				continue
			}
		}
		return orig.Pos()
	}
}

// funcParamsEndPos returns the value of base.Pos left by noder after
// processing a function signature.
func funcParamsEndPos(fn *ir.Func) src.XPos {
	sig := fn.Nname.Type()

	fields := sig.Results().FieldSlice()
	if len(fields) == 0 {
		fields = sig.Params().FieldSlice()
		if len(fields) == 0 {
			fields = sig.Recvs().FieldSlice()
			if len(fields) == 0 {
				if fn.OClosure != nil {
					return fn.Nname.Ntype.Pos()
				}
				return fn.Pos()
			}
		}
	}

	return fields[len(fields)-1].Pos
}

type dupTypes struct {
	origs map[types2.Type]types2.Type
}

func (d *dupTypes) orig(t types2.Type) types2.Type {
	if orig, ok := d.origs[t]; ok {
		return orig
	}
	return t
}

func (d *dupTypes) add(t, orig types2.Type) {
	if t == orig {
		return
	}

	if d.origs == nil {
		d.origs = make(map[types2.Type]types2.Type)
	}
	assert(d.origs[t] == nil)
	d.origs[t] = orig

	switch t := t.(type) {
	case *types2.Pointer:
		orig := orig.(*types2.Pointer)
		d.add(t.Elem(), orig.Elem())

	case *types2.Slice:
		orig := orig.(*types2.Slice)
		d.add(t.Elem(), orig.Elem())

	case *types2.Map:
		orig := orig.(*types2.Map)
		d.add(t.Key(), orig.Key())
		d.add(t.Elem(), orig.Elem())

	case *types2.Array:
		orig := orig.(*types2.Array)
		assert(t.Len() == orig.Len())
		d.add(t.Elem(), orig.Elem())

	case *types2.Chan:
		orig := orig.(*types2.Chan)
		assert(t.Dir() == orig.Dir())
		d.add(t.Elem(), orig.Elem())

	case *types2.Struct:
		orig := orig.(*types2.Struct)
		assert(t.NumFields() == orig.NumFields())
		for i := 0; i < t.NumFields(); i++ {
			d.add(t.Field(i).Type(), orig.Field(i).Type())
		}

	case *types2.Interface:
		orig := orig.(*types2.Interface)
		assert(t.NumExplicitMethods() == orig.NumExplicitMethods())
		assert(t.NumEmbeddeds() == orig.NumEmbeddeds())
		for i := 0; i < t.NumExplicitMethods(); i++ {
			d.add(t.ExplicitMethod(i).Type(), orig.ExplicitMethod(i).Type())
		}
		for i := 0; i < t.NumEmbeddeds(); i++ {
			d.add(t.EmbeddedType(i), orig.EmbeddedType(i))
		}

	case *types2.Signature:
		orig := orig.(*types2.Signature)
		assert((t.Recv() == nil) == (orig.Recv() == nil))
		if t.Recv() != nil {
			d.add(t.Recv().Type(), orig.Recv().Type())
		}
		d.add(t.Params(), orig.Params())
		d.add(t.Results(), orig.Results())

	case *types2.Tuple:
		orig := orig.(*types2.Tuple)
		assert(t.Len() == orig.Len())
		for i := 0; i < t.Len(); i++ {
			d.add(t.At(i).Type(), orig.At(i).Type())
		}

	default:
		assert(types2.Identical(t, orig))
	}
}
