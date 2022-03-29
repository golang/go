// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
)

// TODO(mdempsky): Skip blank declarations? Probably only safe
// for declarations without pragmas.

func (g *irgen) decls(res *ir.Nodes, decls []syntax.Decl) {
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *syntax.ConstDecl:
			g.constDecl(res, decl)
		case *syntax.FuncDecl:
			g.funcDecl(res, decl)
		case *syntax.TypeDecl:
			if ir.CurFunc == nil {
				continue // already handled in irgen.generate
			}
			g.typeDecl(res, decl)
		case *syntax.VarDecl:
			g.varDecl(res, decl)
		default:
			g.unhandled("declaration", decl)
		}
	}
}

func (g *irgen) importDecl(p *noder, decl *syntax.ImportDecl) {
	g.pragmaFlags(decl.Pragma, 0)

	// Get the imported package's path, as resolved already by types2
	// and gcimporter. This is the same path as would be computed by
	// parseImportPath.
	switch pkgNameOf(g.info, decl).Imported().Path() {
	case "unsafe":
		p.importedUnsafe = true
	case "embed":
		p.importedEmbed = true
	}
}

// pkgNameOf returns the PkgName associated with the given ImportDecl.
func pkgNameOf(info *types2.Info, decl *syntax.ImportDecl) *types2.PkgName {
	if name := decl.LocalPkgName; name != nil {
		return info.Defs[name].(*types2.PkgName)
	}
	return info.Implicits[decl].(*types2.PkgName)
}

func (g *irgen) constDecl(out *ir.Nodes, decl *syntax.ConstDecl) {
	g.pragmaFlags(decl.Pragma, 0)

	for _, name := range decl.NameList {
		name, obj := g.def(name)

		// For untyped numeric constants, make sure the value
		// representation matches what the rest of the
		// compiler (really just iexport) expects.
		// TODO(mdempsky): Revisit after #43891 is resolved.
		val := obj.(*types2.Const).Val()
		switch name.Type() {
		case types.UntypedInt, types.UntypedRune:
			val = constant.ToInt(val)
		case types.UntypedFloat:
			val = constant.ToFloat(val)
		case types.UntypedComplex:
			val = constant.ToComplex(val)
		}
		name.SetVal(val)

		out.Append(ir.NewDecl(g.pos(decl), ir.ODCLCONST, name))
	}
}

func (g *irgen) funcDecl(out *ir.Nodes, decl *syntax.FuncDecl) {
	assert(g.curDecl == "")
	// Set g.curDecl to the function name, as context for the type params declared
	// during types2-to-types1 translation if this is a generic function.
	g.curDecl = decl.Name.Value
	obj2 := g.info.Defs[decl.Name]
	recv := types2.AsSignature(obj2.Type()).Recv()
	if recv != nil {
		t2 := deref2(recv.Type())
		// This is a method, so set g.curDecl to recvTypeName.methName instead.
		g.curDecl = t2.(*types2.Named).Obj().Name() + "." + g.curDecl
	}

	fn := ir.NewFunc(g.pos(decl))
	fn.Nname, _ = g.def(decl.Name)
	fn.Nname.Func = fn
	fn.Nname.Defn = fn

	fn.Pragma = g.pragmaFlags(decl.Pragma, funcPragmas)
	if fn.Pragma&ir.Systemstack != 0 && fn.Pragma&ir.Nosplit != 0 {
		base.ErrorfAt(fn.Pos(), "go:nosplit and go:systemstack cannot be combined")
	}
	if fn.Pragma&ir.Nointerface != 0 {
		// Propagate //go:nointerface from Func.Pragma to Field.Nointerface.
		// This is a bit roundabout, but this is the earliest point where we've
		// processed the function's pragma flags, and we've also already created
		// the Fields to represent the receiver's method set.
		if recv := fn.Type().Recv(); recv != nil {
			typ := types.ReceiverBaseType(recv.Type)
			if orig := typ.OrigType(); orig != nil {
				// For a generic method, we mark the methods on the
				// base generic type, since those are the methods
				// that will be stenciled.
				typ = orig
			}
			meth := typecheck.Lookdot1(fn, typecheck.Lookup(decl.Name.Value), typ, typ.Methods(), 0)
			meth.SetNointerface(true)
		}
	}

	if decl.Body != nil && fn.Pragma&ir.Noescape != 0 {
		base.ErrorfAt(fn.Pos(), "can only use //go:noescape with external func implementations")
	}

	if decl.Name.Value == "init" && decl.Recv == nil {
		g.target.Inits = append(g.target.Inits, fn)
	}

	saveHaveEmbed := g.haveEmbed
	saveCurDecl := g.curDecl
	g.curDecl = ""
	g.later(func() {
		defer func(b bool, s string) {
			// Revert haveEmbed and curDecl back to what they were before
			// the "later" function.
			g.haveEmbed = b
			g.curDecl = s
		}(g.haveEmbed, g.curDecl)

		// Set haveEmbed and curDecl to what they were for this funcDecl.
		g.haveEmbed = saveHaveEmbed
		g.curDecl = saveCurDecl
		if fn.Type().HasTParam() {
			g.topFuncIsGeneric = true
		}
		g.funcBody(fn, decl.Recv, decl.Type, decl.Body)
		g.topFuncIsGeneric = false
		if fn.Type().HasTParam() && fn.Body != nil {
			// Set pointers to the dcls/body of a generic function/method in
			// the Inl struct, so it is marked for export, is available for
			// stenciling, and works with Inline_Flood().
			fn.Inl = &ir.Inline{
				Cost: 1,
				Dcl:  fn.Dcl,
				Body: fn.Body,
			}
		}

		out.Append(fn)
	})
}

func (g *irgen) typeDecl(out *ir.Nodes, decl *syntax.TypeDecl) {
	// Set the position for any error messages we might print (e.g. too large types).
	base.Pos = g.pos(decl)
	assert(ir.CurFunc != nil || g.curDecl == "")
	// Set g.curDecl to the type name, as context for the type params declared
	// during types2-to-types1 translation if this is a generic type.
	saveCurDecl := g.curDecl
	g.curDecl = decl.Name.Value
	if decl.Alias {
		name, _ := g.def(decl.Name)
		g.pragmaFlags(decl.Pragma, 0)
		assert(name.Alias()) // should be set by irgen.obj

		out.Append(ir.NewDecl(g.pos(decl), ir.ODCLTYPE, name))
		g.curDecl = ""
		return
	}

	// Prevent size calculations until we set the underlying type.
	types.DeferCheckSize()

	name, obj := g.def(decl.Name)
	ntyp, otyp := name.Type(), obj.Type()
	if ir.CurFunc != nil {
		ntyp.SetVargen()
	}

	pragmas := g.pragmaFlags(decl.Pragma, typePragmas)
	name.SetPragma(pragmas) // TODO(mdempsky): Is this still needed?

	if pragmas&ir.NotInHeap != 0 {
		ntyp.SetNotInHeap(true)
	}

	// We need to use g.typeExpr(decl.Type) here to ensure that for
	// chained, defined-type declarations like:
	//
	//	type T U
	//
	//	//go:notinheap
	//	type U struct { … }
	//
	// we mark both T and U as NotInHeap. If we instead used just
	// g.typ(otyp.Underlying()), then we'd instead set T's underlying
	// type directly to the struct type (which is not marked NotInHeap)
	// and fail to mark T as NotInHeap.
	//
	// Also, we rely here on Type.SetUnderlying allowing passing a
	// defined type and handling forward references like from T to U
	// above. Contrast with go/types's Named.SetUnderlying, which
	// disallows this.
	//
	// [mdempsky: Subtleties like these are why I always vehemently
	// object to new type pragmas.]
	ntyp.SetUnderlying(g.typeExpr(decl.Type))

	tparams := otyp.(*types2.Named).TypeParams()
	if n := tparams.Len(); n > 0 {
		rparams := make([]*types.Type, n)
		for i := range rparams {
			rparams[i] = g.typ(tparams.At(i))
		}
		// This will set hasTParam flag if any rparams are not concrete types.
		ntyp.SetRParams(rparams)
	}
	types.ResumeCheckSize()

	g.curDecl = saveCurDecl
	if otyp, ok := otyp.(*types2.Named); ok && otyp.NumMethods() != 0 {
		methods := make([]*types.Field, otyp.NumMethods())
		for i := range methods {
			m := otyp.Method(i)
			// Set g.curDecl to recvTypeName.methName, as context for the
			// method-specific type params in the receiver.
			g.curDecl = decl.Name.Value + "." + m.Name()
			meth := g.obj(m)
			methods[i] = types.NewField(meth.Pos(), g.selector(m), meth.Type())
			methods[i].Nname = meth
			g.curDecl = ""
		}
		ntyp.Methods().Set(methods)
	}

	out.Append(ir.NewDecl(g.pos(decl), ir.ODCLTYPE, name))
}

func (g *irgen) varDecl(out *ir.Nodes, decl *syntax.VarDecl) {
	pos := g.pos(decl)
	// Set the position for any error messages we might print (e.g. too large types).
	base.Pos = pos
	names := make([]*ir.Name, len(decl.NameList))
	for i, name := range decl.NameList {
		names[i], _ = g.def(name)
	}

	if decl.Pragma != nil {
		pragma := decl.Pragma.(*pragmas)
		varEmbed(g.makeXPos, names[0], decl, pragma, g.haveEmbed)
		g.reportUnused(pragma)
	}

	haveEmbed := g.haveEmbed
	do := func() {
		defer func(b bool) { g.haveEmbed = b }(g.haveEmbed)

		g.haveEmbed = haveEmbed
		values := g.exprList(decl.Values)

		var as2 *ir.AssignListStmt
		if len(values) != 0 && len(names) != len(values) {
			as2 = ir.NewAssignListStmt(pos, ir.OAS2, make([]ir.Node, len(names)), values)
		}

		for i, name := range names {
			if ir.CurFunc != nil {
				out.Append(ir.NewDecl(pos, ir.ODCL, name))
			}
			if as2 != nil {
				as2.Lhs[i] = name
				name.Defn = as2
			} else {
				as := ir.NewAssignStmt(pos, name, nil)
				if len(values) != 0 {
					as.Y = values[i]
					name.Defn = as
				} else if ir.CurFunc == nil {
					name.Defn = as
				}
				if !g.delayTransform() {
					lhs := []ir.Node{as.X}
					rhs := []ir.Node{}
					if as.Y != nil {
						rhs = []ir.Node{as.Y}
					}
					transformAssign(as, lhs, rhs)
					as.X = lhs[0]
					if as.Y != nil {
						as.Y = rhs[0]
					}
				}
				as.SetTypecheck(1)
				out.Append(as)
			}
		}
		if as2 != nil {
			if !g.delayTransform() {
				transformAssign(as2, as2.Lhs, as2.Rhs)
			}
			as2.SetTypecheck(1)
			out.Append(as2)
		}
	}

	// If we're within a function, we need to process the assignment
	// part of the variable declaration right away. Otherwise, we leave
	// it to be handled after all top-level declarations are processed.
	if ir.CurFunc != nil {
		do()
	} else {
		g.later(do)
	}
}

// pragmaFlags returns any specified pragma flags included in allowed,
// and reports errors about any other, unexpected pragmas.
func (g *irgen) pragmaFlags(pragma syntax.Pragma, allowed ir.PragmaFlag) ir.PragmaFlag {
	if pragma == nil {
		return 0
	}
	p := pragma.(*pragmas)
	present := p.Flag & allowed
	p.Flag &^= allowed
	g.reportUnused(p)
	return present
}

// reportUnused reports errors about any unused pragmas.
func (g *irgen) reportUnused(pragma *pragmas) {
	for _, pos := range pragma.Pos {
		if pos.Flag&pragma.Flag != 0 {
			base.ErrorfAt(g.makeXPos(pos.Pos), "misplaced compiler directive")
		}
	}
	if len(pragma.Embeds) > 0 {
		for _, e := range pragma.Embeds {
			base.ErrorfAt(g.makeXPos(e.Pos), "misplaced go:embed directive")
		}
	}
}
