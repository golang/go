// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

func (g *irgen) def(name *syntax.Name) (*ir.Name, types2.Object) {
	obj, ok := g.info.Defs[name]
	if !ok {
		base.FatalfAt(g.pos(name), "unknown name %v", name)
	}
	return g.obj(obj), obj
}

// use returns the Name or InstExpr node associated with the use of name,
// possibly instantiated by type arguments. The returned node will have
// the correct type and be marked as typechecked.
func (g *irgen) use(name *syntax.Name) ir.Node {
	obj2, ok := g.info.Uses[name]
	if !ok {
		base.FatalfAt(g.pos(name), "unknown name %v", name)
	}
	obj := ir.CaptureName(g.pos(name), ir.CurFunc, g.obj(obj2))
	if obj.Defn != nil && obj.Defn.Op() == ir.ONAME {
		// If CaptureName created a closure variable, then transfer the
		// type of the captured name to the new closure variable.
		obj.SetTypecheck(1)
		obj.SetType(obj.Defn.Type())
	}

	if obj.Class == ir.PFUNC {
		if inst, ok := g.info.Instances[name]; ok {
			// This is the case where inferring types required the
			// types of the function arguments.
			targs := make([]ir.Node, inst.TypeArgs.Len())
			for i := range targs {
				targs[i] = ir.TypeNode(g.typ(inst.TypeArgs.At(i)))
			}
			typ := g.substType(obj.Type(), obj.Type().TParams(), targs)
			return typed(typ, ir.NewInstExpr(g.pos(name), ir.OFUNCINST, obj, targs))
		}
	}

	return obj
}

// obj returns the Name that represents the given object. If no such Name exists
// yet, it will be implicitly created. The returned node will have the correct
// type and be marked as typechecked.
//
// For objects declared at function scope, ir.CurFunc must already be
// set to the respective function when the Name is created.
func (g *irgen) obj(obj types2.Object) *ir.Name {
	// For imported objects, we use iimport directly instead of mapping
	// the types2 representation.
	if obj.Pkg() != g.self {
		if sig, ok := obj.Type().(*types2.Signature); ok && sig.Recv() != nil {
			// We can't import a method by name - must import the type
			// and access the method from it.
			base.FatalfAt(g.pos(obj), "tried to import a method directly")
		}
		sym := g.sym(obj)
		if sym.Def != nil {
			return sym.Def.(*ir.Name)
		}
		n := typecheck.Resolve(ir.NewIdent(src.NoXPos, sym))
		if n, ok := n.(*ir.Name); ok {
			n.SetTypecheck(1)
			return n
		}
		base.FatalfAt(g.pos(obj), "failed to resolve %v", obj)
	}

	if name, ok := g.objs[obj]; ok {
		return name // previously mapped
	}

	var name *ir.Name
	pos := g.pos(obj)

	class := typecheck.DeclContext
	if obj.Parent() == g.self.Scope() {
		class = ir.PEXTERN // forward reference to package-block declaration
	}

	// "You are in a maze of twisting little passages, all different."
	switch obj := obj.(type) {
	case *types2.Const:
		name = g.objCommon(pos, ir.OLITERAL, g.sym(obj), class, g.typ(obj.Type()))

	case *types2.Func:
		sig := obj.Type().(*types2.Signature)
		var sym *types.Sym
		var typ *types.Type
		if recv := sig.Recv(); recv == nil {
			if obj.Name() == "init" {
				sym = renameinit()
			} else {
				sym = g.sym(obj)
			}
			typ = g.typ(sig)
		} else {
			sym = g.selector(obj)
			if !sym.IsBlank() {
				sym = ir.MethodSym(g.typ(recv.Type()), sym)
			}
			typ = g.signature(g.param(recv), sig)
		}
		name = g.objCommon(pos, ir.ONAME, sym, ir.PFUNC, typ)

	case *types2.TypeName:
		if obj.IsAlias() {
			name = g.objCommon(pos, ir.OTYPE, g.sym(obj), class, g.typ(obj.Type()))
			name.SetAlias(true)
		} else {
			name = ir.NewDeclNameAt(pos, ir.OTYPE, g.sym(obj))
			g.objFinish(name, class, types.NewNamed(name))
		}

	case *types2.Var:
		sym := g.sym(obj)
		if class == ir.PPARAMOUT && (sym == nil || sym.IsBlank()) {
			// Backend needs names for result parameters,
			// even if they're anonymous or blank.
			nresults := 0
			for _, n := range ir.CurFunc.Dcl {
				if n.Class == ir.PPARAMOUT {
					nresults++
				}
			}
			if sym == nil {
				sym = typecheck.LookupNum("~r", nresults) // 'r' for "result"
			} else {
				sym = typecheck.LookupNum("~b", nresults) // 'b' for "blank"
			}
		}
		name = g.objCommon(pos, ir.ONAME, sym, class, g.typ(obj.Type()))

	default:
		g.unhandled("object", obj)
	}

	g.objs[obj] = name
	name.SetTypecheck(1)
	return name
}

func (g *irgen) objCommon(pos src.XPos, op ir.Op, sym *types.Sym, class ir.Class, typ *types.Type) *ir.Name {
	name := ir.NewDeclNameAt(pos, op, sym)
	g.objFinish(name, class, typ)
	return name
}

func (g *irgen) objFinish(name *ir.Name, class ir.Class, typ *types.Type) {
	sym := name.Sym()

	name.SetType(typ)
	name.Class = class
	if name.Class == ir.PFUNC {
		sym.SetFunc(true)
	}

	name.SetTypecheck(1)
	name.SetWalkdef(1)

	if ir.IsBlank(name) {
		return
	}

	switch class {
	case ir.PEXTERN:
		g.target.Externs = append(g.target.Externs, name)
		fallthrough
	case ir.PFUNC:
		sym.Def = name
		if name.Class == ir.PFUNC && name.Type().Recv() != nil {
			break // methods are exported with their receiver type
		}
		if types.IsExported(sym.Name) {
			// Generic functions can be marked for export here, even
			// though they will not be compiled until instantiated.
			typecheck.Export(name)
		}
		if base.Flag.AsmHdr != "" && !name.Sym().Asm() {
			name.Sym().SetAsm(true)
			g.target.Asms = append(g.target.Asms, name)
		}

	default:
		// Function-scoped declaration.
		name.Curfn = ir.CurFunc
		if name.Op() == ir.ONAME {
			ir.CurFunc.Dcl = append(ir.CurFunc.Dcl, name)
		}
	}
}
