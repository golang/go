// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file will evolve, since we plan to do a mix of stenciling and passing
// around dictionaries.

package noder

import (
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"fmt"
)

// stencil scans functions for instantiated generic function calls and
// creates the required stencils for simple generic functions.
func (g *irgen) stencil() {
	g.target.Stencils = make(map[*types.Sym]*ir.Func)
	for _, decl := range g.target.Decls {
		if decl.Op() != ir.ODCLFUNC || decl.Type().NumTParams() > 0 {
			// Skip any non-function declarations and skip generic functions
			continue
		}

		// For each non-generic function, search for any function calls using
		// generic function instantiations. (We don't yet handle generic
		// function instantiations that are not immediately called.)
		// Then create the needed instantiated function if it hasn't been
		// created yet, and change to calling that function directly.
		f := decl.(*ir.Func)
		modified := false
		ir.VisitList(f.Body, func(n ir.Node) {
			if n.Op() != ir.OCALLFUNC || n.(*ir.CallExpr).X.Op() != ir.OFUNCINST {
				return
			}
			// We have found a function call using a generic function
			// instantiation.
			call := n.(*ir.CallExpr)
			inst := call.X.(*ir.InstExpr)
			sym := makeInstName(inst)
			//fmt.Printf("Found generic func call in %v to %v\n", f, s)
			st := g.target.Stencils[sym]
			if st == nil {
				// If instantiation doesn't exist yet, create it and add
				// to the list of decls.
				st = genericSubst(sym, inst)
				g.target.Stencils[sym] = st
				g.target.Decls = append(g.target.Decls, st)
				if base.Flag.W > 1 {
					ir.Dump(fmt.Sprintf("\nstenciled %v", st), st)
				}
			}
			// Replace the OFUNCINST with a direct reference to the
			// new stenciled function
			call.X = st.Nname
			modified = true
		})
		if base.Flag.W > 1 && modified {
			ir.Dump(fmt.Sprintf("\nmodified %v", decl), decl)
		}
	}

}

// makeInstName makes the unique name for a stenciled generic function, based on
// the name of the function and the types of the type params.
func makeInstName(inst *ir.InstExpr) *types.Sym {
	b := bytes.NewBufferString("#")
	b.WriteString(inst.X.(*ir.Name).Name().Sym().Name)
	b.WriteString("[")
	for i, targ := range inst.Targs {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(targ.Name().Sym().Name)
	}
	b.WriteString("]")
	return typecheck.Lookup(b.String())
}

// Struct containing info needed for doing the substitution as we create the
// instantiation of a generic function with specified type arguments.
type subster struct {
	newf    *ir.Func // Func node for the new stenciled function
	tparams *types.Fields
	targs   []ir.Node
	// The substitution map from name nodes in the generic function to the
	// name nodes in the new stenciled function.
	vars map[*ir.Name]*ir.Name
}

// genericSubst returns a new function with the specified name. The function is an
// instantiation of a generic function with type params, as specified by inst.
func genericSubst(name *types.Sym, inst *ir.InstExpr) *ir.Func {
	// Similar to noder.go: funcDecl
	nameNode := inst.X.(*ir.Name)
	gf := nameNode.Func
	newf := ir.NewFunc(inst.Pos())
	newf.Nname = ir.NewNameAt(inst.Pos(), name)
	newf.Nname.Func = newf
	newf.Nname.Defn = newf

	subst := &subster{
		newf:    newf,
		tparams: nameNode.Type().TParams().Fields(),
		targs:   inst.Targs,
		vars:    make(map[*ir.Name]*ir.Name),
	}

	newf.Dcl = make([]*ir.Name, len(gf.Dcl))
	for i, n := range gf.Dcl {
		newf.Dcl[i] = subst.node(n).(*ir.Name)
	}
	newf.Body = subst.list(gf.Body)

	// Ugly: we have to insert the Name nodes of the parameters/results into
	// the function type. The current function type has no Nname fields set,
	// because it came via conversion from the types2 type.
	oldt := inst.Type()
	newt := types.NewSignature(oldt.Pkg(), nil, nil, subst.fields(ir.PPARAM, oldt.Params(), newf.Dcl),
		subst.fields(ir.PPARAMOUT, oldt.Results(), newf.Dcl))

	newf.Nname.Ntype = ir.TypeNode(newt)
	newf.Nname.SetType(newt)
	ir.MarkFunc(newf.Nname)
	newf.SetTypecheck(1)
	newf.Nname.SetTypecheck(1)
	// TODO(danscales) - remove later, but avoid confusion for now.
	newf.Pragma = ir.Noinline
	return newf
}

// node is like DeepCopy(), but creates distinct ONAME nodes, and also descends
// into closures. It substitutes type arguments for type parameters in all the new
// nodes.
func (subst *subster) node(n ir.Node) ir.Node {
	// Use closure to capture all state needed by the ir.EditChildren argument.
	var edit func(ir.Node) ir.Node
	edit = func(x ir.Node) ir.Node {
		switch x.Op() {
		case ir.ONAME:
			name := x.(*ir.Name)
			if v := subst.vars[name]; v != nil {
				return v
			}
			m := ir.NewNameAt(name.Pos(), name.Sym())
			t := x.Type()
			newt := subst.typ(t)
			m.SetType(newt)
			m.Curfn = subst.newf
			m.Class = name.Class
			subst.vars[name] = m
			m.SetTypecheck(1)
			return m
		case ir.OLITERAL, ir.ONIL:
			if x.Sym() != nil {
				return x
			}
		}
		m := ir.Copy(x)
		if _, isExpr := m.(ir.Expr); isExpr {
			m.SetType(subst.typ(x.Type()))
		}
		ir.EditChildren(m, edit)
		if x.Op() == ir.OCLOSURE {
			x := x.(*ir.ClosureExpr)
			// Need to save/duplicate x.Func.Nname,
			// x.Func.Nname.Ntype, x.Func.Dcl, x.Func.ClosureVars, and
			// x.Func.Body.
			oldfn := x.Func
			newfn := ir.NewFunc(oldfn.Pos())
			if oldfn.ClosureCalled() {
				newfn.SetClosureCalled(true)
			}
			m.(*ir.ClosureExpr).Func = newfn
			newfn.Nname = ir.NewNameAt(oldfn.Nname.Pos(), oldfn.Nname.Sym())
			newfn.Nname.SetType(oldfn.Nname.Type())
			newfn.Nname.Ntype = subst.node(oldfn.Nname.Ntype).(ir.Ntype)
			newfn.Body = subst.list(oldfn.Body)
			// Make shallow copy of the Dcl and ClosureVar slices
			newfn.Dcl = append([]*ir.Name(nil), oldfn.Dcl...)
			newfn.ClosureVars = append([]*ir.Name(nil), oldfn.ClosureVars...)
		}
		return m
	}

	return edit(n)
}

func (subst *subster) list(l []ir.Node) []ir.Node {
	s := make([]ir.Node, len(l))
	for i, n := range l {
		s[i] = subst.node(n)
	}
	return s
}

// typ substitutes any type parameter found with the corresponding type argument.
func (subst *subster) typ(t *types.Type) *types.Type {
	for i, tp := range subst.tparams.Slice() {
		if tp.Type == t {
			return subst.targs[i].Type()
		}
	}

	switch t.Kind() {
	case types.TARRAY:
		elem := t.Elem()
		newelem := subst.typ(elem)
		if subst.typ(elem) != elem {
			return types.NewArray(newelem, t.NumElem())
		}

	case types.TPTR:
		elem := t.Elem()
		newelem := subst.typ(elem)
		if subst.typ(elem) != elem {
			return types.NewPtr(newelem)
		}

	case types.TSLICE:
		elem := t.Elem()
		newelem := subst.typ(elem)
		if subst.typ(elem) != elem {
			return types.NewSlice(newelem)
		}

	case types.TSTRUCT:
		newfields := make([]*types.Field, t.NumFields())
		change := false
		for i, f := range t.Fields().Slice() {
			t2 := subst.typ(f.Type)
			if t2 != f.Type {
				change = true
			}
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
		}
		if change {
			return types.NewStruct(t.Pkg(), newfields)
		}

		// TODO: case TFUNC
		// TODO: case TCHAN
		// TODO: case TMAP
		// TODO: case TINTER
	}
	return t
}

// fields sets the Nname field for the Field nodes inside a type signature, based
// on the corresponding in/out parameters in dcl. It depends on the in and out
// parameters being in order in dcl.
func (subst *subster) fields(class ir.Class, oldt *types.Type, dcl []*ir.Name) []*types.Field {
	oldfields := oldt.FieldSlice()
	newfields := make([]*types.Field, len(oldfields))
	var i int

	// Find the starting index in dcl of declarations of the class (either
	// PPARAM or PPARAMOUT).
	for i = range dcl {
		if dcl[i].Class == class {
			break
		}
	}

	// Create newfields nodes that are copies of the oldfields nodes, but
	// with substitution for any type params, and with Nname set to be the node in
	// Dcl for the corresponding PPARAM or PPARAMOUT.
	for j := range oldfields {
		newfields[j] = oldfields[j].Copy()
		newfields[j].Type = subst.typ(oldfields[j].Type)
		newfields[j].Nname = dcl[i]
		i++
	}
	return newfields
}
