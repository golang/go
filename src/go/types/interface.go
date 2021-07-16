// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/internal/typeparams"
	"go/token"
	"sort"
)

func (check *Checker) interfaceType(ityp *Interface, iface *ast.InterfaceType, def *Named) {
	var tlist []ast.Expr
	var tname *ast.Ident // "type" name of first entry in a type list declaration

	for _, f := range iface.Methods.List {
		if len(f.Names) == 0 {
			// We have an embedded type; possibly a union of types.
			ityp.embeddeds = append(ityp.embeddeds, parseUnion(check, flattenUnion(nil, f.Type)))
			check.posMap[ityp] = append(check.posMap[ityp], f.Type.Pos())
			continue
		}

		// We have a method with name f.Names[0], or a type
		// of a type list (name.Name == "type").
		// (The parser ensures that there's only one method
		// and we don't care if a constructed AST has more.)
		name := f.Names[0]
		if name.Name == "_" {
			check.errorf(name, _BlankIfaceMethod, "invalid method name _")
			continue // ignore
		}

		// TODO(rfindley) Remove type list handling once the parser doesn't accept type lists anymore.
		if name.Name == "type" {
			// Report an error for the first type list per interface
			// if we don't allow type lists, but continue.
			if !allowTypeLists && tlist == nil {
				check.softErrorf(name, _Todo, "use generalized embedding syntax instead of a type list")
			}
			// For now, collect all type list entries as if it
			// were a single union, where each union element is
			// of the form ~T.
			// TODO(rfindley) remove once we disallow type lists
			op := new(ast.UnaryExpr)
			op.Op = token.TILDE
			op.X = f.Type
			tlist = append(tlist, op)
			// Report an error if we have multiple type lists in an
			// interface, but only if they are permitted in the first place.
			if allowTypeLists && tname != nil && tname != name {
				check.errorf(name, _Todo, "cannot have multiple type lists in an interface")
			}
			tname = name
			continue
		}

		typ := check.typ(f.Type)
		sig, _ := typ.(*Signature)
		if sig == nil {
			if typ != Typ[Invalid] {
				check.invalidAST(f.Type, "%s is not a method signature", typ)
			}
			continue // ignore
		}

		// Always type-check method type parameters but complain if they are not enabled.
		// (This extra check is needed here because interface method signatures don't have
		// a receiver specification.)
		if sig.tparams != nil {
			var at positioner = f.Type
			if tparams := typeparams.Get(f.Type); tparams != nil {
				at = tparams
			}
			check.errorf(at, _Todo, "methods cannot have type parameters")
		}

		// use named receiver type if available (for better error messages)
		var recvTyp Type = ityp
		if def != nil {
			recvTyp = def
		}
		sig.recv = NewVar(name.Pos(), check.pkg, "", recvTyp)

		m := NewFunc(name.Pos(), check.pkg, name.Name, sig)
		check.recordDef(name, m)
		ityp.methods = append(ityp.methods, m)
	}

	// type constraints
	if tlist != nil {
		ityp.embeddeds = append(ityp.embeddeds, parseUnion(check, tlist))
		// Types T in a type list are added as ~T expressions but we don't
		// have the position of the '~'. Use the first type position instead.
		check.posMap[ityp] = append(check.posMap[ityp], tlist[0].(*ast.UnaryExpr).X.Pos())
	}

	// All methods and embedded elements for this interface are collected;
	// i.e., this interface is may be used in a type set computation.
	ityp.complete = true

	if len(ityp.methods) == 0 && len(ityp.embeddeds) == 0 {
		// empty interface
		ityp.tset = &topTypeSet
		return
	}

	// sort for API stability
	sortMethods(ityp.methods)
	sortTypes(ityp.embeddeds)

	// Compute type set with a non-nil *Checker as soon as possible
	// to report any errors. Subsequent uses of type sets should be
	// using this computed type set and won't need to pass in a *Checker.
	check.later(func() { newTypeSet(check, iface.Pos(), ityp) })
}

func flattenUnion(list []ast.Expr, x ast.Expr) []ast.Expr {
	if o, _ := x.(*ast.BinaryExpr); o != nil && o.Op == token.OR {
		list = flattenUnion(list, o.X)
		x = o.Y
	}
	return append(list, x)
}

// newTypeSet may be called with check == nil.
// TODO(gri) move this function into typeset.go eventually
func newTypeSet(check *Checker, pos token.Pos, ityp *Interface) *TypeSet {
	if ityp.tset != nil {
		return ityp.tset
	}

	// If the interface is not fully set up yet, the type set will
	// not be complete, which may lead to errors when using the the
	// type set (e.g. missing method). Don't compute a partial type
	// set (and don't store it!), so that we still compute the full
	// type set eventually. Instead, return the top type set and
	// let any follow-on errors play out.
	//
	// TODO(gri) Consider recording when this happens and reporting
	// it as an error (but only if there were no other errors so to
	// to not have unnecessary follow-on errors).
	if !ityp.complete {
		return &topTypeSet
	}

	if check != nil && trace {
		// Types don't generally have position information.
		// If we don't have a valid pos provided, try to use
		// one close enough.
		if !pos.IsValid() && len(ityp.methods) > 0 {
			pos = ityp.methods[0].pos
		}

		check.trace(pos, "type set for %s", ityp)
		check.indent++
		defer func() {
			check.indent--
			check.trace(pos, "=> %s ", ityp.typeSet())
		}()
	}

	// An infinitely expanding interface (due to a cycle) is detected
	// elsewhere (Checker.validType), so here we simply assume we only
	// have valid interfaces. Mark the interface as complete to avoid
	// infinite recursion if the validType check occurs later for some
	// reason.
	ityp.tset = new(TypeSet) // TODO(gri) is this sufficient?

	// Methods of embedded interfaces are collected unchanged; i.e., the identity
	// of a method I.m's Func Object of an interface I is the same as that of
	// the method m in an interface that embeds interface I. On the other hand,
	// if a method is embedded via multiple overlapping embedded interfaces, we
	// don't provide a guarantee which "original m" got chosen for the embedding
	// interface. See also issue #34421.
	//
	// If we don't care to provide this identity guarantee anymore, instead of
	// reusing the original method in embeddings, we can clone the method's Func
	// Object and give it the position of a corresponding embedded interface. Then
	// we can get rid of the mpos map below and simply use the cloned method's
	// position.

	var todo []*Func
	var seen objset
	var methods []*Func
	mpos := make(map[*Func]token.Pos) // method specification or method embedding position, for good error messages
	addMethod := func(pos token.Pos, m *Func, explicit bool) {
		switch other := seen.insert(m); {
		case other == nil:
			methods = append(methods, m)
			mpos[m] = pos
		case explicit:
			if check == nil {
				panic(fmt.Sprintf("%v: duplicate method %s", m.pos, m.name))
			}
			// check != nil
			check.errorf(atPos(pos), _DuplicateDecl, "duplicate method %s", m.name)
			check.errorf(atPos(mpos[other.(*Func)]), _DuplicateDecl, "\tother declaration of %s", m.name) // secondary error, \t indented
		default:
			// We have a duplicate method name in an embedded (not explicitly declared) method.
			// Check method signatures after all types are computed (issue #33656).
			// If we're pre-go1.14 (overlapping embeddings are not permitted), report that
			// error here as well (even though we could do it eagerly) because it's the same
			// error message.
			if check == nil {
				// check method signatures after all locally embedded interfaces are computed
				todo = append(todo, m, other.(*Func))
				break
			}
			// check != nil
			check.later(func() {
				if !check.allowVersion(m.pkg, 1, 14) || !check.identical(m.typ, other.Type()) {
					check.errorf(atPos(pos), _DuplicateDecl, "duplicate method %s", m.name)
					check.errorf(atPos(mpos[other.(*Func)]), _DuplicateDecl, "\tother declaration of %s", m.name) // secondary error, \t indented
				}
			})
		}
	}

	for _, m := range ityp.methods {
		addMethod(m.pos, m, true)
	}

	// collect embedded elements
	var allTypes Type
	var posList []token.Pos
	if check != nil {
		posList = check.posMap[ityp]
	}
	for i, typ := range ityp.embeddeds {
		var pos token.Pos // embedding position
		if posList != nil {
			pos = posList[i]
		}
		var types Type
		switch t := under(typ).(type) {
		case *Interface:
			tset := newTypeSet(check, pos, t)
			for _, m := range tset.methods {
				addMethod(pos, m, false) // use embedding position pos rather than m.pos

			}
			types = tset.types
		case *Union:
			// TODO(gri) combine with default case once we have
			//           converted all tests to new notation and we
			//           can report an error when we don't have an
			//           interface before go1.18.
			types = typ
		case *TypeParam:
			if check != nil && !check.allowVersion(check.pkg, 1, 18) {
				check.errorf(atPos(pos), _InvalidIfaceEmbed, "%s is a type parameter, not an interface", typ)
				continue
			}
			types = typ
		default:
			if typ == Typ[Invalid] {
				continue
			}
			if check != nil && !check.allowVersion(check.pkg, 1, 18) {
				check.errorf(atPos(pos), _InvalidIfaceEmbed, "%s is not an interface", typ)
				continue
			}
			types = typ
		}
		allTypes = intersect(allTypes, types)
	}

	// process todo's (this only happens if check == nil)
	for i := 0; i < len(todo); i += 2 {
		m := todo[i]
		other := todo[i+1]
		if !Identical(m.typ, other.typ) {
			panic(fmt.Sprintf("%v: duplicate method %s", m.pos, m.name))
		}
	}

	if methods != nil {
		sort.Sort(byUniqueMethodName(methods))
		ityp.tset.methods = methods
	}
	ityp.tset.types = allTypes

	return ityp.tset
}

func sortTypes(list []Type) {
	sort.Stable(byUniqueTypeName(list))
}

// byUniqueTypeName named type lists can be sorted by their unique type names.
type byUniqueTypeName []Type

func (a byUniqueTypeName) Len() int           { return len(a) }
func (a byUniqueTypeName) Less(i, j int) bool { return sortName(a[i]) < sortName(a[j]) }
func (a byUniqueTypeName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func sortName(t Type) string {
	if named := asNamed(t); named != nil {
		return named.obj.Id()
	}
	return ""
}

func sortMethods(list []*Func) {
	sort.Sort(byUniqueMethodName(list))
}

func assertSortedMethods(list []*Func) {
	if !debug {
		panic("internal error: assertSortedMethods called outside debug mode")
	}
	if !sort.IsSorted(byUniqueMethodName(list)) {
		panic("internal error: methods not sorted")
	}
}

// byUniqueMethodName method lists can be sorted by their unique method names.
type byUniqueMethodName []*Func

func (a byUniqueMethodName) Len() int           { return len(a) }
func (a byUniqueMethodName) Less(i, j int) bool { return a[i].Id() < a[j].Id() }
func (a byUniqueMethodName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
