// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// ----------------------------------------------------------------------------
// API

// An Interface represents an interface type.
type Interface struct {
	check     *Checker      // for error reporting; nil once type set is computed
	obj       *TypeName     // corresponding declared object; or nil (for better error messages)
	methods   []*Func       // ordered list of explicitly declared methods
	embeddeds []Type        // ordered list of explicitly embedded elements
	embedPos  *[]syntax.Pos // positions of embedded elements; or nil (for error messages) - use pointer to save space
	complete  bool          // indicates that all fields (except for tset) are set up

	tset *_TypeSet // type set described by this interface, computed lazily
}

// typeSet returns the type set for interface t.
func (t *Interface) typeSet() *_TypeSet { return computeInterfaceTypeSet(t.check, nopos, t) }

// emptyInterface represents the empty interface
var emptyInterface = Interface{complete: true, tset: &topTypeSet}

// NewInterface returns a new interface for the given methods and embedded types.
// NewInterface takes ownership of the provided methods and may modify their types
// by setting missing receivers.
//
// Deprecated: Use NewInterfaceType instead which allows arbitrary embedded types.
func NewInterface(methods []*Func, embeddeds []*Named) *Interface {
	tnames := make([]Type, len(embeddeds))
	for i, t := range embeddeds {
		tnames[i] = t
	}
	return NewInterfaceType(methods, tnames)
}

// NewInterfaceType returns a new interface for the given methods and embedded types.
// NewInterfaceType takes ownership of the provided methods and may modify their types
// by setting missing receivers.
func NewInterfaceType(methods []*Func, embeddeds []Type) *Interface {
	if len(methods) == 0 && len(embeddeds) == 0 {
		return &emptyInterface
	}

	// set method receivers if necessary
	typ := new(Interface)
	for _, m := range methods {
		if sig := m.typ.(*Signature); sig.recv == nil {
			sig.recv = NewVar(m.pos, m.pkg, "", typ)
		}
	}

	// sort for API stability
	sortMethods(methods)

	typ.methods = methods
	typ.embeddeds = embeddeds
	typ.complete = true

	return typ
}

// NumExplicitMethods returns the number of explicitly declared methods of interface t.
func (t *Interface) NumExplicitMethods() int { return len(t.methods) }

// ExplicitMethod returns the i'th explicitly declared method of interface t for 0 <= i < t.NumExplicitMethods().
// The methods are ordered by their unique Id.
func (t *Interface) ExplicitMethod(i int) *Func { return t.methods[i] }

// NumEmbeddeds returns the number of embedded types in interface t.
func (t *Interface) NumEmbeddeds() int { return len(t.embeddeds) }

// Embedded returns the i'th embedded defined (*Named) type of interface t for 0 <= i < t.NumEmbeddeds().
// The result is nil if the i'th embedded type is not a defined type.
//
// Deprecated: Use EmbeddedType which is not restricted to defined (*Named) types.
func (t *Interface) Embedded(i int) *Named { tname, _ := t.embeddeds[i].(*Named); return tname }

// EmbeddedType returns the i'th embedded type of interface t for 0 <= i < t.NumEmbeddeds().
func (t *Interface) EmbeddedType(i int) Type { return t.embeddeds[i] }

// NumMethods returns the total number of methods of interface t.
func (t *Interface) NumMethods() int { return t.typeSet().NumMethods() }

// Method returns the i'th method of interface t for 0 <= i < t.NumMethods().
// The methods are ordered by their unique Id.
func (t *Interface) Method(i int) *Func { return t.typeSet().Method(i) }

// Empty reports whether t is the empty interface.
func (t *Interface) Empty() bool { return t.typeSet().IsAll() }

// IsComparable reports whether each type in interface t's type set is comparable.
func (t *Interface) IsComparable() bool { return t.typeSet().IsComparable() }

// IsConstraint reports whether interface t is not just a method set.
func (t *Interface) IsConstraint() bool { return t.typeSet().IsConstraint() }

func (t *Interface) Underlying() Type { return t }
func (t *Interface) String() string   { return TypeString(t, nil) }

// ----------------------------------------------------------------------------
// Implementation

func (check *Checker) interfaceType(ityp *Interface, iface *syntax.InterfaceType, def *Named) {
	var tlist []syntax.Expr // types collected from all type lists
	var tname *syntax.Name  // most recent "type" name

	addEmbedded := func(pos syntax.Pos, typ Type) {
		ityp.embeddeds = append(ityp.embeddeds, typ)
		if ityp.embedPos == nil {
			ityp.embedPos = new([]syntax.Pos)
		}
		*ityp.embedPos = append(*ityp.embedPos, pos)
	}

	for _, f := range iface.MethodList {
		if f.Name == nil {
			// We have an embedded type; possibly a union of types.
			addEmbedded(posFor(f.Type), parseUnion(check, flattenUnion(nil, f.Type)))
			continue
		}
		// f.Name != nil

		// We have a method with name f.Name, or a type of a type list (f.Name.Value == "type").
		name := f.Name.Value
		if name == "_" {
			if check.conf.CompilerErrorMessages {
				check.error(f.Name, "methods must have a unique non-blank name")
			} else {
				check.error(f.Name, "invalid method name _")
			}
			continue // ignore
		}

		// TODO(gri) Remove type list handling once the parser doesn't accept type lists anymore.
		if name == "type" {
			// Report an error for the first type list per interface
			// if we don't allow type lists, but continue.
			if !check.conf.AllowTypeLists && tlist == nil {
				check.softErrorf(f.Name, "use generalized embedding syntax instead of a type list")
			}
			// For now, collect all type list entries as if it
			// were a single union, where each union element is
			// of the form ~T.
			op := new(syntax.Operation)
			// We should also set the position (but there is no setter);
			// we don't care because this code will eventually go away.
			op.Op = syntax.Tilde
			op.X = f.Type
			tlist = append(tlist, op)
			// Report an error if we have multiple type lists in an
			// interface, but only if they are permitted in the first place.
			if check.conf.AllowTypeLists && tname != nil && tname != f.Name {
				check.error(f.Name, "cannot have multiple type lists in an interface")
			}
			tname = f.Name
			continue
		}

		typ := check.typ(f.Type)
		sig, _ := typ.(*Signature)
		if sig == nil {
			if typ != Typ[Invalid] {
				check.errorf(f.Type, invalidAST+"%s is not a method signature", typ)
			}
			continue // ignore
		}

		// Always type-check method type parameters but complain if they are not enabled.
		// (This extra check is needed here because interface method signatures don't have
		// a receiver specification.)
		if sig.tparams != nil && !acceptMethodTypeParams {
			check.error(f.Type, "methods cannot have type parameters")
		}

		// use named receiver type if available (for better error messages)
		var recvTyp Type = ityp
		if def != nil {
			recvTyp = def
		}
		sig.recv = NewVar(f.Name.Pos(), check.pkg, "", recvTyp)

		m := NewFunc(f.Name.Pos(), check.pkg, name, sig)
		check.recordDef(f.Name, m)
		ityp.methods = append(ityp.methods, m)
	}

	// If we saw a type list, add it like an embedded union.
	if tlist != nil {
		// Types T in a type list are added as ~T expressions but we don't
		// have the position of the '~'. Use the first type position instead.
		addEmbedded(tlist[0].(*syntax.Operation).X.Pos(), parseUnion(check, tlist))
	}

	// All methods and embedded elements for this interface are collected;
	// i.e., this interface may be used in a type set computation.
	ityp.complete = true

	if len(ityp.methods) == 0 && len(ityp.embeddeds) == 0 {
		// empty interface
		ityp.tset = &topTypeSet
		return
	}

	// sort for API stability
	// (don't sort embeddeds: they must correspond to *embedPos entries)
	sortMethods(ityp.methods)

	// Compute type set with a non-nil *Checker as soon as possible
	// to report any errors. Subsequent uses of type sets will use
	// this computed type set and won't need to pass in a *Checker.
	//
	// Pin the checker to the interface type in the interim, in case the type set
	// must be used before delayed funcs are processed (see issue #48234).
	// TODO(rfindley): clean up use of *Checker with computeInterfaceTypeSet
	ityp.check = check
	check.later(func() {
		computeInterfaceTypeSet(check, iface.Pos(), ityp)
		ityp.check = nil
	})
}

func flattenUnion(list []syntax.Expr, x syntax.Expr) []syntax.Expr {
	if o, _ := x.(*syntax.Operation); o != nil && o.Op == syntax.Or {
		list = flattenUnion(list, o.X)
		x = o.Y
	}
	return append(list, x)
}
