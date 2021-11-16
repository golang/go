// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"go/token"
	"sort"
)

// ----------------------------------------------------------------------------
// API

// A _TypeSet represents the type set of an interface.
type _TypeSet struct {
	comparable bool // if set, the interface is or embeds comparable
	// TODO(gri) consider using a set for the methods for faster lookup
	methods []*Func  // all methods of the interface; sorted by unique ID
	terms   termlist // type terms of the type set
}

// IsEmpty reports whether type set s is the empty set.
func (s *_TypeSet) IsEmpty() bool { return s.terms.isEmpty() }

// IsAll reports whether type set s is the set of all types (corresponding to the empty interface).
func (s *_TypeSet) IsAll() bool { return !s.comparable && len(s.methods) == 0 && s.terms.isAll() }

// IsMethodSet reports whether the interface t is fully described by its method set.
func (s *_TypeSet) IsMethodSet() bool { return !s.comparable && s.terms.isAll() }

// IsComparable reports whether each type in the set is comparable.
func (s *_TypeSet) IsComparable() bool {
	if s.terms.isAll() {
		return s.comparable
	}
	return s.is(func(t *term) bool {
		return t != nil && Comparable(t.typ)
	})
}

// TODO(gri) IsTypeSet is not a great name for this predicate. Find a better one.

// IsTypeSet reports whether the type set s is represented by a finite set of underlying types.
func (s *_TypeSet) IsTypeSet() bool {
	return !s.comparable && len(s.methods) == 0
}

// NumMethods returns the number of methods available.
func (s *_TypeSet) NumMethods() int { return len(s.methods) }

// Method returns the i'th method of type set s for 0 <= i < s.NumMethods().
// The methods are ordered by their unique ID.
func (s *_TypeSet) Method(i int) *Func { return s.methods[i] }

// LookupMethod returns the index of and method with matching package and name, or (-1, nil).
func (s *_TypeSet) LookupMethod(pkg *Package, name string) (int, *Func) {
	// TODO(gri) s.methods is sorted - consider binary search
	return lookupMethod(s.methods, pkg, name)
}

func (s *_TypeSet) String() string {
	switch {
	case s.IsEmpty():
		return "∅"
	case s.IsAll():
		return "𝓤"
	}

	hasMethods := len(s.methods) > 0
	hasTerms := s.hasTerms()

	var buf bytes.Buffer
	buf.WriteByte('{')
	if s.comparable {
		buf.WriteString("comparable")
		if hasMethods || hasTerms {
			buf.WriteString("; ")
		}
	}
	for i, m := range s.methods {
		if i > 0 {
			buf.WriteString("; ")
		}
		buf.WriteString(m.String())
	}
	if hasMethods && hasTerms {
		buf.WriteString("; ")
	}
	if hasTerms {
		buf.WriteString(s.terms.String())
	}
	buf.WriteString("}")
	return buf.String()
}

// ----------------------------------------------------------------------------
// Implementation

// hasTerms reports whether the type set has specific type terms.
func (s *_TypeSet) hasTerms() bool { return !s.terms.isEmpty() && !s.terms.isAll() }

// singleType returns the single type in s if there is exactly one; otherwise the result is nil.
func (s *_TypeSet) singleType() Type { return s.terms.singleType() }

// includes reports whether t ∈ s.
func (s *_TypeSet) includes(t Type) bool { return s.terms.includes(t) }

// subsetOf reports whether s1 ⊆ s2.
func (s1 *_TypeSet) subsetOf(s2 *_TypeSet) bool { return s1.terms.subsetOf(s2.terms) }

// TODO(gri) TypeSet.is and TypeSet.underIs should probably also go into termlist.go

// is calls f with the specific type terms of s and reports whether
// all calls to f returned true. If there are no specific terms, is
// returns the result of f(nil).
func (s *_TypeSet) is(f func(*term) bool) bool {
	if !s.hasTerms() {
		return f(nil)
	}
	for _, t := range s.terms {
		assert(t.typ != nil)
		if !f(t) {
			return false
		}
	}
	return true
}

// underIs calls f with the underlying types of the specific type terms
// of s and reports whether all calls to f returned true. If there are
// no specific terms, is returns the result of f(nil).
func (s *_TypeSet) underIs(f func(Type) bool) bool {
	if !s.hasTerms() {
		return f(nil)
	}
	for _, t := range s.terms {
		assert(t.typ != nil)
		// x == under(x) for ~x terms
		u := t.typ
		if !t.tilde {
			u = under(u)
		}
		if debug {
			assert(Identical(u, under(u)))
		}
		if !f(u) {
			return false
		}
	}
	return true
}

// topTypeSet may be used as type set for the empty interface.
var topTypeSet = _TypeSet{terms: allTermlist}

// computeInterfaceTypeSet may be called with check == nil.
func computeInterfaceTypeSet(check *Checker, pos token.Pos, ityp *Interface) *_TypeSet {
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
	ityp.tset = &_TypeSet{terms: allTermlist} // TODO(gri) is this sufficient?

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
				if !check.allowVersion(m.pkg, 1, 14) || !Identical(m.typ, other.Type()) {
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
	var allTerms = allTermlist
	for i, typ := range ityp.embeddeds {
		// The embedding position is nil for imported interfaces
		// and also for interface copies after substitution (but
		// in that case we don't need to report errors again).
		var pos token.Pos // embedding position
		if ityp.embedPos != nil {
			pos = (*ityp.embedPos)[i]
		}
		var terms termlist
		switch u := under(typ).(type) {
		case *Interface:
			tset := computeInterfaceTypeSet(check, pos, u)
			// If typ is local, an error was already reported where typ is specified/defined.
			if check != nil && check.isImportedConstraint(typ) && !check.allowVersion(check.pkg, 1, 18) {
				check.errorf(atPos(pos), _UnsupportedFeature, "embedding constraint interface %s requires go1.18 or later", typ)
				continue
			}
			if tset.comparable {
				ityp.tset.comparable = true
			}
			for _, m := range tset.methods {
				addMethod(pos, m, false) // use embedding position pos rather than m.pos
			}
			terms = tset.terms
		case *Union:
			if check != nil && !check.allowVersion(check.pkg, 1, 18) {
				check.errorf(atPos(pos), _InvalidIfaceEmbed, "embedding interface element %s requires go1.18 or later", u)
				continue
			}
			tset := computeUnionTypeSet(check, pos, u)
			if tset == &invalidTypeSet {
				continue // ignore invalid unions
			}
			terms = tset.terms
		default:
			if u == Typ[Invalid] {
				continue
			}
			if check != nil && !check.allowVersion(check.pkg, 1, 18) {
				check.errorf(atPos(pos), _InvalidIfaceEmbed, "embedding non-interface type %s requires go1.18 or later", typ)
				continue
			}
			terms = termlist{{false, typ}}
		}
		// The type set of an interface is the intersection
		// of the type sets of all its elements.
		// Intersection cannot produce longer termlists and
		// thus cannot overflow.
		allTerms = allTerms.intersect(terms)
	}
	ityp.embedPos = nil // not needed anymore (errors have been reported)

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
	ityp.tset.terms = allTerms

	return ityp.tset
}

func sortMethods(list []*Func) {
	sort.Sort(byUniqueMethodName(list))
}

func assertSortedMethods(list []*Func) {
	if !debug {
		panic("assertSortedMethods called outside debug mode")
	}
	if !sort.IsSorted(byUniqueMethodName(list)) {
		panic("methods not sorted")
	}
}

// byUniqueMethodName method lists can be sorted by their unique method names.
type byUniqueMethodName []*Func

func (a byUniqueMethodName) Len() int           { return len(a) }
func (a byUniqueMethodName) Less(i, j int) bool { return a[i].Id() < a[j].Id() }
func (a byUniqueMethodName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// invalidTypeSet is a singleton type set to signal an invalid type set
// due to an error. It's also a valid empty type set, so consumers of
// type sets may choose to ignore it.
var invalidTypeSet _TypeSet

// computeUnionTypeSet may be called with check == nil.
// The result is &invalidTypeSet if the union overflows.
func computeUnionTypeSet(check *Checker, pos token.Pos, utyp *Union) *_TypeSet {
	if utyp.tset != nil {
		return utyp.tset
	}

	// avoid infinite recursion (see also computeInterfaceTypeSet)
	utyp.tset = new(_TypeSet)

	var allTerms termlist
	for _, t := range utyp.terms {
		var terms termlist
		switch u := under(t.typ).(type) {
		case *Interface:
			terms = computeInterfaceTypeSet(check, pos, u).terms
		default:
			if t.typ == Typ[Invalid] {
				continue
			}
			terms = termlist{(*term)(t)}
		}
		// The type set of a union expression is the union
		// of the type sets of each term.
		allTerms = allTerms.union(terms)
		if len(allTerms) > maxTermCount {
			if check != nil {
				check.errorf(atPos(pos), _InvalidUnion, "cannot handle more than %d union terms (implementation limitation)", maxTermCount)
			}
			utyp.tset = &invalidTypeSet
			return utyp.tset
		}
	}
	utyp.tset.terms = allTerms

	return utyp.tset
}
