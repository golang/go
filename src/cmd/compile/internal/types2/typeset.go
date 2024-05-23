// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"sort"
	"strings"
)

// ----------------------------------------------------------------------------
// API

// A _TypeSet represents the type set of an interface.
// Because of existing language restrictions, methods can be "factored out"
// from the terms. The actual type set is the intersection of the type set
// implied by the methods and the type set described by the terms and the
// comparable bit. To test whether a type is included in a type set
// ("implements" relation), the type must implement all methods _and_ be
// an element of the type set described by the terms and the comparable bit.
// If the term list describes the set of all types and comparable is true,
// only comparable types are meant; in all other cases comparable is false.
type _TypeSet struct {
	methods    []*Func  // all methods of the interface; sorted by unique ID
	terms      termlist // type terms of the type set
	comparable bool     // invariant: !comparable || terms.isAll()
}

// IsEmpty reports whether type set s is the empty set.
func (s *_TypeSet) IsEmpty() bool { return s.terms.isEmpty() }

// IsAll reports whether type set s is the set of all types (corresponding to the empty interface).
func (s *_TypeSet) IsAll() bool { return s.IsMethodSet() && len(s.methods) == 0 }

// IsMethodSet reports whether the interface t is fully described by its method set.
func (s *_TypeSet) IsMethodSet() bool { return !s.comparable && s.terms.isAll() }

// IsComparable reports whether each type in the set is comparable.
func (s *_TypeSet) IsComparable(seen map[Type]bool) bool {
	if s.terms.isAll() {
		return s.comparable
	}
	return s.is(func(t *term) bool {
		return t != nil && comparableType(t.typ, false, seen, nil)
	})
}

// NumMethods returns the number of methods available.
func (s *_TypeSet) NumMethods() int { return len(s.methods) }

// Method returns the i'th method of type set s for 0 <= i < s.NumMethods().
// The methods are ordered by their unique ID.
func (s *_TypeSet) Method(i int) *Func { return s.methods[i] }

// LookupMethod returns the index of and method with matching package and name, or (-1, nil).
func (s *_TypeSet) LookupMethod(pkg *Package, name string, foldCase bool) (int, *Func) {
	return methodIndex(s.methods, pkg, name, foldCase)
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

	var buf strings.Builder
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
// no specific terms, underIs returns the result of f(nil).
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
func computeInterfaceTypeSet(check *Checker, pos syntax.Pos, ityp *Interface) *_TypeSet {
	if ityp.tset != nil {
		return ityp.tset
	}

	// If the interface is not fully set up yet, the type set will
	// not be complete, which may lead to errors when using the
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

	if check != nil && check.conf.Trace {
		// Types don't generally have position information.
		// If we don't have a valid pos provided, try to use
		// one close enough.
		if !pos.IsKnown() && len(ityp.methods) > 0 {
			pos = ityp.methods[0].pos
		}

		check.trace(pos, "-- type set for %s", ityp)
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

	var unionSets map[*Union]*_TypeSet
	if check != nil {
		if check.unionTypeSets == nil {
			check.unionTypeSets = make(map[*Union]*_TypeSet)
		}
		unionSets = check.unionTypeSets
	} else {
		unionSets = make(map[*Union]*_TypeSet)
	}

	// Methods of embedded interfaces are collected unchanged; i.e., the identity
	// of a method I.m's Func Object of an interface I is the same as that of
	// the method m in an interface that embeds interface I. On the other hand,
	// if a method is embedded via multiple overlapping embedded interfaces, we
	// don't provide a guarantee which "original m" got chosen for the embedding
	// interface. See also go.dev/issue/34421.
	//
	// If we don't care to provide this identity guarantee anymore, instead of
	// reusing the original method in embeddings, we can clone the method's Func
	// Object and give it the position of a corresponding embedded interface. Then
	// we can get rid of the mpos map below and simply use the cloned method's
	// position.

	var seen objset
	var allMethods []*Func
	mpos := make(map[*Func]syntax.Pos) // method specification or method embedding position, for good error messages
	addMethod := func(pos syntax.Pos, m *Func, explicit bool) {
		switch other := seen.insert(m); {
		case other == nil:
			allMethods = append(allMethods, m)
			mpos[m] = pos
		case explicit:
			if check != nil {
				err := check.newError(DuplicateDecl)
				err.addf(atPos(pos), "duplicate method %s", m.name)
				err.addf(atPos(mpos[other.(*Func)]), "other declaration of method %s", m.name)
				err.report()
			}
		default:
			// We have a duplicate method name in an embedded (not explicitly declared) method.
			// Check method signatures after all types are computed (go.dev/issue/33656).
			// If we're pre-go1.14 (overlapping embeddings are not permitted), report that
			// error here as well (even though we could do it eagerly) because it's the same
			// error message.
			if check != nil {
				check.later(func() {
					if pos.IsKnown() && !check.allowVersion(atPos(pos), go1_14) || !Identical(m.typ, other.Type()) {
						err := check.newError(DuplicateDecl)
						err.addf(atPos(pos), "duplicate method %s", m.name)
						err.addf(atPos(mpos[other.(*Func)]), "other declaration of method %s", m.name)
						err.report()
					}
				}).describef(atPos(pos), "duplicate method check for %s", m.name)
			}
		}
	}

	for _, m := range ityp.methods {
		addMethod(m.pos, m, true)
	}

	// collect embedded elements
	allTerms := allTermlist
	allComparable := false
	for i, typ := range ityp.embeddeds {
		// The embedding position is nil for imported interfaces.
		// We don't need to do version checks in those cases.
		var pos syntax.Pos // embedding position
		if ityp.embedPos != nil {
			pos = (*ityp.embedPos)[i]
		}
		var comparable bool
		var terms termlist
		switch u := under(typ).(type) {
		case *Interface:
			// For now we don't permit type parameters as constraints.
			assert(!isTypeParam(typ))
			tset := computeInterfaceTypeSet(check, pos, u)
			// If typ is local, an error was already reported where typ is specified/defined.
			if pos.IsKnown() && check != nil && check.isImportedConstraint(typ) && !check.verifyVersionf(atPos(pos), go1_18, "embedding constraint interface %s", typ) {
				continue
			}
			comparable = tset.comparable
			for _, m := range tset.methods {
				addMethod(pos, m, false) // use embedding position pos rather than m.pos
			}
			terms = tset.terms
		case *Union:
			if pos.IsKnown() && check != nil && !check.verifyVersionf(atPos(pos), go1_18, "embedding interface element %s", u) {
				continue
			}
			tset := computeUnionTypeSet(check, unionSets, pos, u)
			if tset == &invalidTypeSet {
				continue // ignore invalid unions
			}
			assert(!tset.comparable)
			assert(len(tset.methods) == 0)
			terms = tset.terms
		default:
			if !isValid(u) {
				continue
			}
			if pos.IsKnown() && check != nil && !check.verifyVersionf(atPos(pos), go1_18, "embedding non-interface type %s", typ) {
				continue
			}
			terms = termlist{{false, typ}}
		}

		// The type set of an interface is the intersection of the type sets of all its elements.
		// Due to language restrictions, only embedded interfaces can add methods, they are handled
		// separately. Here we only need to intersect the term lists and comparable bits.
		allTerms, allComparable = intersectTermLists(allTerms, allComparable, terms, comparable)
	}

	ityp.tset.comparable = allComparable
	if len(allMethods) != 0 {
		sortMethods(allMethods)
		ityp.tset.methods = allMethods
	}
	ityp.tset.terms = allTerms

	return ityp.tset
}

// TODO(gri) The intersectTermLists function belongs to the termlist implementation.
//           The comparable type set may also be best represented as a term (using
//           a special type).

// intersectTermLists computes the intersection of two term lists and respective comparable bits.
// xcomp, ycomp are valid only if xterms.isAll() and yterms.isAll() respectively.
func intersectTermLists(xterms termlist, xcomp bool, yterms termlist, ycomp bool) (termlist, bool) {
	terms := xterms.intersect(yterms)
	// If one of xterms or yterms is marked as comparable,
	// the result must only include comparable types.
	comp := xcomp || ycomp
	if comp && !terms.isAll() {
		// only keep comparable terms
		i := 0
		for _, t := range terms {
			assert(t.typ != nil)
			if comparableType(t.typ, false /* strictly comparable */, nil, nil) {
				terms[i] = t
				i++
			}
		}
		terms = terms[:i]
		if !terms.isAll() {
			comp = false
		}
	}
	assert(!comp || terms.isAll()) // comparable invariant
	return terms, comp
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
func (a byUniqueMethodName) Less(i, j int) bool { return a[i].less(&a[j].object) }
func (a byUniqueMethodName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// invalidTypeSet is a singleton type set to signal an invalid type set
// due to an error. It's also a valid empty type set, so consumers of
// type sets may choose to ignore it.
var invalidTypeSet _TypeSet

// computeUnionTypeSet may be called with check == nil.
// The result is &invalidTypeSet if the union overflows.
func computeUnionTypeSet(check *Checker, unionSets map[*Union]*_TypeSet, pos syntax.Pos, utyp *Union) *_TypeSet {
	if tset, _ := unionSets[utyp]; tset != nil {
		return tset
	}

	// avoid infinite recursion (see also computeInterfaceTypeSet)
	unionSets[utyp] = new(_TypeSet)

	var allTerms termlist
	for _, t := range utyp.terms {
		var terms termlist
		u := under(t.typ)
		if ui, _ := u.(*Interface); ui != nil {
			// For now we don't permit type parameters as constraints.
			assert(!isTypeParam(t.typ))
			terms = computeInterfaceTypeSet(check, pos, ui).terms
		} else if !isValid(u) {
			continue
		} else {
			if t.tilde && !Identical(t.typ, u) {
				// There is no underlying type which is t.typ.
				// The corresponding type set is empty.
				t = nil // ∅ term
			}
			terms = termlist{(*term)(t)}
		}
		// The type set of a union expression is the union
		// of the type sets of each term.
		allTerms = allTerms.union(terms)
		if len(allTerms) > maxTermCount {
			if check != nil {
				check.errorf(atPos(pos), InvalidUnion, "cannot handle more than %d union terms (implementation limitation)", maxTermCount)
			}
			unionSets[utyp] = &invalidTypeSet
			return unionSets[utyp]
		}
	}
	unionSets[utyp].terms = allTerms

	return unionSets[utyp]
}
