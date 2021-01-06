// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package apidiff

import (
	"go/types"
	"sort"
)

// Two types are correspond if they are identical except for defined types,
// which must correspond.
//
// Two defined types correspond if they can be interchanged in the old and new APIs,
// possibly after a renaming.
//
// This is not a pure function. If we come across named types while traversing,
// we establish correspondence.
func (d *differ) correspond(old, new types.Type) bool {
	return d.corr(old, new, nil)
}

// corr determines whether old and new correspond. The argument p is a list of
// known interface identities, to avoid infinite recursion.
//
// corr calls itself recursively as much as possible, to establish more
// correspondences and so check more of the API. E.g. if the new function has more
// parameters than the old, compare all the old ones before returning false.
//
// Compare this to the implementation of go/types.Identical.
func (d *differ) corr(old, new types.Type, p *ifacePair) bool {
	// Structure copied from types.Identical.
	switch old := old.(type) {
	case *types.Basic:
		return types.Identical(old, new)

	case *types.Array:
		if new, ok := new.(*types.Array); ok {
			return d.corr(old.Elem(), new.Elem(), p) && old.Len() == new.Len()
		}

	case *types.Slice:
		if new, ok := new.(*types.Slice); ok {
			return d.corr(old.Elem(), new.Elem(), p)
		}

	case *types.Map:
		if new, ok := new.(*types.Map); ok {
			return d.corr(old.Key(), new.Key(), p) && d.corr(old.Elem(), new.Elem(), p)
		}

	case *types.Chan:
		if new, ok := new.(*types.Chan); ok {
			return d.corr(old.Elem(), new.Elem(), p) && old.Dir() == new.Dir()
		}

	case *types.Pointer:
		if new, ok := new.(*types.Pointer); ok {
			return d.corr(old.Elem(), new.Elem(), p)
		}

	case *types.Signature:
		if new, ok := new.(*types.Signature); ok {
			pe := d.corr(old.Params(), new.Params(), p)
			re := d.corr(old.Results(), new.Results(), p)
			return old.Variadic() == new.Variadic() && pe && re
		}

	case *types.Tuple:
		if new, ok := new.(*types.Tuple); ok {
			for i := 0; i < old.Len(); i++ {
				if i >= new.Len() || !d.corr(old.At(i).Type(), new.At(i).Type(), p) {
					return false
				}
			}
			return old.Len() == new.Len()
		}

	case *types.Struct:
		if new, ok := new.(*types.Struct); ok {
			for i := 0; i < old.NumFields(); i++ {
				if i >= new.NumFields() {
					return false
				}
				of := old.Field(i)
				nf := new.Field(i)
				if of.Anonymous() != nf.Anonymous() ||
					old.Tag(i) != new.Tag(i) ||
					!d.corr(of.Type(), nf.Type(), p) ||
					!d.corrFieldNames(of, nf) {
					return false
				}
			}
			return old.NumFields() == new.NumFields()
		}

	case *types.Interface:
		if new, ok := new.(*types.Interface); ok {
			// Deal with circularity. See the comment in types.Identical.
			q := &ifacePair{old, new, p}
			for p != nil {
				if p.identical(q) {
					return true // same pair was compared before
				}
				p = p.prev
			}
			oldms := d.sortedMethods(old)
			newms := d.sortedMethods(new)
			for i, om := range oldms {
				if i >= len(newms) {
					return false
				}
				nm := newms[i]
				if d.methodID(om) != d.methodID(nm) || !d.corr(om.Type(), nm.Type(), q) {
					return false
				}
			}
			return old.NumMethods() == new.NumMethods()
		}

	case *types.Named:
		if new, ok := new.(*types.Named); ok {
			return d.establishCorrespondence(old, new)
		}
		if new, ok := new.(*types.Basic); ok {
			// Basic types are defined types, too, so we have to support them.

			return d.establishCorrespondence(old, new)
		}

	default:
		panic("unknown type kind")
	}
	return false
}

// Compare old and new field names. We are determining correspondence across packages,
// so just compare names, not packages. For an unexported, embedded field of named
// type (non-named embedded fields are possible with aliases), we check that the type
// names correspond. We check the types for correspondence before this is called, so
// we've established correspondence.
func (d *differ) corrFieldNames(of, nf *types.Var) bool {
	if of.Anonymous() && nf.Anonymous() && !of.Exported() && !nf.Exported() {
		if on, ok := of.Type().(*types.Named); ok {
			nn := nf.Type().(*types.Named)
			return d.establishCorrespondence(on, nn)
		}
	}
	return of.Name() == nf.Name()
}

// Establish that old corresponds with new if it does not already
// correspond to something else.
func (d *differ) establishCorrespondence(old *types.Named, new types.Type) bool {
	oldname := old.Obj()
	oldc := d.correspondMap[oldname]
	if oldc == nil {
		// For now, assume the types don't correspond unless they are from the old
		// and new packages, respectively.
		//
		// This is too conservative. For instance,
		//    [old] type A = q.B; [new] type A q.C
		// could be OK if in package q, B is an alias for C.
		// Or, using p as the name of the current old/new packages:
		//    [old] type A = q.B; [new] type A int
		// could be OK if in q,
		//    [old] type B int; [new] type B = p.A
		// In this case, p.A and q.B name the same type in both old and new worlds.
		// Note that this case doesn't imply circular package imports: it's possible
		// that in the old world, p imports q, but in the new, q imports p.
		//
		// However, if we didn't do something here, then we'd incorrectly allow cases
		// like the first one above in which q.B is not an alias for q.C
		//
		// What we should do is check that the old type, in the new world's package
		// of the same path, doesn't correspond to something other than the new type.
		// That is a bit hard, because there is no easy way to find a new package
		// matching an old one.
		if newn, ok := new.(*types.Named); ok {
			if old.Obj().Pkg() != d.old || newn.Obj().Pkg() != d.new {
				return old.Obj().Id() == newn.Obj().Id()
			}
		}
		// If there is no correspondence, create one.
		d.correspondMap[oldname] = new
		// Check that the corresponding types are compatible.
		d.checkCompatibleDefined(oldname, old, new)
		return true
	}
	return types.Identical(oldc, new)
}

func (d *differ) sortedMethods(iface *types.Interface) []*types.Func {
	ms := make([]*types.Func, iface.NumMethods())
	for i := 0; i < iface.NumMethods(); i++ {
		ms[i] = iface.Method(i)
	}
	sort.Slice(ms, func(i, j int) bool { return d.methodID(ms[i]) < d.methodID(ms[j]) })
	return ms
}

func (d *differ) methodID(m *types.Func) string {
	// If the method belongs to one of the two packages being compared, use
	// just its name even if it's unexported. That lets us treat unexported names
	// from the old and new packages as equal.
	if m.Pkg() == d.old || m.Pkg() == d.new {
		return m.Name()
	}
	return m.Id()
}

// Copied from the go/types package:

// An ifacePair is a node in a stack of interface type pairs compared for identity.
type ifacePair struct {
	x, y *types.Interface
	prev *ifacePair
}

func (p *ifacePair) identical(q *ifacePair) bool {
	return p.x == q.x && p.y == q.y || p.x == q.y && p.y == q.x
}
