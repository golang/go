// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package apidiff

import (
	"fmt"
	"go/types"
	"reflect"
)

func (d *differ) checkCompatible(otn *types.TypeName, old, new types.Type) {
	switch old := old.(type) {
	case *types.Interface:
		if new, ok := new.(*types.Interface); ok {
			d.checkCompatibleInterface(otn, old, new)
			return
		}

	case *types.Struct:
		if new, ok := new.(*types.Struct); ok {
			d.checkCompatibleStruct(otn, old, new)
			return
		}

	case *types.Chan:
		if new, ok := new.(*types.Chan); ok {
			d.checkCompatibleChan(otn, old, new)
			return
		}

	case *types.Basic:
		if new, ok := new.(*types.Basic); ok {
			d.checkCompatibleBasic(otn, old, new)
			return
		}

	case *types.Named:
		panic("unreachable")

	default:
		d.checkCorrespondence(otn, "", old, new)
		return

	}
	// Here if old and new are different kinds of types.
	d.typeChanged(otn, "", old, new)
}

func (d *differ) checkCompatibleChan(otn *types.TypeName, old, new *types.Chan) {
	d.checkCorrespondence(otn, ", element type", old.Elem(), new.Elem())
	if old.Dir() != new.Dir() {
		if new.Dir() == types.SendRecv {
			d.compatible(otn, "", "removed direction")
		} else {
			d.incompatible(otn, "", "changed direction")
		}
	}
}

func (d *differ) checkCompatibleBasic(otn *types.TypeName, old, new *types.Basic) {
	// Certain changes to numeric types are compatible. Approximately, the info must
	// be the same, and the new values must be a superset of the old.
	if old.Kind() == new.Kind() {
		// old and new are identical
		return
	}
	if compatibleBasics[[2]types.BasicKind{old.Kind(), new.Kind()}] {
		d.compatible(otn, "", "changed from %s to %s", old, new)
	} else {
		d.typeChanged(otn, "", old, new)
	}
}

// All pairs (old, new) of compatible basic types.
var compatibleBasics = map[[2]types.BasicKind]bool{
	{types.Uint8, types.Uint16}:         true,
	{types.Uint8, types.Uint32}:         true,
	{types.Uint8, types.Uint}:           true,
	{types.Uint8, types.Uint64}:         true,
	{types.Uint16, types.Uint32}:        true,
	{types.Uint16, types.Uint}:          true,
	{types.Uint16, types.Uint64}:        true,
	{types.Uint32, types.Uint}:          true,
	{types.Uint32, types.Uint64}:        true,
	{types.Uint, types.Uint64}:          true,
	{types.Int8, types.Int16}:           true,
	{types.Int8, types.Int32}:           true,
	{types.Int8, types.Int}:             true,
	{types.Int8, types.Int64}:           true,
	{types.Int16, types.Int32}:          true,
	{types.Int16, types.Int}:            true,
	{types.Int16, types.Int64}:          true,
	{types.Int32, types.Int}:            true,
	{types.Int32, types.Int64}:          true,
	{types.Int, types.Int64}:            true,
	{types.Float32, types.Float64}:      true,
	{types.Complex64, types.Complex128}: true,
}

// Interface compatibility:
// If the old interface has an unexported method, the new interface is compatible
// if its exported method set is a superset of the old. (Users could not implement,
// only embed.)
//
// If the old interface did not have an unexported method, the new interface is
// compatible if its exported method set is the same as the old, and it has no
// unexported methods. (Adding an unexported method makes the interface
// unimplementable outside the package.)
//
// TODO: must also check that if any methods were added or removed, every exposed
// type in the package that implemented the interface in old still implements it in
// new. Otherwise external assignments could fail.
func (d *differ) checkCompatibleInterface(otn *types.TypeName, old, new *types.Interface) {
	// Method sets are checked in checkCompatibleDefined.

	// Does the old interface have an unexported method?
	if unexportedMethod(old) != nil {
		d.checkMethodSet(otn, old, new, additionsCompatible)
	} else {
		// Perform an equivalence check, but with more information.
		d.checkMethodSet(otn, old, new, additionsIncompatible)
		if u := unexportedMethod(new); u != nil {
			d.incompatible(otn, u.Name(), "added unexported method")
		}
	}
}

// Return an unexported method from the method set of t, or nil if there are none.
func unexportedMethod(t *types.Interface) *types.Func {
	for i := 0; i < t.NumMethods(); i++ {
		if m := t.Method(i); !m.Exported() {
			return m
		}
	}
	return nil
}

// We need to check three things for structs:
//  1. The set of exported fields must be compatible. This ensures that keyed struct
//     literals continue to compile. (There is no compatibility guarantee for unkeyed
//     struct literals.)
//  2. The set of exported *selectable* fields must be compatible. This includes the exported
//     fields of all embedded structs. This ensures that selections continue to compile.
//  3. If the old struct is comparable, so must the new one be. This ensures that equality
//     expressions and uses of struct values as map keys continue to compile.
//
// An unexported embedded struct can't appear in a struct literal outside the
// package, so it doesn't have to be present, or have the same name, in the new
// struct.
//
// Field tags are ignored: they have no compile-time implications.
func (d *differ) checkCompatibleStruct(obj types.Object, old, new *types.Struct) {
	d.checkCompatibleObjectSets(obj, exportedFields(old), exportedFields(new))
	d.checkCompatibleObjectSets(obj, exportedSelectableFields(old), exportedSelectableFields(new))
	// Removing comparability from a struct is an incompatible change.
	if types.Comparable(old) && !types.Comparable(new) {
		d.incompatible(obj, "", "old is comparable, new is not")
	}
}

// exportedFields collects all the immediate fields of the struct that are exported.
// This is also the set of exported keys for keyed struct literals.
func exportedFields(s *types.Struct) map[string]types.Object {
	m := map[string]types.Object{}
	for i := 0; i < s.NumFields(); i++ {
		f := s.Field(i)
		if f.Exported() {
			m[f.Name()] = f
		}
	}
	return m
}

// exportedSelectableFields collects all the exported fields of the struct, including
// exported fields of embedded structs.
//
// We traverse the struct breadth-first, because of the rule that a lower-depth field
// shadows one at a higher depth.
func exportedSelectableFields(s *types.Struct) map[string]types.Object {
	var (
		m    = map[string]types.Object{}
		next []*types.Struct // embedded structs at the next depth
		seen []*types.Struct // to handle recursive embedding
	)
	for cur := []*types.Struct{s}; len(cur) > 0; cur, next = next, nil {
		seen = append(seen, cur...)
		// We only want to consider unambiguous fields. Ambiguous fields (where there
		// is more than one field of the same name at the same level) are legal, but
		// cannot be selected.
		for name, f := range unambiguousFields(cur) {
			// Record an exported field we haven't seen before. If we have seen it,
			// it occurred a lower depth, so it shadows this field.
			if f.Exported() && m[name] == nil {
				m[name] = f
			}
			// Remember embedded structs for processing at the next depth,
			// but only if we haven't seen the struct at this depth or above.
			if !f.Anonymous() {
				continue
			}
			t := f.Type().Underlying()
			if p, ok := t.(*types.Pointer); ok {
				t = p.Elem().Underlying()
			}
			if t, ok := t.(*types.Struct); ok && !contains(seen, t) {
				next = append(next, t)
			}
		}
	}
	return m
}

func contains(ts []*types.Struct, t *types.Struct) bool {
	for _, s := range ts {
		if types.Identical(s, t) {
			return true
		}
	}
	return false
}

// Given a set of structs at the same depth, the unambiguous fields are the ones whose
// names appear exactly once.
func unambiguousFields(structs []*types.Struct) map[string]*types.Var {
	fields := map[string]*types.Var{}
	seen := map[string]bool{}
	for _, s := range structs {
		for i := 0; i < s.NumFields(); i++ {
			f := s.Field(i)
			name := f.Name()
			if seen[name] {
				delete(fields, name)
			} else {
				seen[name] = true
				fields[name] = f
			}
		}
	}
	return fields
}

// Anything removed or change from the old set is an incompatible change.
// Anything added to the new set is a compatible change.
func (d *differ) checkCompatibleObjectSets(obj types.Object, old, new map[string]types.Object) {
	for name, oldo := range old {
		newo := new[name]
		if newo == nil {
			d.incompatible(obj, name, "removed")
		} else {
			d.checkCorrespondence(obj, name, oldo.Type(), newo.Type())
		}
	}
	for name := range new {
		if old[name] == nil {
			d.compatible(obj, name, "added")
		}
	}
}

func (d *differ) checkCompatibleDefined(otn *types.TypeName, old *types.Named, new types.Type) {
	// We've already checked that old and new correspond.
	d.checkCompatible(otn, old.Underlying(), new.Underlying())
	// If there are different kinds of types (e.g. struct and interface), don't bother checking
	// the method sets.
	if reflect.TypeOf(old.Underlying()) != reflect.TypeOf(new.Underlying()) {
		return
	}
	// Interface method sets are checked in checkCompatibleInterface.
	if _, ok := old.Underlying().(*types.Interface); ok {
		return
	}

	// A new method set is compatible with an old if the new exported methods are a superset of the old.
	d.checkMethodSet(otn, old, new, additionsCompatible)
	d.checkMethodSet(otn, types.NewPointer(old), types.NewPointer(new), additionsCompatible)
}

const (
	additionsCompatible   = true
	additionsIncompatible = false
)

func (d *differ) checkMethodSet(otn *types.TypeName, oldt, newt types.Type, addcompat bool) {
	// TODO: find a way to use checkCompatibleObjectSets for this.
	oldMethodSet := exportedMethods(oldt)
	newMethodSet := exportedMethods(newt)
	msname := otn.Name()
	if _, ok := oldt.(*types.Pointer); ok {
		msname = "*" + msname
	}
	for name, oldMethod := range oldMethodSet {
		newMethod := newMethodSet[name]
		if newMethod == nil {
			var part string
			// Due to embedding, it's possible that the method's receiver type is not
			// the same as the defined type whose method set we're looking at. So for
			// a type T with removed method M that is embedded in some other type U,
			// we will generate two "removed" messages for T.M, one for its own type
			// T and one for the embedded type U. We want both messages to appear,
			// but the messageSet dedup logic will allow only one message for a given
			// object. So use the part string to distinguish them.
			if receiverNamedType(oldMethod).Obj() != otn {
				part = fmt.Sprintf(", method set of %s", msname)
			}
			d.incompatible(oldMethod, part, "removed")
		} else {
			obj := oldMethod
			// If a value method is changed to a pointer method and has a signature
			// change, then we can get two messages for the same method definition: one
			// for the value method set that says it's removed, and another for the
			// pointer method set that says it changed. To keep both messages (since
			// messageSet dedups), use newMethod for the second. (Slight hack.)
			if !hasPointerReceiver(oldMethod) && hasPointerReceiver(newMethod) {
				obj = newMethod
			}
			d.checkCorrespondence(obj, "", oldMethod.Type(), newMethod.Type())
		}
	}

	// Check for added methods.
	for name, newMethod := range newMethodSet {
		if oldMethodSet[name] == nil {
			if addcompat {
				d.compatible(newMethod, "", "added")
			} else {
				d.incompatible(newMethod, "", "added")
			}
		}
	}
}

// exportedMethods collects all the exported methods of type's method set.
func exportedMethods(t types.Type) map[string]types.Object {
	m := map[string]types.Object{}
	ms := types.NewMethodSet(t)
	for i := 0; i < ms.Len(); i++ {
		obj := ms.At(i).Obj()
		if obj.Exported() {
			m[obj.Name()] = obj
		}
	}
	return m
}

func receiverType(method types.Object) types.Type {
	return method.Type().(*types.Signature).Recv().Type()
}

func receiverNamedType(method types.Object) *types.Named {
	switch t := receiverType(method).(type) {
	case *types.Pointer:
		return t.Elem().(*types.Named)
	case *types.Named:
		return t
	default:
		panic("unreachable")
	}
}

func hasPointerReceiver(method types.Object) bool {
	_, ok := receiverType(method).(*types.Pointer)
	return ok
}
