// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements target types for expressions in assignment contexts.

package types2

// target represents the target type in an assignment context.
type target struct {
	typ Type
	// TODO(gri) desc is only used in Checker.funcInst - do we need it?
	//           (setting/computing desc may be expensive for not being used)
	//           Also, if we keep it, review consistency of description.
	desc  string
	hint  bool // if set, the target may be used as untyped composite literal hint before Go 1.28
	isVar bool // target is a variable (incl. parameter or result); see Checker.funcInst
}

// newTarget creates a new target for the given type and description.
// The result is nil if typ is nil.
func newTarget(typ Type, desc string) *target {
	if typ != nil {
		return &target{typ, desc, false, false}
	}
	return nil
}

// newHint is like newTarget but it marks the result as a hint.
func newHint(typ Type, desc string) *target {
	if typ != nil {
		return &target{typ, desc, true, false}
	}
	return nil
}

// newVarTarget is like newTarget but it marks the result as a variable.
func newVarTarget(typ Type, desc string) *target {
	if typ != nil {
		return &target{typ, desc, false, true}
	}
	return nil
}

// sig returns the target type T if it is a signature (the result may be its Named or Alias type, if any).
// The result is nil if T is nil or T's type set has no common signature type.
func (T *target) sig() Type {
	if T != nil {
		if u, _ := commonUnder(T.typ, nil); u != nil {
			if _, ok := u.(*Signature); ok {
				return T.typ
			}
		}
	}
	return nil
}

// String returns a string representation of T, for debugging.
func (T *target) String() string {
	if T != nil {
		return sprintf(nil, true, "target{%s, %q}", T.typ, T.desc)
	}
	return "no target"
}
