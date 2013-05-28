// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"

	"code.google.com/p/go.tools/go/exact"
)

// TODO(gri) initialize is very close to the 2nd half of assign1to1.
func (check *checker) assign(obj Object, x *operand) {
	// Determine typ of lhs: If the object doesn't have a type
	// yet, determine it from the type of x; if x is invalid,
	// set the object type to Typ[Invalid].
	var typ Type
	switch obj := obj.(type) {
	default:
		unreachable()

	case *Const:
		typ = obj.typ // may already be Typ[Invalid]
		if typ == nil {
			typ = Typ[Invalid]
			if x.mode != invalid {
				typ = x.typ
			}
			obj.typ = typ
		}

	case *Var:
		typ = obj.typ // may already be Typ[Invalid]
		if typ == nil {
			typ = Typ[Invalid]
			if x.mode != invalid {
				typ = x.typ
				if isUntyped(typ) {
					// convert untyped types to default types
					if typ == Typ[UntypedNil] {
						check.errorf(x.pos(), "use of untyped nil")
						typ = Typ[Invalid]
					} else {
						typ = defaultType(typ)
					}
				}
			}
			obj.typ = typ
		}
	}

	// nothing else to check if we don't have a valid lhs or rhs
	if typ == Typ[Invalid] || x.mode == invalid {
		return
	}

	if !check.assignment(x, typ) {
		if x.mode != invalid {
			if x.typ != Typ[Invalid] && typ != Typ[Invalid] {
				check.errorf(x.pos(), "cannot initialize %s (type %s) with %s", obj.Name(), typ, x)
			}
		}
		return
	}

	// for constants, set their value
	if obj, _ := obj.(*Const); obj != nil {
		obj.val = exact.MakeUnknown() // failure case: we don't know the constant value
		if x.mode == constant {
			if isConstType(x.typ) {
				obj.val = x.val
			} else if x.typ != Typ[Invalid] {
				check.errorf(x.pos(), "%s has invalid constant type", x)
			}
		} else if x.mode != invalid {
			check.errorf(x.pos(), "%s is not constant", x)
		}
	}
}

func (check *checker) assignMulti(lhs []Object, rhs []ast.Expr) {
	assert(len(lhs) > 0)

	const decl = false

	// If the lhs and rhs have corresponding expressions, treat each
	// matching pair as an individual pair.
	if len(lhs) == len(rhs) {
		var x operand
		for i, e := range rhs {
			check.expr(&x, e, nil, -1)
			if x.mode == invalid {
				goto Error
			}
			check.assign(lhs[i], &x)
		}
		return
	}

	// Otherwise, the rhs must be a single expression (possibly
	// a function call returning multiple values, or a comma-ok
	// expression).
	if len(rhs) == 1 {
		// len(lhs) > 1
		// Start with rhs so we have expression types
		// for declarations with implicit types.
		var x operand
		check.expr(&x, rhs[0], nil, -1)
		if x.mode == invalid {
			goto Error
		}

		if t, ok := x.typ.(*Tuple); ok && len(lhs) == t.Len() {
			// function result
			x.mode = value
			for i := 0; i < len(lhs); i++ {
				obj := t.At(i)
				x.expr = nil // TODO(gri) should do better here
				x.typ = obj.typ
				check.assign(lhs[i], &x)
			}
			return
		}

		if x.mode == valueok && len(lhs) == 2 {
			// comma-ok expression
			x.mode = value
			check.assign(lhs[0], &x)

			x.typ = Typ[UntypedBool]
			check.assign(lhs[1], &x)
			return
		}
	}

	check.errorf(lhs[0].Pos(), "assignment count mismatch: %d = %d", len(lhs), len(rhs))

Error:
	// In case of a declaration, set all lhs types to Typ[Invalid].
	for _, obj := range lhs {
		switch obj := obj.(type) {
		case *Const:
			if obj.typ == nil {
				obj.typ = Typ[Invalid]
			}
			obj.val = exact.MakeUnknown()
		case *Var:
			if obj.typ == nil {
				obj.typ = Typ[Invalid]
			}
		default:
			unreachable()
		}
	}
}
