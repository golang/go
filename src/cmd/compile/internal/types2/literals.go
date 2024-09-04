// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of composite literals.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

func (check *Checker) compositeLit(T *target, x *operand, e *syntax.CompositeLit, hint Type) {
	var typ, base Type
	var isElem bool // true if composite literal is an element of an enclosing composite literal

	switch {
	case e.Type != nil:
		// composite literal type present - use it
		// [...]T array types may only appear with composite literals.
		// Check for them here so we don't have to handle ... in general.
		if atyp, _ := e.Type.(*syntax.ArrayType); atyp != nil && isdddArray(atyp) {
			// We have an "open" [...]T array type.
			// Create a new ArrayType with unknown length (-1)
			// and finish setting it up after analyzing the literal.
			typ = &Array{len: -1, elem: check.varType(atyp.Elem)}
			base = typ
			break
		}
		typ = check.typ(e.Type)
		base = typ

	case hint != nil:
		// no composite literal type present - use hint (element type of enclosing type)
		typ = hint
		base = typ
		// *T implies &T{}
		if b, ok := deref(coreType(base)); ok {
			base = b
		}
		isElem = true

	default:
		// TODO(gri) provide better error messages depending on context
		check.error(e, UntypedLit, "missing type in composite literal")
		x.mode = invalid
		return
	}

	switch utyp := coreType(base).(type) {
	case *Struct:
		// Prevent crash if the struct referred to is not yet set up.
		// See analogous comment for *Array.
		if utyp.fields == nil {
			check.error(e, InvalidTypeCycle, "invalid recursive type")
			x.mode = invalid
			return
		}
		if len(e.ElemList) == 0 {
			break
		}
		// Convention for error messages on invalid struct literals:
		// we mention the struct type only if it clarifies the error
		// (e.g., a duplicate field error doesn't need the struct type).
		fields := utyp.fields
		if _, ok := e.ElemList[0].(*syntax.KeyValueExpr); ok {
			// all elements must have keys
			visited := make([]bool, len(fields))
			for _, e := range e.ElemList {
				kv, _ := e.(*syntax.KeyValueExpr)
				if kv == nil {
					check.error(e, MixedStructLit, "mixture of field:value and value elements in struct literal")
					continue
				}
				key, _ := kv.Key.(*syntax.Name)
				// do all possible checks early (before exiting due to errors)
				// so we don't drop information on the floor
				check.expr(nil, x, kv.Value)
				if key == nil {
					check.errorf(kv, InvalidLitField, "invalid field name %s in struct literal", kv.Key)
					continue
				}
				i := fieldIndex(fields, check.pkg, key.Value, false)
				if i < 0 {
					var alt Object
					if j := fieldIndex(fields, check.pkg, key.Value, true); j >= 0 {
						alt = fields[j]
					}
					msg := check.lookupError(base, key.Value, alt, true)
					check.error(kv.Key, MissingLitField, msg)
					continue
				}
				fld := fields[i]
				check.recordUse(key, fld)
				etyp := fld.typ
				check.assignment(x, etyp, "struct literal")
				// 0 <= i < len(fields)
				if visited[i] {
					check.errorf(kv, DuplicateLitField, "duplicate field name %s in struct literal", key.Value)
					continue
				}
				visited[i] = true
			}
		} else {
			// no element must have a key
			for i, e := range e.ElemList {
				if kv, _ := e.(*syntax.KeyValueExpr); kv != nil {
					check.error(kv, MixedStructLit, "mixture of field:value and value elements in struct literal")
					continue
				}
				check.expr(nil, x, e)
				if i >= len(fields) {
					check.errorf(x, InvalidStructLit, "too many values in struct literal of type %s", base)
					break // cannot continue
				}
				// i < len(fields)
				fld := fields[i]
				if !fld.Exported() && fld.pkg != check.pkg {
					check.errorf(x, UnexportedLitField, "implicit assignment to unexported field %s in struct literal of type %s", fld.name, base)
					continue
				}
				etyp := fld.typ
				check.assignment(x, etyp, "struct literal")
			}
			if len(e.ElemList) < len(fields) {
				check.errorf(inNode(e, e.Rbrace), InvalidStructLit, "too few values in struct literal of type %s", base)
				// ok to continue
			}
		}

	case *Array:
		// Prevent crash if the array referred to is not yet set up. Was go.dev/issue/18643.
		// This is a stop-gap solution. Should use Checker.objPath to report entire
		// path starting with earliest declaration in the source. TODO(gri) fix this.
		if utyp.elem == nil {
			check.error(e, InvalidTypeCycle, "invalid recursive type")
			x.mode = invalid
			return
		}
		n := check.indexedElts(e.ElemList, utyp.elem, utyp.len)
		// If we have an array of unknown length (usually [...]T arrays, but also
		// arrays [n]T where n is invalid) set the length now that we know it and
		// record the type for the array (usually done by check.typ which is not
		// called for [...]T). We handle [...]T arrays and arrays with invalid
		// length the same here because it makes sense to "guess" the length for
		// the latter if we have a composite literal; e.g. for [n]int{1, 2, 3}
		// where n is invalid for some reason, it seems fair to assume it should
		// be 3 (see also Checked.arrayLength and go.dev/issue/27346).
		if utyp.len < 0 {
			utyp.len = n
			// e.Type is missing if we have a composite literal element
			// that is itself a composite literal with omitted type. In
			// that case there is nothing to record (there is no type in
			// the source at that point).
			if e.Type != nil {
				check.recordTypeAndValue(e.Type, typexpr, utyp, nil)
			}
		}

	case *Slice:
		// Prevent crash if the slice referred to is not yet set up.
		// See analogous comment for *Array.
		if utyp.elem == nil {
			check.error(e, InvalidTypeCycle, "invalid recursive type")
			x.mode = invalid
			return
		}
		check.indexedElts(e.ElemList, utyp.elem, -1)

	case *Map:
		// Prevent crash if the map referred to is not yet set up.
		// See analogous comment for *Array.
		if utyp.key == nil || utyp.elem == nil {
			check.error(e, InvalidTypeCycle, "invalid recursive type")
			x.mode = invalid
			return
		}
		// If the map key type is an interface (but not a type parameter),
		// the type of a constant key must be considered when checking for
		// duplicates.
		keyIsInterface := isNonTypeParamInterface(utyp.key)
		visited := make(map[any][]Type, len(e.ElemList))
		for _, e := range e.ElemList {
			kv, _ := e.(*syntax.KeyValueExpr)
			if kv == nil {
				check.error(e, MissingLitKey, "missing key in map literal")
				continue
			}
			check.exprWithHint(x, kv.Key, utyp.key)
			check.assignment(x, utyp.key, "map literal")
			if x.mode == invalid {
				continue
			}
			if x.mode == constant_ {
				duplicate := false
				xkey := keyVal(x.val)
				if keyIsInterface {
					for _, vtyp := range visited[xkey] {
						if Identical(vtyp, x.typ) {
							duplicate = true
							break
						}
					}
					visited[xkey] = append(visited[xkey], x.typ)
				} else {
					_, duplicate = visited[xkey]
					visited[xkey] = nil
				}
				if duplicate {
					check.errorf(x, DuplicateLitKey, "duplicate key %s in map literal", x.val)
					continue
				}
			}
			check.exprWithHint(x, kv.Value, utyp.elem)
			check.assignment(x, utyp.elem, "map literal")
		}

	default:
		// when "using" all elements unpack KeyValueExpr
		// explicitly because check.use doesn't accept them
		for _, e := range e.ElemList {
			if kv, _ := e.(*syntax.KeyValueExpr); kv != nil {
				// Ideally, we should also "use" kv.Key but we can't know
				// if it's an externally defined struct key or not. Going
				// forward anyway can lead to other errors. Give up instead.
				e = kv.Value
			}
			check.use(e)
		}
		// if utyp is invalid, an error was reported before
		if isValid(utyp) {
			var qualifier string
			if isElem {
				qualifier = " element"
			}
			var cause string
			if utyp == nil {
				cause = " (no core type)"
			}
			check.errorf(e, InvalidLit, "invalid composite literal%s type %s%s", qualifier, typ, cause)
			x.mode = invalid
			return
		}
	}

	x.mode = value
	x.typ = typ
}
