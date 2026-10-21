// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// This file checks the M6-S4 spread elements `..src` of struct composite
// literals (SPEC T-104.5). Upstream files call in through hook lines
// only: literals.go (the struct case) and expr.go (misplaced spreads).
// Every rule below applies only to a package of .lgo files
// (Config.Logos), so plain Go keeps its types exactly as upstream. The
// parser only produces *syntax.SpreadExpr in LogosMode.

const logosSpreadMismatch = "[correctness::spread_type_mismatch]"

// LogosSpread records one checked `..src` element for the lowering
// (logos/spread.go): Src is the source expression, Fields names the
// target fields it fills, in field order. Info.LogosSpreads maps each
// spread-bearing composite literal to its spreads in source order.
type LogosSpread struct {
	Src    syntax.Expr
	Fields []string
}

// logosSpreadLiteral checks a struct composite literal holding `..src`
// spread elements (SPEC T-104.5). It reports true when it handled the
// literal: the literal holds at least one top-level SpreadExpr, and the
// whole element list (keyed elements plus spreads) is checked here, so
// the caller returns without running the upstream keyed/positional
// paths. A literal without a top-level spread reports false and the
// caller runs the upstream path unchanged.
//
// A spread copies the source fields whose names match target fields
// with identical types. The same name with a different type fails with
// spread_type_mismatch. Explicit fields win: they are checked like
// upstream keyed elements and suppress the spread copy. The source is
// checked once as an ordinary value expression. Fields neither explicit
// nor spread-copied keep the upstream zero rules, with spread copies
// counting as provided.
func (check *Checker) logosSpreadLiteral(x *operand, e *syntax.CompositeLit, utyp *Struct, typ Type) bool {
	if !check.isLogosFile(e) {
		return false
	}
	var spreads []*syntax.SpreadExpr
	for _, el := range e.ElemList {
		if se, ok := el.(*syntax.SpreadExpr); ok {
			spreads = append(spreads, se)
		}
	}
	if len(spreads) == 0 {
		return false
	}
	ok := true
	provided := make(map[string]bool, len(e.ElemList))
	visited := make(trie[*Var])
	// Check the explicit keyed elements with the upstream rules: every
	// non-spread element must be a keyed field assignment.
	for _, el := range e.ElemList {
		if _, isSpread := el.(*syntax.SpreadExpr); isSpread {
			continue
		}
		kv, isKV := el.(*syntax.KeyValueExpr)
		if !isKV {
			check.error(el, MixedStructLit, "mixture of field:value and value elements in struct literal")
			ok = false
			continue
		}
		key, _ := kv.Key.(*syntax.Name)
		check.genericExpr(x, kv.Value, nil)
		if key == nil {
			check.errorf(kv, InvalidLitField, "invalid field name %s in struct literal", kv.Key)
			ok = false
			continue
		}
		obj, index, indirect := lookupFieldOrMethod(utyp, false, check.pkg, key.Value, false)
		if obj == nil {
			alt, _, _ := lookupFieldOrMethod(utyp, false, check.pkg, key.Value, true)
			msg := check.lookupError(typ, key.Value, alt, true)
			check.error(kv.Key, MissingLitField, msg)
			ok = false
			continue
		}
		fld, _ := obj.(*Var)
		if fld == nil {
			check.errorf(kv.Key, MissingLitField, "%s is not a field", kv.Key)
			ok = false
			continue
		}
		if len(index) > 1 && !check.verifyVersionf(kv.Key, go1_27, "use of promoted field %s in struct literal of type %s", fieldPath(utyp, index), typ) {
			ok = false
			continue
		}
		if indirect {
			check.errorf(kv.Key, InvalidLitField, "invalid implicit pointer indirection to reach %s", kv.Key)
			ok = false
			continue
		}
		check.recordUse(key, fld)
		etyp := fld.typ
		check.assignment(x, etyp, "struct literal")
		provided[fld.name] = true
		if alt, n := visited.insert(index, fld); n != 0 {
			if fld == alt {
				check.errorf(kv, DuplicateLitField, "duplicate field name %s in struct literal", fld.name)
			} else if n < len(index) {
				check.errorf(kv, DuplicateLitField, "cannot specify promoted field %s and enclosing embedded field %s", fld.name, alt.name)
			} else {
				check.errorf(kv, DuplicateLitField, "cannot specify embedded field %s and enclosed promoted field %s", fld.name, alt.name)
			}
			ok = false
		}
	}
	// Check each spread source once and map its copied fields.
	var infos []LogosSpread
	for _, se := range spreads {
		fields := check.logosSpreadSource(utyp, typ, se, provided)
		if fields == nil {
			ok = false
			continue
		}
		for _, name := range fields {
			provided[name] = true
		}
		infos = append(infos, LogosSpread{Src: se.X, Fields: fields})
	}
	// Zero rules count spread copies as provided.
	seen := make(map[Type]bool)
	for _, fld := range utyp.fields {
		if !provided[fld.name] && isValid(fld.typ) && !isTypeParam(fld.typ) && check.logosZeroNeedsInit(fld.typ, seen) {
			check.errorf(e, InvalidLit, "%s field %s.%s (type %s) needs a value", logosNonNilUninit, typ, fld.name, fld.typ)
			ok = false
			break
		}
	}
	if check.Info.LogosSpreads == nil {
		check.Info.LogosSpreads = make(map[*syntax.CompositeLit][]LogosSpread)
	}
	check.Info.LogosSpreads[e] = infos
	if !ok {
		x.invalidate()
		return true
	}
	x.mode_ = value
	x.typ_ = typ
	return true
}

// logosSpreadSource checks one `..src` element against the target struct
// utyp: the source must be a struct (or a pointer to one), and every
// target field neither explicit nor already filled copies the same-named
// source field when the types are identical. The same name with a
// different type fails with spread_type_mismatch. provided marks the
// fields explicit elements and earlier spreads already filled; later
// spreads overwrite earlier ones field by field. It returns the copied
// target field names in field order, or nil on error.
func (check *Checker) logosSpreadSource(utyp *Struct, typ Type, se *syntax.SpreadExpr, provided map[string]bool) []string {
	var v operand
	check.genericExpr(&v, se.X, nil)
	if !v.isValid() || !isValid(v.typ()) {
		return nil
	}
	base := v.typ()
	if u, _ := commonUnder(base, nil); u != nil {
		base = u
	}
	if b, isPtr := deref(base); isPtr {
		base = b
	}
	src, isStruct := base.Underlying().(*Struct)
	if !isStruct {
		check.errorf(se.X, InvalidLit, "spread source must be a struct type, have %s", v.typ())
		return nil
	}
	fields := make([]string, 0, len(utyp.fields))
	for _, fld := range utyp.fields {
		if provided[fld.name] {
			continue
		}
		obj, _, _ := lookupFieldOrMethod(src, false, check.pkg, fld.name, false)
		if obj == nil {
			continue
		}
		srcFld, isVar := obj.(*Var)
		if !isVar {
			continue
		}
		if !srcFld.Exported() && srcFld.pkg != check.pkg {
			continue
		}
		if !fld.Exported() && fld.pkg != check.pkg {
			continue
		}
		if !Identical(srcFld.typ, fld.typ) {
			check.errorf(se.X, InvalidLitField, "%s field %s has type %s in source but type %s in %s", logosSpreadMismatch, fld.name, srcFld.typ, fld.typ, typ)
			return nil
		}
		fields = append(fields, fld.name)
	}
	return fields
}

// logosSpreadMisplaced reports a `..src` spread outside a struct
// composite literal element: only array, slice, and map literals can
// carry one there (struct literals take the logosSpreadLiteral path),
// and none of them accepts a spread.
func (check *Checker) logosSpreadMisplaced(x *operand, e *syntax.SpreadExpr) {
	check.error(e, InvalidSyntaxTree, "spread `..src` is allowed only as a struct composite literal element")
	check.use(e.X)
	x.invalidate()
}
