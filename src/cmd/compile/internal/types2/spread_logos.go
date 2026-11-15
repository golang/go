// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const logosSpreadMismatch = "[correctness::spread_type_mismatch]"

type LogosSpread struct {
	Src    syntax.Expr
	Fields []string
}

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
		if !check.logosLambdaDefer(x, kv.Value) {
			check.genericExpr(x, kv.Value, nil)
		}
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
	var infos []LogosSpread
	readonlySpread := false
	for _, se := range spreads {
		fields, ro := check.logosSpreadSource(utyp, typ, se, provided)
		if fields == nil {
			ok = false
			continue
		}
		readonlySpread = readonlySpread || ro
		for _, name := range fields {
			provided[name] = true
		}
		infos = append(infos, LogosSpread{Src: se.X, Fields: fields})
	}
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
	if readonlySpread {
		x.typ_ = logosWithReadonlyDescendants(typ)
	} else {
		x.typ_ = typ
	}
	return true
}

func (check *Checker) logosSpreadSource(utyp *Struct, typ Type, se *syntax.SpreadExpr, provided map[string]bool) ([]string, bool) {
	var v operand
	check.genericExpr(&v, se.X, nil)
	if !v.isValid() || !isValid(v.typ()) {
		return nil, false
	}
	check.logosConstOrigin(&v)
	base, descendants, readonly := logosReadonlyBase(v.typ())
	readonlySource := readonly || descendants
	if u, _ := commonUnder(base, nil); u != nil {
		base = u
	}
	if b, isPtr := deref(base); isPtr {
		base, descendants, readonly = logosReadonlyBase(b)
		readonlySource = readonlySource || readonly || descendants
	}
	src, isStruct := base.Underlying().(*Struct)
	if !isStruct {
		check.errorf(se.X, InvalidLit, "spread source must be a struct type, have %s", v.typ())
		return nil, false
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
			return nil, false
		}
		fields = append(fields, fld.name)
	}
	return fields, readonlySource && len(fields) > 0
}

func (check *Checker) logosSpreadMisplaced(x *operand, e *syntax.SpreadExpr) {
	check.error(e, InvalidSyntaxTree, "spread `..src` is allowed only as a struct composite literal element")
	check.use(e.X)
	x.invalidate()
}
