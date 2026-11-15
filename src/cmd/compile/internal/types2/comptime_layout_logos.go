// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
)

func isLogosLayoutName(name string) bool {
	switch name {
	case "SizeOf", "AlignOf", "HasPointers":
		return true
	}
	return false
}

func (check *Checker) logosLayoutCall(x *operand, call *syntax.CallExpr, fname string) bool {
	if !check.logosComptimePosition() {
		return false
	}
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok || len(call.ArgList) != 0 {
		return false
	}
	targs := syntax.UnpackListExpr(idx.Index)
	if len(targs) != 1 {
		return false
	}
	targ := check.varType(targs[0])
	x.expr = call
	if !isValid(targ) {
		x.invalidate()
		return true
	}
	if !check.hasVarSize(targ) && !check.logosGenericFieldType(targs[0]) {
		if fname == "SizeOf" {
			if size := check.conf.sizeof(targ); size < 0 {
				check.errorf(x, TypeTooLarge, "%s is too large", targs[0])
				return true
			}
		}
		x.mode_ = constant_
		x.typ_ = logosLayoutConstantType(fname)
		x.val = check.logosLayoutFolded(targ, fname)
		return true
	}
	if check.logosPred.depth > 0 {
		if check.logosPred.calls == nil {
			check.logosPred.calls = make(map[*syntax.CallExpr]bool)
		}
		check.logosPred.calls[call] = true
		x.mode_ = value
		x.typ_ = logosLayoutDeclaredType(fname)
		return true
	}
	x.mode_ = constant_
	x.typ_ = logosLayoutConstantType(fname)
	x.val = logosLayoutPlaceholder(fname)
	return true
}

func (check *Checker) logosComptimePosition() bool {
	if check.logosPred.depth > 0 || check.logosFieldBind != nil {
		return true
	}
	return check.logosComptimeInGeneric()
}

func logosLayoutPlaceholder(fname string) constant.Value {
	if fname == "HasPointers" {
		return constant.MakeBool(false)
	}
	return constant.MakeInt64(1)
}

func (check *Checker) logosLayoutFolded(targ Type, fname string) constant.Value {
	switch fname {
	case "HasPointers":
		return constant.MakeBool(LogosHasPointers(targ))
	case "SizeOf":
		return constant.MakeInt64(check.conf.sizeof(targ))
	}
	return constant.MakeInt64(check.conf.alignof(targ))
}

func logosLayoutDeclaredType(fname string) Type {
	if fname == "HasPointers" {
		return Typ[Bool]
	}
	return Typ[Int]
}

func logosLayoutConstantType(fname string) Type {
	if fname == "HasPointers" {
		return Typ[UntypedBool]
	}
	return Typ[UntypedInt]
}

func LogosHasPointers(t Type) bool {
	switch u := Unalias(t).Underlying().(type) {
	case *Basic:
		return u.kind == UnsafePointer || u.Info()&IsString != 0
	case *Pointer, *Slice, *Map, *Chan, *Signature, *Interface:
		return true
	case *ConstView:
		return LogosHasPointers(u.Elem)
	case *ReadonlyDescendants:
		return LogosHasPointers(u.Elem)
	case *Option:
		return LogosHasPointers(u.Elem)
	case *LogosTuple:
		return LogosHasPointers(u.under)
	case *Array:
		return u.Len() > 0 && LogosHasPointers(u.Elem())
	case *Struct:
		for i := range u.NumFields() {
			if LogosHasPointers(u.Field(i).Type()) {
				return true
			}
		}
	}
	return false
}
