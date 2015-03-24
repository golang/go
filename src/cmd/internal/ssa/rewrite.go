// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/ssa/types" // TODO: use golang.org/x/tools/go/types instead
)

func applyRewrite(f *Func, r func(*Value) bool) {
	// repeat rewrites until we find no more rewrites
	for {
		change := false
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				if r(v) {
					change = true
				}
			}
		}
		if !change {
			return
		}
	}
}

// Common functions called from rewriting rules

func is64BitInt(t Type) bool {
	return typeIdentical(t, TypeInt64) ||
		typeIdentical(t, TypeUint64) ||
		(typeIdentical(t, TypeInt) && intSize == 8) ||
		(typeIdentical(t, TypeUint) && intSize == 8) ||
		(typeIdentical(t, TypeUintptr) && ptrSize == 8)
}

func is32BitInt(t Type) bool {
	return typeIdentical(t, TypeInt32) ||
		typeIdentical(t, TypeUint32) ||
		(typeIdentical(t, TypeInt) && intSize == 4) ||
		(typeIdentical(t, TypeUint) && intSize == 4) ||
		(typeIdentical(t, TypeUintptr) && ptrSize == 4)
}

func isSigned(t Type) bool {
	return typeIdentical(t, TypeInt) ||
		typeIdentical(t, TypeInt8) ||
		typeIdentical(t, TypeInt16) ||
		typeIdentical(t, TypeInt32) ||
		typeIdentical(t, TypeInt64)
}

func typeSize(t Type) int {
	switch t {
	case TypeInt32, TypeUint32:
		return 4
	case TypeInt64, TypeUint64:
		return 8
	case TypeUintptr:
		return ptrSize
	case TypeInt, TypeUint:
		return intSize
	default:
		if _, ok := t.(*types.Pointer); ok {
			return ptrSize
		}
		panic("TODO: width of " + t.String())
	}
}
