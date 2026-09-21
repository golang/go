// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/ssa/ssaop"

// IsFlagOp reports if v is an OP with the flag type.
func (v *Value) IsFlagOp() bool {
	if v.Type.IsFlags() || v.Type.IsTuple() && v.Type.FieldType(1).IsFlags() {
		return true
	}
	// PPC64 carry generators put their carry in a non-flag-typed register
	// in their output.
	switch v.Op {
	case ssaop.OpPPC64SUBC, ssaop.OpPPC64ADDC, ssaop.OpPPC64SUBCconst, ssaop.OpPPC64ADDCconst:
		return true
	}
	return false
}

// HasFlagInput reports whether v has a flag value as any of its inputs.
func (v *Value) HasFlagInput() bool {
	for _, a := range v.Args {
		if a.IsFlagOp() {
			return true
		}
	}
	// PPC64 carry dependencies are conveyed through their final argument,
	// so we treat those operations as taking flags as well.
	switch v.Op {
	case ssaop.OpPPC64SUBE, ssaop.OpPPC64ADDE, ssaop.OpPPC64SUBZEzero, ssaop.OpPPC64ADDZE, ssaop.OpPPC64ADDZEzero:
		return true
	}
	return false
}
