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
	if b, ok := t.Underlying().(*types.Basic); ok {
		switch b.Kind() {
		case types.Int64, types.Uint64:
			return true
		}
	}
	return false
}

func is32BitInt(t Type) bool {
	if b, ok := t.Underlying().(*types.Basic); ok {
		switch b.Kind() {
		case types.Int32, types.Uint32:
			return true
		}
	}
	return false
}

func isSigned(t Type) bool {
	if b, ok := t.Underlying().(*types.Basic); ok {
		switch b.Kind() {
		case types.Int8, types.Int16, types.Int32, types.Int64:
			return true
		}
	}
	return false
}

var sizer types.Sizes = &types.StdSizes{int64(ptrSize), int64(ptrSize)} // TODO(khr): from config
func typeSize(t Type) int64 {
	return sizer.Sizeof(t)
}
