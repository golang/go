// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

func applyRewrite(f *Func, r func(*Value) bool) {
	// repeat rewrites until we find no more rewrites
	var curv *Value
	defer func() {
		if curv != nil {
			fmt.Printf("panic during rewrite of %s\n", curv.LongString())
			// TODO(khr): print source location also
		}
	}()
	for {
		change := false
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				curv = v
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
	return t.Size() == 8 && t.IsInteger()
}

func is32BitInt(t Type) bool {
	return t.Size() == 4 && t.IsInteger()
}

func isPtr(t Type) bool {
	return t.IsPtr()
}

func isSigned(t Type) bool {
	return t.IsSigned()
}

func typeSize(t Type) int64 {
	return t.Size()
}
