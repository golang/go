// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

func applyRewrite(f *Func, rb func(*Block) bool, rv func(*Value, *Config) bool) {
	// repeat rewrites until we find no more rewrites
	var curb *Block
	var curv *Value
	defer func() {
		if curb != nil {
			curb.Fatalf("panic during rewrite of block %s\n", curb.LongString())
		}
		if curv != nil {
			curv.Fatalf("panic during rewrite of value %s\n", curv.LongString())
			// TODO(khr): print source location also
		}
	}()
	config := f.Config
	for {
		change := false
		for _, b := range f.Blocks {
			if b.Kind == BlockDead {
				continue
			}
			if b.Control != nil && b.Control.Op == OpCopy {
				for b.Control.Op == OpCopy {
					b.Control = b.Control.Args[0]
				}
			}
			curb = b
			if rb(b) {
				change = true
			}
			curb = nil
			for _, v := range b.Values {
				// elide any copies generated during rewriting
				for i, a := range v.Args {
					if a.Op != OpCopy {
						continue
					}
					for a.Op == OpCopy {
						a = a.Args[0]
					}
					v.Args[i] = a
				}

				// apply rewrite function
				curv = v
				if rv(v, config) {
					change = true
				}
				curv = nil
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

func is16BitInt(t Type) bool {
	return t.Size() == 2 && t.IsInteger()
}

func is8BitInt(t Type) bool {
	return t.Size() == 1 && t.IsInteger()
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

// addOff adds two int64 offsets. Fails if wraparound happens.
func addOff(x, y int64) int64 {
	z := x + y
	// x and y have same sign and z has a different sign => overflow
	if x^y >= 0 && x^z < 0 {
		panic(fmt.Sprintf("offset overflow %d %d", x, y))
	}
	return z
}

func mergeSym(x, y interface{}) interface{} {
	if x == nil {
		return y
	}
	if y == nil {
		return x
	}
	panic(fmt.Sprintf("mergeSym with two non-nil syms %s %s", x, y))
	return nil
}

func inBounds(idx, len int64) bool {
	return idx >= 0 && idx < len
}
