// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "log"

func applyRewrite(f *Func, rb func(*Block) bool, rv func(*Value, *Config) bool) {
	// repeat rewrites until we find no more rewrites
	var curb *Block
	var curv *Value
	defer func() {
		if curb != nil {
			log.Printf("panic during rewrite of %s\n", curb.LongString())
		}
		if curv != nil {
			log.Printf("panic during rewrite of %s\n", curv.LongString())
			panic("rewrite failed")
			// TODO(khr): print source location also
		}
	}()
	config := f.Config
	for {
		change := false
		for _, b := range f.Blocks {
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

func isPtr(t Type) bool {
	return t.IsPtr()
}

func isSigned(t Type) bool {
	return t.IsSigned()
}

func typeSize(t Type) int64 {
	return t.Size()
}

// addOff adds two offset aux values.  Each should be an int64.  Fails if wraparound happens.
func addOff(a, b interface{}) interface{} {
	return addOffset(a.(int64), b.(int64))
}
func addOffset(x, y int64) int64 {
	z := x + y
	// x and y have same sign and z has a different sign => overflow
	if x^y >= 0 && x^z < 0 {
		log.Panicf("offset overflow %d %d\n", x, y)
	}
	return z
}

func inBounds(idx, len int64) bool {
	return idx >= 0 && idx < len
}
