// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

// This file contains a simple pass to compute *stack* pointer demographics:
//	stack_pointers: count of ir.Name-s that are stack allocated.
//	stack_pointers_noescape: count of ir.Name-s that are stack allocated and does not escape.
//	stack_pointers_only_load_store_noescape: count of ir.Name-s that are stack allocated,
//		does not escape and are only used in Loads or Stores(as the address, not value).
//	stack_scalar_pointers_noescape: count of scalar ir.Name-s that are stack allocated and does
//		not escape.
//	stack_struct_pointers_noescape: count of struct pointers that does
//		not escape.
// Also the counts are incomplete, the escape analysis is a very naive and
// limited one, so the noescape counts should be smaller or equal to the actual
// variable counts of the analyzed function.
// TODO: count heap pointers as well, heap pointers are ssa values though, so we can consider
// counting ssa values of pointer types instead of ir.Name-s.

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
)

func ptrStats(f *ssa.Func) {
	// Record statistics if stats are enabled.
	s := f.NewStats("ptrstats")
	if s == nil {
		return
	}
	// Go compiler does not maintain the uses of ssa values, we need to construct
	// that structure here.
	// We only care about pointers here.
	uses := make(map[*ssa.Value][]*ssa.Value)
	varLives := map[ssa.Aux]bool{}
	for _, b := range f.Blocks {
		for _, c := range b.Controls {
			if c == nil {
				break
			}
			if c.Op == ssaop.OpOffPtr || c.Op == ssaop.OpLocalAddr {
				panic("unexpected pointer value in block control")
			}
		}
		for _, v := range b.Values {
			for _, arg := range v.Args {
				if arg.Op != ssaop.OpOffPtr && arg.Op != ssaop.OpLocalAddr {
					continue
				}
				uses[arg] = append(uses[arg], v)
			}
			if v.Op == ssaop.OpVarLive {
				varLives[v.Aux] = true
			}
		}
	}
	// Compute the escape-ness of variables and the memory access demographic patterns.
	escapes := make(map[ssa.Aux]bool)
	notOnlyLoadStore := make(map[ssa.Aux]bool)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == ssaop.OpLocalAddr {
				if n := v.Aux.(*ir.Name); n.Class == ir.PAUTO || isABIInternalParam(f, n) {
					e, o := isEscaping(v, uses)
					if e || varLives[n] {
						escapes[n] = true
					}
					if o {
						notOnlyLoadStore[n] = true
					}
				}
			}
		}
	}

	nameSeen := make(map[*ir.Name]bool)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !v.Type.IsPtr() {
				continue
			}
			// Peel off OffPtr
			base := v
			for base.Op == ssaop.OpOffPtr {
				base = base.Args[0]
			}
			if base.Op == ssaop.OpLocalAddr {
				n, ok := base.Aux.(*ir.Name)
				if ok && (n.Class == ir.PAUTO || isABIInternalParam(f, n)) {
					if nameSeen[n] {
						continue
					}
					nameSeen[n] = true
					s.Record("stack_pointers", 1)
					if !escapes[base.Aux] {
						s.Record("stack_pointers_noescape", 1)
						if !notOnlyLoadStore[base.Aux] {
							s.Record("stack_pointers_only_load_store_noescape", 1)
						}
						if n.Type() != nil {
							if n.Type().IsScalar() {
								s.Record("stack_scalar_pointers_noescape", 1)
							} else if n.Type().IsStruct() {
								s.Record("stack_struct_pointers_noescape", 1)
							}
						}
					}
					continue
				}
			}
		}
	}
}

// isEscaping is a simplified escape analysis. escapes reports whether the
// pointer has a use the passes cannot reason about (stored as a value,
// passed to a call, compared, ...). notOnlyLoadStore reports whether it
// has any use besides a direct Load or Store through it.
func isEscaping(v *ssa.Value, uses map[*ssa.Value][]*ssa.Value) (escapes bool, notOnlyLoadStore bool) {
	q := []*ssa.Value{v}
	for len(q) > 0 {
		curr := q[0]
		q = q[1:]
		for _, use := range uses[curr] {
			switch use.Op {
			case ssaop.OpOffPtr:
				notOnlyLoadStore = true
				q = append(q, use)
			case ssaop.OpLoad:
				// Nothing special
			case ssaop.OpStore:
				if curr == use.Args[1] {
					// Storing the address as value.
					escapes = true
					notOnlyLoadStore = true
					return
				}
			case ssaop.OpZero, ssaop.OpMove:
				notOnlyLoadStore = true
			default:
				escapes = true
				notOnlyLoadStore = true
				return
			}
		}
	}
	return
}
