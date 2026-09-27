// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

// This file contains a simple pass to compute *stack* pointer demographics:
//	stack_pointers: count of ir.Name-s that are stack allocated.
//	stack_pointers_decomposeAddr_candidate: count of ir.Name-s that are stack allocated and qualifies for
//		decompose addr.
//	stack_pointers_mem2reg_candidate: count of ir.Name-s that are stack allocated,
//		qualifies for mem2reg and are only used in Loads or Stores(as the address, not value).
//	stack_scalar_pointers_decomposeAddr_candidate: count of scalar ir.Name-s that are stack allocated and
//		qualifies for decompose addr.
//	stack_scalar_pointers_decomposeAddr_candidate: count of scalar ir.Name-s that are stack allocated and
//		qualifies for decomposeAddr.
//	stack_struct_pointers_decomposeAddr_candidate: count of struct pointers that are stack allocated and
//		qualifies for decomposeAddr.

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

	demographics, _, _ := variableDemographics(f)

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
					if demographics[base.Aux].lszmco {
						s.Record("stack_pointers_decomposeAddr_candidate", 1)
						if demographics[base.Aux].ls {
							s.Record("stack_pointers_mem2reg_candidate", 1)
						}
						if n.Type() != nil {
							if n.Type().IsScalar() {
								s.Record("stack_scalar_pointers_decomposeAddr_candidate", 1)
							} else if n.Type().IsStruct() {
								s.Record("stack_struct_pointers_decomposeAddr_candidate", 1)
							}
						}
					}
					continue
				}
			}
		}
	}
}
