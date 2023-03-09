// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unusedwrite

import (
	_ "embed"
	"fmt"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ssa"
)

//go:embed doc.go
var doc string

// Analyzer reports instances of writes to struct fields and arrays
// that are never read.
var Analyzer = &analysis.Analyzer{
	Name:     "unusedwrite",
	Doc:      analysisutil.MustExtractDoc(doc, "unusedwrite"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/unusedwrite",
	Requires: []*analysis.Analyzer{buildssa.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	ssainput := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA)
	for _, fn := range ssainput.SrcFuncs {
		// TODO(taking): Iterate over fn._Instantiations() once exported. If so, have 1 report per Pos().
		reports := checkStores(fn)
		for _, store := range reports {
			switch addr := store.Addr.(type) {
			case *ssa.FieldAddr:
				pass.Reportf(store.Pos(),
					"unused write to field %s",
					getFieldName(addr.X.Type(), addr.Field))
			case *ssa.IndexAddr:
				pass.Reportf(store.Pos(),
					"unused write to array index %s", addr.Index)
			}
		}
	}
	return nil, nil
}

// checkStores returns *Stores in fn whose address is written to but never used.
func checkStores(fn *ssa.Function) []*ssa.Store {
	var reports []*ssa.Store
	// Visit each block. No need to visit fn.Recover.
	for _, blk := range fn.Blocks {
		for _, instr := range blk.Instrs {
			// Identify writes.
			if store, ok := instr.(*ssa.Store); ok {
				// Consider field/index writes to an object whose elements are copied and not shared.
				// MapUpdate is excluded since only the reference of the map is copied.
				switch addr := store.Addr.(type) {
				case *ssa.FieldAddr:
					if isDeadStore(store, addr.X, addr) {
						reports = append(reports, store)
					}
				case *ssa.IndexAddr:
					if isDeadStore(store, addr.X, addr) {
						reports = append(reports, store)
					}
				}
			}
		}
	}
	return reports
}

// isDeadStore determines whether a field/index write to an object is dead.
// Argument "obj" is the object, and "addr" is the instruction fetching the field/index.
func isDeadStore(store *ssa.Store, obj ssa.Value, addr ssa.Instruction) bool {
	// Consider only struct or array objects.
	if !hasStructOrArrayType(obj) {
		return false
	}
	// Check liveness: if the value is used later, then don't report the write.
	for _, ref := range *obj.Referrers() {
		if ref == store || ref == addr {
			continue
		}
		switch ins := ref.(type) {
		case ssa.CallInstruction:
			return false
		case *ssa.FieldAddr:
			// Check whether the same field is used.
			if ins.X == obj {
				if faddr, ok := addr.(*ssa.FieldAddr); ok {
					if faddr.Field == ins.Field {
						return false
					}
				}
			}
			// Otherwise another field is used, and this usage doesn't count.
			continue
		case *ssa.IndexAddr:
			if ins.X == obj {
				return false
			}
			continue // Otherwise another object is used
		case *ssa.Lookup:
			if ins.X == obj {
				return false
			}
			continue // Otherwise another object is used
		case *ssa.Store:
			if ins.Val == obj {
				return false
			}
			continue // Otherwise other object is stored
		default: // consider live if the object is used in any other instruction
			return false
		}
	}
	return true
}

// isStructOrArray returns whether the underlying type is struct or array.
func isStructOrArray(tp types.Type) bool {
	if named, ok := tp.(*types.Named); ok {
		tp = named.Underlying()
	}
	switch tp.(type) {
	case *types.Array:
		return true
	case *types.Struct:
		return true
	}
	return false
}

// hasStructOrArrayType returns whether a value is of struct or array type.
func hasStructOrArrayType(v ssa.Value) bool {
	if instr, ok := v.(ssa.Instruction); ok {
		if alloc, ok := instr.(*ssa.Alloc); ok {
			// Check the element type of an allocated register (which always has pointer type)
			// e.g., for
			//   func (t T) f() { ...}
			// the receiver object is of type *T:
			//   t0 = local T (t)   *T
			if tp, ok := alloc.Type().(*types.Pointer); ok {
				return isStructOrArray(tp.Elem())
			}
			return false
		}
	}
	return isStructOrArray(v.Type())
}

// getFieldName returns the name of a field in a struct.
// It the field is not found, then it returns the string format of the index.
//
// For example, for struct T {x int, y int), getFieldName(*T, 1) returns "y".
func getFieldName(tp types.Type, index int) string {
	if pt, ok := tp.(*types.Pointer); ok {
		tp = pt.Elem()
	}
	if named, ok := tp.(*types.Named); ok {
		tp = named.Underlying()
	}
	if stp, ok := tp.(*types.Struct); ok {
		return stp.Field(index).Name()
	}
	return fmt.Sprintf("%d", index)
}
