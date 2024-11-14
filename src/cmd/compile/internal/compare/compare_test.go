// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compare

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"cmd/internal/sys"
	"testing"
)

type typefn func() *types.Type

func init() {
	// These are the few constants that need to be initialized in order to use
	// the types package without using the typecheck package by calling
	// typecheck.InitUniverse() (the normal way to initialize the types package).
	types.PtrSize = 8
	types.RegSize = 8
	types.MaxWidth = 1 << 50
	base.Ctxt = &obj.Link{Arch: &obj.LinkArch{Arch: &sys.Arch{Alignment: 1, CanMergeLoads: true}}}
	typecheck.InitUniverse()
}

func TestEqStructCost(t *testing.T) {
	repeat := func(n int, typ *types.Type) []*types.Type {
		typs := make([]*types.Type, n)
		for i := range typs {
			typs[i] = typ
		}
		return typs
	}

	tt := []struct {
		name             string
		cost             int64
		nonMergeLoadCost int64
		fieldTypes       []*types.Type
	}{
		{"struct without fields", 0, 0, nil},
		{"struct with 1 byte field", 1, 1, repeat(1, types.ByteType)},
		{"struct with 8 byte fields", 1, 8, repeat(8, types.ByteType)},
		{"struct with 16 byte fields", 2, 16, repeat(16, types.ByteType)},
		{"struct with 32 byte fields", 4, 32, repeat(32, types.ByteType)},
		{"struct with 2 int32 fields", 1, 2, repeat(2, types.Types[types.TINT32])},
		{"struct with 2 int32 fields and 1 int64", 2, 3,
			[]*types.Type{
				types.Types[types.TINT32],
				types.Types[types.TINT32],
				types.Types[types.TINT64],
			},
		},
		{"struct with 1 int field and 1 string", 3, 3,
			[]*types.Type{
				types.Types[types.TINT64],
				types.Types[types.TSTRING],
			},
		},
		{"struct with 2 strings", 4, 4, repeat(2, types.Types[types.TSTRING])},
		{"struct with 1 large byte array field", 26, 101,
			[]*types.Type{
				types.NewArray(types.Types[types.TUINT16], 101),
			},
		},
		{"struct with string array field", 4, 4,
			[]*types.Type{
				types.NewArray(types.Types[types.TSTRING], 2),
			},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func { t ->
			fields := make([]*types.Field, len(tc.fieldTypes))
			for i, ftyp := range tc.fieldTypes {
				fields[i] = types.NewField(src.NoXPos, typecheck.LookupNum("f", i), ftyp)
			}
			typ := types.NewStruct(fields)
			types.CalcSize(typ)

			want := tc.cost
			base.Ctxt.Arch.CanMergeLoads = true
			actual := EqStructCost(typ)
			if actual != want {
				t.Errorf("CanMergeLoads=true EqStructCost(%v) = %d, want %d", typ, actual, want)
			}

			base.Ctxt.Arch.CanMergeLoads = false
			want = tc.nonMergeLoadCost
			actual = EqStructCost(typ)
			if actual != want {
				t.Errorf("CanMergeLoads=false EqStructCost(%v) = %d, want %d", typ, actual, want)
			}
		})
	}
}
