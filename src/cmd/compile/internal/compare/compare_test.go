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
	typecheck.InitUniverse()
	base.Ctxt = &obj.Link{Arch: &obj.LinkArch{Arch: &sys.Arch{Alignment: 1, CanMergeLoads: true}}}
}

func TestEqStructCost(t *testing.T) {
	newByteField := func(parent *types.Type, offset int64) *types.Field {
		f := types.NewField(src.XPos{}, parent.Sym(), types.ByteType)
		f.Offset = offset
		return f
	}
	newArrayField := func(parent *types.Type, offset int64, len int64, kind types.Kind) *types.Field {
		f := types.NewField(src.XPos{}, parent.Sym(), types.NewArray(types.Types[kind], len))
		// Call Type.Size here to force the size calculation to be done. If not done here the size returned later is incorrect.
		f.Type.Size()
		f.Offset = offset
		return f
	}
	newField := func(parent *types.Type, offset int64, kind types.Kind) *types.Field {
		f := types.NewField(src.XPos{}, parent.Sym(), types.Types[kind])
		f.Offset = offset
		return f
	}
	tt := []struct {
		name             string
		cost             int64
		nonMergeLoadCost int64
		tfn              typefn
	}{
		{"struct without fields", 0, 0,
			func() *types.Type {
				return types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
			}},
		{"struct with 1 byte field", 1, 1,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := []*types.Field{
					newByteField(parent, 0),
				}
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 8 byte fields", 1, 8,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 8)
				for i := range fields {
					fields[i] = newByteField(parent, int64(i))
				}
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 16 byte fields", 2, 16,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 16)
				for i := range fields {
					fields[i] = newByteField(parent, int64(i))
				}
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 32 byte fields", 4, 32,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 32)
				for i := range fields {
					fields[i] = newByteField(parent, int64(i))
				}
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 2 int32 fields", 1, 2,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 2)
				for i := range fields {
					fields[i] = newField(parent, int64(i*4), types.TINT32)
				}
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 2 int32 fields and 1 int64", 2, 3,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 3)
				fields[0] = newField(parent, int64(0), types.TINT32)
				fields[1] = newField(parent, int64(4), types.TINT32)
				fields[2] = newField(parent, int64(8), types.TINT64)
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 1 int field and 1 string", 3, 3,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 2)
				fields[0] = newField(parent, int64(0), types.TINT64)
				fields[1] = newField(parent, int64(8), types.TSTRING)
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 2 strings", 4, 4,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 2)
				fields[0] = newField(parent, int64(0), types.TSTRING)
				fields[1] = newField(parent, int64(8), types.TSTRING)
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 1 large byte array field", 26, 101,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := []*types.Field{
					newArrayField(parent, 0, 101, types.TUINT16),
				}
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with string array field", 4, 4,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := []*types.Field{
					newArrayField(parent, 0, 2, types.TSTRING),
				}
				parent.SetFields(fields)
				return parent
			},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			want := tc.cost
			base.Ctxt.Arch.CanMergeLoads = true
			actual := EqStructCost(tc.tfn())
			if actual != want {
				t.Errorf("CanMergeLoads=true EqStructCost(%v) = %d, want %d", tc.tfn, actual, want)
			}

			base.Ctxt.Arch.CanMergeLoads = false
			want = tc.nonMergeLoadCost
			actual = EqStructCost(tc.tfn())
			if actual != want {
				t.Errorf("CanMergeLoads=false EqStructCost(%v) = %d, want %d", tc.tfn, actual, want)
			}
		})
	}
}
