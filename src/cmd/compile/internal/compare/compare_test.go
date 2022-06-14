package compare

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
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
	types.InitTypes(func(sym *types.Sym, typ *types.Type) types.Object {
		obj := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, sym)
		obj.SetType(typ)
		sym.Def = obj
		return obj
	})
	base.Ctxt = &obj.Link{Arch: &obj.LinkArch{Arch: &sys.Arch{Alignment: 1, CanMergeLoads: true}}}
}

func TestEqStructCost(t *testing.T) {
	newByteField := func(parent *types.Type, offset int64) *types.Field {
		f := types.NewField(src.XPos{}, parent.Sym(), types.ByteType)
		f.Offset = offset
		return f
	}
	newField := func(parent *types.Type, offset int64, kind types.Kind) *types.Field {
		f := types.NewField(src.XPos{}, parent.Sym(), types.Types[kind])
		f.Offset = offset
		return f
	}
	tt := []struct {
		name string
		cost int64
		tfn  typefn
	}{
		{"struct without fields", 0,
			func() *types.Type {
				return types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
			}},
		{"struct with 1 byte field", 1,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := []*types.Field{
					newByteField(parent, 0),
				}
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 8 byte fields", 1,
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
		{"struct with 16 byte fields", 2,
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
		{"struct with 32 byte fields", 4,
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
		{"struct with 2 int32 fields", 1,
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
		{"struct with 2 int32 fields and 1 int64", 2,
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
		{"struct with 1 int field and 1 string", 3,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 2)
				fields[0] = newField(parent, int64(0), types.TINT64)
				fields[1] = newField(parent, int64(8), types.TSTRING)
				parent.SetFields(fields)
				return parent
			},
		},
		{"struct with 2 strings", 4,
			func() *types.Type {
				parent := types.NewStruct(types.NewPkg("main", ""), []*types.Field{})
				fields := make([]*types.Field, 2)
				fields[0] = newField(parent, int64(0), types.TSTRING)
				fields[1] = newField(parent, int64(8), types.TSTRING)
				parent.SetFields(fields)
				return parent
			},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			want := tc.cost
			actual := EqStructCost(tc.tfn())
			if actual != want {
				t.Errorf("EqStructCost(%v) = %d, want %d", tc.tfn, actual, want)
			}
		})
	}
}
