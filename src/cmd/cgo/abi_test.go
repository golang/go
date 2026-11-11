// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"internal/testenv"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// TestInspectTypeClassification verifies AST type-expression parsing (pointers,
// selectors, fixed-length arrays, slices, inline structs with HostLayout) and
// BindingType map resolution for opaque C structs versus defined Go types with
// scalar underlying types.
func TestInspectTypeClassification(t *testing.T) {
	typeMap := map[string]BindingType{
		"_Ctype_S":    {Name: "_Ctype_S", Size: 8, Align: 4},
		"MyInt":       {Name: "MyInt", Def: "int32", Size: 4, Align: 4},
		"MyFloat":     {Name: "MyFloat", Def: "float64", Size: 8, Align: 8},
		"GoOnly":      {Name: "GoOnly", Def: "struct { a [4]int64 }", Size: 32, Align: 8},
		"structIdent": {Name: "structIdent", Def: "int32", Size: 4, Align: 4},
	}

	tests := []struct {
		typeName      string
		wantSize      int64
		wantAlign     int64
		wantFloat     bool
		wantSigned    bool
		wantComposite bool
	}{
		{"*byte", 8, 8, false, false, false},
		{"unsafe.Pointer", 8, 8, false, false, false},
		{"_Ctype_long", 8, 8, false, true, false},
		{"[3]int16", 6, 2, false, false, true},
		{"[2]float32", 8, 4, false, false, true},
		{"struct { _ structs.HostLayout; a int8; b int32 }", 8, 4, false, false, true},
		{"_Ctype_S", 8, 4, false, false, true},
		{"MyInt", 4, 4, false, true, false},
		{"MyFloat", 8, 8, true, false, false},
		{"structIdent", 4, 4, false, true, false},
		{"_Ctype_GoOnly", 8, 8, false, false, false},
	}

	sysv := getABI("linux", "amd64")
	for _, tt := range tests {
		info := sysv.inspectType(tt.typeName, typeMap)
		if info.size != tt.wantSize || info.align != tt.wantAlign || info.isFloat != tt.wantFloat || info.signed != tt.wantSigned || info.composite != tt.wantComposite {
			t.Errorf("inspectType(%q) = (size=%d, align=%d, float=%v, signed=%v, composite=%v), want (%d, %d, %v, %v, %v)",
				tt.typeName, info.size, info.align, info.isFloat, info.signed, info.composite,
				tt.wantSize, tt.wantAlign, tt.wantFloat, tt.wantSigned, tt.wantComposite)
		}
	}

	// Non-literal array lengths (such as unevaluated named constants) must panic
	// rather than silently treating the array as zero-length.
	func() {
		defer func() {
			if recover() == nil {
				t.Errorf("inspectType([N]int32) did not panic")
			}
		}()
		sysv.inspectType("[N]int32", typeMap)
	}()
}

// TestInspectTypeNestedAndCycles verifies that inspectType flattens deeply
// nested single-field wrapper structs into scalar leaves for register ABI
// classification (e.g. SysV SSE / ARM64 HFA) while terminating cleanly on
// recursive type-definition or struct cycles.
func TestInspectTypeNestedAndCycles(t *testing.T) {
	extMap := map[string]BindingType{
		"CycleA":      {Name: "CycleA", Def: "CycleB", Size: 8, Align: 8},
		"CycleB":      {Name: "CycleB", Def: "CycleA", Size: 8, Align: 8},
		"CycleStruct": {Name: "CycleStruct", Def: "struct { x CycleStruct }", Size: 8, Align: 8},
	}
	prev := "float32"
	for i := range 20 {
		name := "Chain_" + string(rune('A'+i))
		extMap[name] = BindingType{Name: name, Def: "struct { v " + prev + " }", Size: 4, Align: 4}
		prev = name
	}
	sysv := getABI("linux", "amd64")
	if info := sysv.inspectType(prev, extMap); !info.fieldsOK || len(info.fields) != 1 || !info.fields[0].isFloat || info.fields[0].size != 4 {
		t.Errorf("inspectType(20-level chain %s) = %+v, want 1 float32 leaf", prev, info)
	}
	if info := sysv.inspectType("CycleA", extMap); info.fieldsOK || info.size != 8 {
		t.Errorf("inspectType(CycleA) = %+v, want fieldsOK=false size=8 on type-definition cycle", info)
	}
	if info := sysv.inspectType("CycleStruct", extMap); info.fieldsOK || info.size != 8 {
		t.Errorf("inspectType(CycleStruct) = %+v, want fieldsOK=false size=8 on recursive struct", info)
	}
}

func TestLayoutAMD64(t *testing.T) {
	typeMap := map[string]BindingType{
		"_Ctype_Small":  {Name: "_Ctype_Small", Size: 8, Align: 4},
		"_Ctype_Medium": {Name: "_Ctype_Medium", Size: 12, Align: 4},
		"_Ctype_Large":  {Name: "_Ctype_Large", Size: 32, Align: 4},
	}

	tests := []struct {
		name string
		fn   BindingFunc
		want FuncLayout
	}{
		{
			name: "scalars",
			fn:   BindingFunc{Name: "Add", Params: []string{"int32", "int32"}, Ret: "int32"},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 4, Regs: []string{"DI"}},
					{GoOffset: 4, Size: 4, Regs: []string{"SI"}},
				},
				Ret: RetLoc{GoOffset: 8, Size: 4, Regs: []string{"AX"}},
			},
		},
		{
			name: "floats",
			fn:   BindingFunc{Name: "AddFloat", Params: []string{"float32", "float64"}, Ret: "float64"},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 4, IsFloat: true, Regs: []string{"X0"}},
					{GoOffset: 8, Size: 8, IsFloat: true, Regs: []string{"X1"}},
				},
				Ret:           RetLoc{GoOffset: 16, Size: 8, IsFloat: true, Regs: []string{"X0"}},
				FloatRegCount: 2,
			},
		},
		{
			name: "structs_by_val_and_stack",
			fn:   BindingFunc{Name: "PassStructs", Params: []string{"_Ctype_Small", "_Ctype_Large"}},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 8, Regs: []string{"DI"}},
					{GoOffset: 8, Size: 32, StackOff: 0},
				},
				StackSize: 32,
			},
		},
		{
			name: "two_reg_struct_spills_when_one_reg_left",
			fn:   BindingFunc{Name: "SpillMedium", Params: []string{"int64", "int64", "int64", "int64", "int64", "_Ctype_Medium", "int64"}},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 8, Regs: []string{"DI"}},
					{GoOffset: 8, Size: 8, Regs: []string{"SI"}},
					{GoOffset: 16, Size: 8, Regs: []string{"DX"}},
					{GoOffset: 24, Size: 8, Regs: []string{"CX"}},
					{GoOffset: 32, Size: 8, Regs: []string{"R8"}},
					{GoOffset: 40, Size: 12, StackOff: 0},
					{GoOffset: 56, Size: 8, Regs: []string{"R9"}},
				},
				StackSize: 16,
			},
		},
		{
			name: "small_struct_return",
			fn:   BindingFunc{Name: "MakeSmall", Params: []string{"int32"}, Ret: "_Ctype_Small"},
			want: FuncLayout{
				Params: []ParamLoc{{GoOffset: 0, Size: 4, Regs: []string{"DI"}}},
				Ret:    RetLoc{GoOffset: 4, Size: 8, Regs: []string{"AX"}},
			},
		},
		{
			name: "large_struct_return_indirect_takes_DI",
			fn:   BindingFunc{Name: "MakeLarge", Params: []string{"int32"}, Ret: "_Ctype_Large"},
			want: FuncLayout{
				Params: []ParamLoc{{GoOffset: 0, Size: 4, Regs: []string{"SI"}}},
				Ret:    RetLoc{GoOffset: 4, Size: 32, Indirect: true},
			},
		},
		{
			name: "stack_spilling",
			fn: BindingFunc{
				Name: "TenArgs",
				Params: []string{
					"int64", "int64", "int64", "int64",
					"int64", "int64", "int64", "int64",
					"int64", "int64",
				},
			},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 8, Regs: []string{"DI"}},
					{GoOffset: 8, Size: 8, Regs: []string{"SI"}},
					{GoOffset: 16, Size: 8, Regs: []string{"DX"}},
					{GoOffset: 24, Size: 8, Regs: []string{"CX"}},
					{GoOffset: 32, Size: 8, Regs: []string{"R8"}},
					{GoOffset: 40, Size: 8, Regs: []string{"R9"}},
					{GoOffset: 48, Size: 8, StackOff: 0},
					{GoOffset: 56, Size: 8, StackOff: 8},
					{GoOffset: 64, Size: 8, StackOff: 16},
					{GoOffset: 72, Size: 8, StackOff: 24},
				},
				StackSize: 32,
			},
		},
	}

	sysv := getABI("linux", "amd64")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sysv.layout(tt.fn, typeMap)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("amd64 layout mismatch:\ngot:  %+v\nwant: %+v", got, tt.want)
			}
		})
	}
}

func TestLayoutSysVStructsAMD64(t *testing.T) {
	tests := []struct {
		name          string
		def           string
		size          int64
		align         int64
		wantParamRegs []string
		wantRetRegs   []string
	}{
		{
			name:          "Float2",
			def:           "struct {\n\ta float32\n\tb float32\n}",
			size:          8,
			align:         4,
			wantParamRegs: []string{"X0"},
			wantRetRegs:   []string{"X0"},
		},
		{
			name:          "DoubleFloat",
			def:           "struct {\n\td float64\n\tf float32\n}",
			size:          16,
			align:         8,
			wantParamRegs: []string{"X0", "X1"},
			wantRetRegs:   []string{"X0", "X1"},
		},
		{
			name:          "FloatBlankBytes",
			def:           "struct {\n\tf float32\n\t_ [4]byte\n}",
			size:          8,
			align:         4,
			wantParamRegs: []string{"DI"},
			wantRetRegs:   []string{"AX"},
		},
		{
			name:          "Int64Double",
			def:           "struct {\n\ti int64\n\td float64\n}",
			size:          16,
			align:         8,
			wantParamRegs: []string{"DI", "X0"},
			wantRetRegs:   []string{"AX", "X0"},
		},
		{
			name:          "DoubleInt64",
			def:           "struct {\n\td float64\n\ti int64\n}",
			size:          16,
			align:         8,
			wantParamRegs: []string{"X0", "DI"},
			wantRetRegs:   []string{"X0", "AX"},
		},
		{
			name:          "FloatInt32Double",
			def:           "struct {\n\tf float32\n\ti int32\n\td float64\n}",
			size:          16,
			align:         8,
			wantParamRegs: []string{"DI", "X0"},
			wantRetRegs:   []string{"AX", "X0"},
		},
	}

	sysv := getABI("linux", "amd64")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			typeMap := map[string]BindingType{
				tt.name: {Name: tt.name, Def: tt.def, Size: tt.size, Align: tt.align},
			}
			lay := sysv.layout(BindingFunc{Name: tt.name, Params: []string{tt.name}, Ret: tt.name}, typeMap)
			if !slices.Equal(lay.Params[0].Regs, tt.wantParamRegs) || !slices.Equal(lay.Ret.Regs, tt.wantRetRegs) {
				t.Errorf("layout(%s): got param regs %v, ret regs %v; want param regs %v, ret regs %v",
					tt.name, lay.Params[0].Regs, lay.Ret.Regs, tt.wantParamRegs, tt.wantRetRegs)
			}
		})
	}
}

func checkOdd5No8ByteMoves(t *testing.T, arch string) {
	t.Helper()
	typeMap := map[string]BindingType{
		"Odd5": {Name: "Odd5", Def: "struct {\n\tb [5]uint8\n}", Size: 5, Align: 1},
	}
	cfg := getABI("linux", arch)
	var buf bytes.Buffer
	cfg.writeTrampoline(&buf, BindingFunc{
		Name:   "AbiOdd5Bounce",
		CName:  "c_abi_odd5_bounce",
		Params: []string{"Odd5"},
		Ret:    "Odd5",
	}, typeMap)
	asm := buf.String()
	for _, line := range strings.Split(asm, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "MOVQ\t") || strings.HasPrefix(trimmed, "MOVD\t") {
			for _, bad := range []string{"(R12)", "(R19)", "(AX)", "(CX)", "(DX)", "(BX)", "(SI)", "(DI)", "(R8)", "(R9)", "(R10)", "(R11)", "(R0)", "(R1)", "(R2)", "(R3)", "(R4)"} {
				if strings.Contains(trimmed, bad) {
					t.Errorf("%s Odd5 trampoline has 8-byte memory move %q in:\n%s", arch, trimmed, asm)
				}
			}
		}
	}
}

func TestWriteTrampolineAMD64(t *testing.T) {
	fn := BindingFunc{
		Name:   "_Add",
		CName:  "Add",
		Params: []string{"int32", "int32"},
		Ret:    "int32",
	}

	sysv := getABI("linux", "amd64")
	var amdBuf bytes.Buffer
	sysv.writeTrampoline(&amdBuf, fn, nil)
	amdAsm := amdBuf.String()
	for _, want := range []string{
		"TEXT _cgo_trampoline__Add(SB), NOSPLIT, $0-0",
		"MOVQ\tDI, R12",
		"MOVL\t0(R12), DI",
		"MOVL\t4(R12), SI",
		"XORL\tAX, AX",
		"CALL\tAdd(SB)",
		"MOVL\tAX, 8(R12)",
	} {
		if !strings.Contains(amdAsm, want) {
			t.Errorf("amd64 trampoline missing %q in:\n%s", want, amdAsm)
		}
	}
	if strings.Contains(amdAsm, "(SP)") {
		t.Errorf("amd64 register-only trampoline unexpectedly touches stack:\n%s", amdAsm)
	}

	// Indirect call trampoline.
	var indBuf bytes.Buffer
	sysv.writeTrampoline(&indBuf, BindingFunc{
		Name:     "fp__BinOp",
		Params:   []string{"int32", "int32"},
		Ret:      "int32",
		Indirect: true,
	}, nil)
	indAsm := indBuf.String()
	for _, want := range []string{
		"MOVL\t8(R12), DI",
		"MOVL\t12(R12), SI",
		"MOVQ\t0(R12), R11",
		"CALL\tR11",
		"MOVL\tAX, 16(R12)",
	} {
		if !strings.Contains(indAsm, want) {
			t.Errorf("amd64 indirect trampoline missing %q in:\n%s", want, indAsm)
		}
	}

	// Odd-sized struct loads/stores must decompose without 8-byte overreads.
	checkOdd5No8ByteMoves(t, "amd64")

	// C-to-Go callback trampoline.
	var cbBuf bytes.Buffer
	sysv.writeCallbackTrampoline(&cbBuf, BindingFunc{
		Name:   "GoCallback",
		Params: []string{"int32", "int32"},
		Ret:    "unsafe.Pointer",
	}, nil)
	cbAsm := cbBuf.String()
	for _, want := range []string{
		"TEXT GoCallback(SB), NOSPLIT, $32-0",
		"MOVL\tDI, 0(SP)",
		"MOVL\tSI, 4(SP)",
		"MOVQ\t$0, 8(SP)",
		"CALL\t_cgo_wait_runtime_init_done(SB)",
		"MOVQ\tAX, 16(SP)",
		"MOVQ\tAX, CX",
		"LEAQ\t·_cgo_fn_GoCallback(SB), DI",
		"MOVQ\t0(DI), DI",
		"LEAQ\t0(SP), SI",
		"MOVQ\t$32, DX",
		"CALL\tcrosscall2(SB)",
		"MOVQ\t16(SP), DI",
		"CALL\t_cgo_release_context(SB)",
		"MOVQ\t8(SP), AX",
	} {
		if !strings.Contains(cbAsm, want) {
			t.Errorf("amd64 callback trampoline missing %q in:\n%s", want, cbAsm)
		}
	}
}

func TestLayoutARM64(t *testing.T) {
	typeMap := map[string]BindingType{
		"_Ctype_Small":  {Name: "_Ctype_Small", Size: 8, Align: 4},
		"_Ctype_Medium": {Name: "_Ctype_Medium", Size: 12, Align: 4},
		"_Ctype_Large":  {Name: "_Ctype_Large", Size: 32, Align: 4},
	}

	tests := []struct {
		name string
		fn   BindingFunc
		want FuncLayout
	}{
		{
			name: "scalars",
			fn:   BindingFunc{Name: "Add", Params: []string{"int32", "int32"}, Ret: "int32"},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 4, Regs: []string{"R0"}},
					{GoOffset: 4, Size: 4, Regs: []string{"R1"}},
				},
				Ret: RetLoc{GoOffset: 8, Size: 4, Regs: []string{"R0"}},
			},
		},
		{
			name: "floats",
			fn:   BindingFunc{Name: "AddFloat", Params: []string{"float32", "float64"}, Ret: "float64"},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 4, IsFloat: true, Regs: []string{"F0"}},
					{GoOffset: 8, Size: 8, IsFloat: true, Regs: []string{"F1"}},
				},
				Ret:           RetLoc{GoOffset: 16, Size: 8, IsFloat: true, Regs: []string{"F0"}},
				FloatRegCount: 2,
			},
		},
		{
			name: "structs_by_val_and_ref",
			fn:   BindingFunc{Name: "PassStructs", Params: []string{"_Ctype_Small", "_Ctype_Medium", "_Ctype_Large"}},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 8, Regs: []string{"R0"}},
					{GoOffset: 8, Size: 12, Regs: []string{"R1", "R2"}},
					{GoOffset: 20, Size: 32, PassByPtr: true, Regs: []string{"R3"}},
				},
			},
		},
		{
			name: "two_reg_struct_spills_when_one_reg_left",
			fn: BindingFunc{
				Name: "SpillMedium",
				Params: []string{
					"int64", "int64", "int64", "int64",
					"int64", "int64", "int64", "_Ctype_Medium", "int64",
				},
			},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 8, Regs: []string{"R0"}},
					{GoOffset: 8, Size: 8, Regs: []string{"R1"}},
					{GoOffset: 16, Size: 8, Regs: []string{"R2"}},
					{GoOffset: 24, Size: 8, Regs: []string{"R3"}},
					{GoOffset: 32, Size: 8, Regs: []string{"R4"}},
					{GoOffset: 40, Size: 8, Regs: []string{"R5"}},
					{GoOffset: 48, Size: 8, Regs: []string{"R6"}},
					{GoOffset: 56, Size: 12, StackOff: 0},
					{GoOffset: 72, Size: 8, StackOff: 16},
				},
				StackSize: 32,
			},
		},
		{
			name: "small_struct_return",
			fn:   BindingFunc{Name: "MakeSmall", Params: []string{"int32"}, Ret: "_Ctype_Small"},
			want: FuncLayout{
				Params: []ParamLoc{{GoOffset: 0, Size: 4, Regs: []string{"R0"}}},
				Ret:    RetLoc{GoOffset: 4, Size: 8, Regs: []string{"R0"}},
			},
		},
		{
			name: "large_struct_return_indirect",
			fn:   BindingFunc{Name: "MakeLarge", Params: []string{"int32"}, Ret: "_Ctype_Large"},
			want: FuncLayout{
				Params: []ParamLoc{{GoOffset: 0, Size: 4, Regs: []string{"R0"}}},
				Ret:    RetLoc{GoOffset: 4, Size: 32, Indirect: true},
			},
		},
		{
			name: "stack_spilling",
			fn: BindingFunc{
				Name: "TenArgs",
				Params: []string{
					"int64", "int64", "int64", "int64",
					"int64", "int64", "int64", "int64",
					"int64", "int64",
				},
			},
			want: FuncLayout{
				Params: []ParamLoc{
					{GoOffset: 0, Size: 8, Regs: []string{"R0"}},
					{GoOffset: 8, Size: 8, Regs: []string{"R1"}},
					{GoOffset: 16, Size: 8, Regs: []string{"R2"}},
					{GoOffset: 24, Size: 8, Regs: []string{"R3"}},
					{GoOffset: 32, Size: 8, Regs: []string{"R4"}},
					{GoOffset: 40, Size: 8, Regs: []string{"R5"}},
					{GoOffset: 48, Size: 8, Regs: []string{"R6"}},
					{GoOffset: 56, Size: 8, Regs: []string{"R7"}},
					{GoOffset: 64, Size: 8, StackOff: 0},
					{GoOffset: 72, Size: 8, StackOff: 8},
				},
				StackSize: 16,
			},
		},
	}

	aapcs := getABI("linux", "arm64")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := aapcs.layout(tt.fn, typeMap)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("arm64 layout mismatch:\ngot:  %+v\nwant: %+v", got, tt.want)
			}
		})
	}
}

func TestLayoutAndTrampolineARM64(t *testing.T) {
	typeMap := map[string]BindingType{
		"Float5":      {Name: "Float5", Def: "struct {\n\ta float32\n\tb float32\n\tc float32\n\td float32\n\te float32\n}", Size: 20, Align: 4},
		"Double4":     {Name: "Double4", Def: "struct {\n\ta float64\n\tb float64\n\tc float64\n\td float64\n}", Size: 32, Align: 8},
		"Int64Double": {Name: "Int64Double", Def: "struct {\n\ti int64\n\td float64\n}", Size: 16, Align: 8},
	}

	aapcs := getABI("linux", "arm64")
	if lay := aapcs.layout(BindingFunc{Name: "D4", Params: []string{"Double4"}, Ret: "Double4"}, typeMap); !reflect.DeepEqual(lay.Params[0].Regs, []string{"F0", "F1", "F2", "F3"}) || !reflect.DeepEqual(lay.Ret.Regs, []string{"F0", "F1", "F2", "F3"}) || lay.Ret.Indirect {
		t.Errorf("ARM64 Double4 HFA layout = %+v", lay)
	}
	if lay := aapcs.layout(BindingFunc{Name: "F5", Params: []string{"Float5"}, Ret: "Float5"}, typeMap); !lay.Params[0].PassByPtr || !lay.Ret.Indirect || aapcs.IndirectRetReg != "R8" {
		t.Errorf("ARM64 Float5 non-HFA layout = %+v", lay)
	}
	if lay := aapcs.layout(BindingFunc{Name: "Mix", Params: []string{"Int64Double"}, Ret: "Int64Double"}, typeMap); !reflect.DeepEqual(lay.Params[0].Regs, []string{"R0", "R1"}) || !reflect.DeepEqual(lay.Ret.Regs, []string{"R0", "R1"}) {
		t.Errorf("ARM64 Int64Double non-HFA layout = %+v", lay)
	}

	var armBuf bytes.Buffer
	aapcs.writeTrampoline(&armBuf, BindingFunc{Name: "_Add", CName: "Add", Params: []string{"int32", "int32"}, Ret: "int32"}, nil)
	for _, want := range []string{
		"TEXT _cgo_trampoline__Add(SB), NOSPLIT|NOFRAME, $0-0",
		"STP.W\t(R30, R19), -32(RSP)",
		"STP\t(R29, R30), 16(RSP)",
		"ADD\t$16, RSP, R29",
		"MOVD\tR0, R19",
		"MOVW\t0(R19), R0",
		"MOVW\t4(R19), R1",
		"CALL\tAdd(SB)",
		"MOVW\tR0, 8(R19)",
		"MOVD\t16(RSP), R29",
		"LDP.P\t32(RSP), (R30, R19)",
	} {
		if !strings.Contains(armBuf.String(), want) {
			t.Errorf("arm64 trampoline missing %q in:\n%s", want, armBuf.String())
		}
	}

	var indBuf bytes.Buffer
	aapcs.writeTrampoline(&indBuf, BindingFunc{Name: "fp__BinOp", Params: []string{"int32", "int32"}, Ret: "int32", Indirect: true}, nil)
	for _, want := range []string{"MOVW\t8(R19), R0", "MOVW\t12(R19), R1", "MOVD\t0(R19), R12", "CALL\t(R12)", "MOVW\tR0, 16(R19)"} {
		if !strings.Contains(indBuf.String(), want) {
			t.Errorf("arm64 indirect trampoline missing %q in:\n%s", want, indBuf.String())
		}
	}

	for _, fn := range []BindingFunc{
		{Name: "AbiNarrowU8", CName: "c_u8", Params: []string{"uint8"}, Ret: "uint8"},
		{Name: "AbiNarrowU16", CName: "c_u16", Params: []string{"uint16"}, Ret: "uint16"},
	} {
		var uBuf bytes.Buffer
		aapcs.writeTrampoline(&uBuf, fn, typeMap)
		uAsm := uBuf.String()
		if !strings.Contains(uAsm, "MOVBU") && !strings.Contains(uAsm, "MOVHU") {
			t.Errorf("arm64 unsigned narrow missing MOVBU/MOVHU:\n%s", uAsm)
		}
		if strings.Contains(uAsm, "\tMOVB\t0(R19)") || strings.Contains(uAsm, "\tMOVH\t0(R19)") {
			t.Errorf("arm64 unsigned narrow used signed load:\n%s", uAsm)
		}
	}

	checkOdd5No8ByteMoves(t, "arm64")

	var cbBuf bytes.Buffer
	aapcs.writeCallbackTrampoline(&cbBuf, BindingFunc{Name: "GoCallback", Params: []string{"int32", "int32"}, Ret: "unsafe.Pointer"}, nil)
	for _, want := range []string{
		"TEXT GoCallback(SB), NOSPLIT|NOFRAME, $0-0",
		"STP.W\t(R30, R27), -64(RSP)",
		"STP\t(R29, R30), 48(RSP)",
		"ADD\t$48, RSP, R29",
		"MOVW\tR0, 16(RSP)",
		"MOVW\tR1, 20(RSP)",
		"MOVD\tZR, 24(RSP)",
		"CALL\t_cgo_wait_runtime_init_done(SB)",
		"MOVD\tR0, 32(RSP)",
		"CALL\tcrosscall2(SB)",
		"CALL\t_cgo_release_context(SB)",
		"MOVD\t24(RSP), R0",
		"MOVD\t48(RSP), R29",
		"LDP.P\t64(RSP), (R30, R27)",
	} {
		if !strings.Contains(cbBuf.String(), want) {
			t.Errorf("arm64 callback trampoline missing %q in:\n%s", want, cbBuf.String())
		}
	}
}

func TestLayoutAndTrampolineDarwinARM64(t *testing.T) {
	typeMap := map[string]BindingType{
		"Float2": {Name: "Float2", Def: "struct {\n\ta, b float32\n}", Size: 8, Align: 4},
		"Float3": {Name: "Float3", Def: "struct {\n\ta, b, c float32\n}", Size: 12, Align: 4},
		"Small2": {Name: "Small2", Def: "struct {\n\ta, b uint8\n}", Size: 2, Align: 1},
		"Medium": {Name: "Medium", Def: "struct {\n\ta, b, c int32\n}", Size: 12, Align: 4},
		"Large":  {Name: "Large", Def: "struct {\n\ta [4]int64\n}", Size: 32, Align: 8},
	}

	linuxCfg := getABI("linux", "arm64")
	for _, targetOS := range []string{"darwin", "ios"} {
		t.Run(targetOS, func(t *testing.T) {
			cfg := getABI(targetOS, "arm64")

			// Packed narrow scalars vs 8-byte slots on Linux.
			packedFn := BindingFunc{
				Name:  "PackedScalars",
				CName: "c_packed_scalars",
				Params: []string{
					"int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64",
					"int8", "int16", "int8", "bool", "int32", "int64", "uint8",
				},
			}
			lay := cfg.layout(packedFn, typeMap)
			wantOffs := []int64{0, 2, 4, 5, 8, 16, 24}
			for i, want := range wantOffs {
				if got := lay.Params[8+i].StackOff; got != want || len(lay.Params[8+i].Regs) != 0 {
					t.Errorf("packed param %d = %+v, want StackOff=%d", 8+i, lay.Params[8+i], want)
				}
			}
			if lay.StackSize != 32 {
				t.Errorf("packed scalars StackSize = %d, want 32", lay.StackSize)
			}
			linLay := linuxCfg.layout(packedFn, typeMap)
			for i, want := range []int64{0, 8, 16, 24, 32, 40, 48} {
				if got := linLay.Params[8+i].StackOff; got != want {
					t.Errorf("linux param %d StackOff = %d, want %d", 8+i, got, want)
				}
			}
			if linLay.StackSize != 64 {
				t.Errorf("linux scalars StackSize = %d, want 64", linLay.StackSize)
			}

			// Trampoline must use exact-width stores for packed narrow stack args.
			var darwinBuf, linuxBuf bytes.Buffer
			cfg.writeTrampoline(&darwinBuf, packedFn, typeMap)
			linuxCfg.writeTrampoline(&linuxBuf, packedFn, typeMap)
			darwinAsm := darwinBuf.String()
			for _, want := range []string{
				"MOVB\tR12, 0(RSP)",
				"MOVH\tR12, 2(RSP)",
				"MOVB\tR12, 4(RSP)",
				"MOVB\tR12, 5(RSP)",
				"MOVW\tR12, 8(RSP)",
				"MOVD\tR12, 16(RSP)",
				"MOVB\tR12, 24(RSP)",
			} {
				if !strings.Contains(darwinAsm, want) {
					t.Errorf("%s/arm64 trampoline missing %q in:\n%s", targetOS, want, darwinAsm)
				}
			}
			for _, bad := range []string{
				"MOVD\tR12, 0(RSP)",
				"MOVD\tR12, 2(RSP)",
				"MOVD\tR12, 4(RSP)",
				"MOVD\tR12, 8(RSP)",
			} {
				if strings.Contains(darwinAsm, bad) {
					t.Errorf("%s/arm64 trampoline overwrites packed stack slot with %q:\n%s", targetOS, bad, darwinAsm)
				}
			}
			if !strings.Contains(linuxBuf.String(), "MOVD\tR12, 0(RSP)") {
				t.Errorf("linux/arm64 trampoline expected 8-byte stack store:\n%s", linuxBuf.String())
			}

			// Packed float32 + integer scalars.
			fFn := BindingFunc{
				Name:  "PackedFloat",
				CName: "c_packed_float",
				Params: []string{
					"int32", "int32", "int32", "int32", "int32", "int32", "int32", "int32",
					"float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32",
					"int16", "float32", "int8",
				},
			}
			fLay := cfg.layout(fFn, typeMap)
			for i, want := range []int64{0, 4, 8} {
				if got := fLay.Params[16+i].StackOff; got != want {
					t.Errorf("packed float param %d StackOff = %d, want %d", 16+i, got, want)
				}
			}
			if fLay.StackSize != 16 {
				t.Errorf("packed float StackSize = %d, want 16", fLay.StackSize)
			}
			var fBuf bytes.Buffer
			cfg.writeTrampoline(&fBuf, fFn, typeMap)
			fAsm := fBuf.String()
			for _, want := range []string{"MOVH\tR12, 0(RSP)", "MOVW\tR12, 4(RSP)", "MOVB\tR12, 8(RSP)"} {
				if !strings.Contains(fAsm, want) {
					t.Errorf("%s/arm64 packed float trampoline missing %q in:\n%s", targetOS, want, fAsm)
				}
			}
			for _, bad := range []string{"MOVD\tR12, 0(RSP)", "MOVD\tR12, 4(RSP)", "MOVD\tR12, 8(RSP)"} {
				if strings.Contains(fAsm, bad) {
					t.Errorf("%s/arm64 packed float trampoline overwrites slot with %q:\n%s", targetOS, bad, fAsm)
				}
			}

			// Packed float32 HFAs align to 4 bytes vs 8 bytes on Linux.
			hfaFn := BindingFunc{
				Name:  "PackedHFA",
				CName: "c_packed_hfa",
				Params: []string{
					"int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64",
					"float64", "float64", "float64", "float64", "float64", "float64",
					"Float3", "float32", "int8", "Float2",
				},
			}
			hfaLay := cfg.layout(hfaFn, typeMap)
			for i, want := range []int64{0, 12, 16, 20} {
				if got := hfaLay.Params[14+i].StackOff; got != want {
					t.Errorf("packed HFA param %d StackOff = %d, want %d", 14+i, got, want)
				}
			}
			if hfaLay.StackSize != 32 {
				t.Errorf("packed HFA StackSize = %d, want 32", hfaLay.StackSize)
			}
			var hfaBuf bytes.Buffer
			cfg.writeTrampoline(&hfaBuf, hfaFn, typeMap)
			hfaAsm := hfaBuf.String()
			for _, want := range []string{"MOVW\tR12, 12(RSP)", "MOVB\tR12, 16(RSP)", "MOVD\tR12, 20(RSP)"} {
				if !strings.Contains(hfaAsm, want) {
					t.Errorf("%s/arm64 packed HFA trampoline missing %q in:\n%s", targetOS, want, hfaAsm)
				}
			}
			for _, bad := range []string{"MOVD\tR12, 12(RSP)", "MOVD\tR12, 16(RSP)"} {
				if strings.Contains(hfaAsm, bad) {
					t.Errorf("%s/arm64 packed HFA trampoline overwrites slot with %q:\n%s", targetOS, bad, hfaAsm)
				}
			}
			linHFA := linuxCfg.layout(hfaFn, typeMap)
			for i, want := range []int64{0, 16, 24, 32} {
				if got := linHFA.Params[14+i].StackOff; got != want {
					t.Errorf("linux HFA param %d StackOff = %d, want %d", 14+i, got, want)
				}
			}
			if linHFA.StackSize != 48 {
				t.Errorf("linux HFA StackSize = %d, want 48", linHFA.StackSize)
			}

			// Non-HFA composites <= 16 bytes stay 8-byte aligned and 8-byte-slot sized (Apple Clang DarwinPCS).
			s2Lay := cfg.layout(BindingFunc{
				Name: "Small2Struct",
				Params: []string{
					"int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64",
					"int8", "Small2", "int16",
				},
			}, typeMap)
			for i, want := range []int64{0, 8, 16} {
				if got := s2Lay.Params[8+i].StackOff; got != want {
					t.Errorf("Small2 param %d StackOff = %d, want %d", 8+i, got, want)
				}
			}
			if s2Lay.StackSize != 32 {
				t.Errorf("Small2 StackSize = %d, want 32", s2Lay.StackSize)
			}
			medLay := cfg.layout(BindingFunc{
				Name: "MediumStruct",
				Params: []string{
					"int32", "int32", "int32", "int32", "int32", "int32", "int32", "int32",
					"int32", "Medium", "int8",
				},
			}, typeMap)
			for i, want := range []int64{0, 8, 24} {
				if got := medLay.Params[8+i].StackOff; got != want {
					t.Errorf("Medium param %d StackOff = %d, want %d", 8+i, got, want)
				}
			}
			if medLay.StackSize != 32 {
				t.Errorf("Medium StackSize = %d, want 32", medLay.StackSize)
			}

			// PassByPtr hidden pointers on the stack align to 8 bytes after narrow scalars.
			ptrLay := cfg.layout(BindingFunc{
				Name: "LargeAfterNarrow",
				Params: []string{
					"int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64",
					"int8", "Large", "int8",
				},
			}, typeMap)
			for i, want := range []int64{0, 8, 16} {
				if got := ptrLay.Params[8+i].StackOff; got != want {
					t.Errorf("PassByPtr param %d StackOff = %d, want %d", 8+i, got, want)
				}
			}
			if !ptrLay.Params[9].PassByPtr || ptrLay.StackSize != 32 {
				t.Errorf("PassByPtr layout = %+v, want PassByPtr=true StackSize=32", ptrLay)
			}
		})
	}
}

// TestCallbackLargeFrameARM64 verifies that arm64 callback trampolines whose
// frame size is not encodable as an ADD/SUB immediate save the C callee-saved
// R27 (the assembler's REGTMP) before the large SP adjustment and restore it
// after undoing that adjustment.
func TestCallbackLargeFrameARM64(t *testing.T) {
	typeMap := map[string]BindingType{
		"Big": {Name: "Big", Def: "struct {\n\ta [4096]byte\n}", Size: 4096, Align: 8},
	}
	fn := BindingFunc{Name: "GoBig", Params: []string{"Big", "int64"}, Ret: "int32"}
	for _, goos := range []string{"linux", "darwin", "windows"} {
		t.Run(goos, func(t *testing.T) {
			var buf bytes.Buffer
			buf.WriteString("#include \"textflag.h\"\n\n")
			getABI(goos, "arm64").writeCallbackTrampoline(&buf, fn, typeMap)
			src := buf.String()

			// Go args: Big at 0, int64 at 4096, ret at 4104, ctxt at 4112;
			// frameSize = 4128, total = 48 + frameSize = 4176.
			want := []string{
				"STP.W\t(R29, R30), -16(RSP)",
				"MOVD.W\tR27, -16(RSP)",
				"SUB\t$4144, RSP",
				"MOVD\tR30, 0(RSP)",
				"ADD\t$4160, RSP, R29",
				"MOVD\tR1, 4112(RSP)", // int64 param after the 4KB struct
				"CALL\tcrosscall2(SB)",
				"ADD\t$4144, RSP\n",
				"MOVD.P\t16(RSP), R27",
				"LDP.P\t16(RSP), (R29, R30)",
				"RET",
			}
			pos := 0
			for _, w := range want {
				i := strings.Index(src[pos:], w)
				if i < 0 {
					t.Fatalf("arm64 large-frame callback missing %q (in order) in:\n%s", w, src)
				}
				pos += i + len(w)
			}
			if strings.Contains(src, "R27, 8(RSP)") || strings.Contains(src, "8(RSP), R27") {
				t.Errorf("arm64 large-frame callback uses compact R27 slot:\n%s", src)
			}

			// Assemble the trampoline and check the expanded machine code:
			// the first use of R27 must be the save, and the last its restore.
			testenv.MustHaveGoBuild(t)
			dir := t.TempDir()
			sfile := filepath.Join(dir, "cb.s")
			ofile := filepath.Join(dir, "cb.o")
			if err := os.WriteFile(sfile, buf.Bytes(), 0o666); err != nil {
				t.Fatal(err)
			}
			cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-p", "main",
				"-I", filepath.Join(testenv.GOROOT(t), "src", "runtime"), "-o", ofile, sfile)
			cmd.Env = append(os.Environ(), "GOOS="+goos, "GOARCH=arm64")
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("go tool asm: %v\n%s", err, out)
			}
			out, err := testenv.Command(t, testenv.GoToolPath(t), "tool", "objdump", ofile).CombinedOutput()
			if err != nil {
				t.Fatalf("go tool objdump: %v\n%s", err, out)
			}
			var r27 []string
			for _, line := range strings.Split(string(out), "\n") {
				if strings.Contains(line, "R27") {
					r27 = append(r27, strings.Join(strings.Fields(line)[3:], " "))
				}
			}
			if len(r27) < 4 || r27[0] != "MOVD.W R27, -16(RSP)" || r27[len(r27)-1] != "MOVD.P 16(RSP), R27" {
				t.Errorf("R27 uses in assembled arm64 large-frame callback = %q, want save first and restore last\n%s", r27, out)
			}
		})
	}
}

func TestLayoutWin64(t *testing.T) {
	typeMap := map[string]BindingType{
		"Odd3": {Name: "Odd3", Def: "struct {\n\tb [3]uint8\n}", Size: 3, Align: 1},
	}

	w64 := getABI("windows", "amd64")
	w64SlotsLay := w64.layout(BindingFunc{
		Name:   "AbiWin64Slots",
		Params: []string{"int64", "float64", "int64", "float64"},
		Ret:    "float64",
	}, typeMap)
	if w64SlotsLay.StackSize != 32 ||
		!reflect.DeepEqual(w64SlotsLay.Params[0].Regs, []string{"CX"}) ||
		!reflect.DeepEqual(w64SlotsLay.Params[1].Regs, []string{"X1"}) ||
		!reflect.DeepEqual(w64SlotsLay.Params[2].Regs, []string{"R8"}) ||
		!reflect.DeepEqual(w64SlotsLay.Params[3].Regs, []string{"X3"}) {
		t.Errorf("Win64 positional layout = %+v", w64SlotsLay)
	}
	w64Odd3Lay := w64.layout(BindingFunc{Name: "Odd3", Params: []string{"Odd3"}, Ret: "Odd3"}, typeMap)
	if !w64Odd3Lay.Params[0].PassByPtr || !reflect.DeepEqual(w64Odd3Lay.Params[0].Regs, []string{"DX"}) || !w64Odd3Lay.Ret.Indirect || w64.IndirectRetReg != "CX" {
		t.Errorf("Win64 Odd3 layout = %+v", w64Odd3Lay)
	}

	var w64CB bytes.Buffer
	w64.writeCallbackTrampoline(&w64CB, BindingFunc{Name: "Win64CB", Params: []string{"int32"}, Ret: "int32"}, nil)
	for _, want := range []string{"MOVL\tCX, 32(SP)", "LEAQ\t32(SP), DX", "CALL\tcrosscall2(SB)", "MOVL\t36(SP), AX"} {
		if !strings.Contains(w64CB.String(), want) {
			t.Errorf("windows/amd64 callback missing %q:\n%s", want, w64CB.String())
		}
	}
}

// TestStructReturn386 verifies the struct return conventions of the 386 C
// ABIs: Linux and NetBSD always return composites through a hidden pointer
// that the callee pops, Windows returns 1, 2, 4, and 8-byte composites in
// AX or AX:DX, and FreeBSD and OpenBSD (-freg-struct-return) do the same
// except that a composite whose only element is a float or double is
// returned in F0. Every target returns complex64 (_Complex float) in AX:DX
// and complex128 (_Complex double) through a hidden pointer.
func TestStructReturn386(t *testing.T) {
	typeMap := map[string]BindingType{
		"S1":   {Name: "S1", Def: "struct {\n\ta int8\n}", Size: 1, Align: 1},
		"S2":   {Name: "S2", Def: "struct {\n\ta int16\n}", Size: 2, Align: 2},
		"S3":   {Name: "S3", Def: "struct {\n\ta [3]int8\n}", Size: 3, Align: 1},
		"S4":   {Name: "S4", Def: "struct {\n\ta int32\n}", Size: 4, Align: 4},
		"S8":   {Name: "S8", Def: "struct {\n\ta int32\n\tb int32\n}", Size: 8, Align: 4},
		"S12":  {Name: "S12", Def: "struct {\n\ta int32\n\tb int32\n\tc int32\n}", Size: 12, Align: 4},
		"SF":   {Name: "SF", Def: "struct {\n\tf float32\n}", Size: 4, Align: 4},
		"SD":   {Name: "SD", Def: "struct {\n\td float64\n}", Size: 8, Align: 4},
		"SNF":  {Name: "SNF", Def: "struct {\n\ts struct {\n\t\tf float32\n\t}\n}", Size: 4, Align: 4},
		"SFA":  {Name: "SFA", Def: "struct {\n\ta [1]float32\n}", Size: 4, Align: 4},
		"SFF":  {Name: "SFF", Def: "struct {\n\ta float32\n\tb float32\n}", Size: 8, Align: 4},
		"SFI":  {Name: "SFI", Def: "struct {\n\tf float32\n\ti int32\n}", Size: 8, Align: 4},
		"Byte": {Name: "Byte", Def: "[4]uint8", Size: 4, Align: 1},
	}
	const (
		sret = "sret"
		ax   = "AX"
		axdx = "AX,DX"
		f0   = "F0"
	)
	// wantRet returns how goos/386 returns a value of type ret.
	wantRet := func(goos, ret string) string {
		switch ret {
		case "complex64":
			return axdx
		case "complex128":
			return sret
		}
		switch goos {
		case "windows", "freebsd", "openbsd":
			if r, ok := map[string]string{
				"S1": ax, "S2": ax, "S3": sret, "S4": ax, "S8": axdx, "S12": sret,
				"SFF": axdx, "SFI": axdx,
			}[ret]; ok {
				return r
			}
			if goos == "windows" {
				// MSVC does not return float-only structs in F0.
				return map[string]string{"SF": ax, "SD": axdx, "SNF": ax, "SFA": ax}[ret]
			}
			return f0
		}
		return sret
	}
	for _, goos := range []string{"linux", "netbsd", "windows", "freebsd", "openbsd"} {
		t.Run(goos, func(t *testing.T) {
			cfg := getABI(goos, "386")
			for _, ret := range []string{"S1", "S2", "S3", "S4", "S8", "S12", "SF", "SD", "SNF", "SFA", "SFF", "SFI", "complex64", "complex128"} {
				want := wantRet(goos, ret)
				lay := cfg.layout(BindingFunc{Name: "F", Params: []string{"int32"}, Ret: ret}, typeMap)
				got := strings.Join(lay.Ret.Regs, ",")
				if lay.Ret.Indirect || lay.Ret.IndirectOnStack {
					got = sret
				}
				if got != want {
					t.Errorf("%s/386 return %s: got %s, want %s (%+v)", goos, ret, got, want, lay.Ret)
				}
				// The argument follows the hidden pointer, if any.
				if wantOff := map[bool]int64{true: 4, false: 0}[want == sret]; lay.Params[0].StackOff != wantOff {
					t.Errorf("%s/386 return %s: int32 param at %d, want %d", goos, ret, lay.Params[0].StackOff, wantOff)
				}
			}

			// Callbacks pop the hidden pointer (RET $4) exactly when
			// there is one, except on Windows, and load F0 for
			// float-only structs on FreeBSD and OpenBSD.
			for _, ret := range []string{"S8", "S12", "SF", "SD", "complex64", "complex128"} {
				var cb bytes.Buffer
				cfg.writeCallbackTrampoline(&cb, BindingFunc{Name: "CB", Params: []string{"int32"}, Ret: ret}, typeMap)
				src := cb.String()
				pop := strings.Contains(src, "BYTE\t$0xc2\n\tWORD\t$4")
				if wantPop := wantRet(goos, ret) == sret && goos != "windows"; pop != wantPop {
					t.Errorf("%s/386 callback returning %s: pops hidden pointer = %v, want %v:\n%s", goos, ret, pop, wantPop, src)
				}
				if wantRet(goos, ret) == f0 {
					op := map[string]string{"SF": "FMOVF\t", "SD": "FMOVD\t"}[ret]
					if !strings.Contains(src, op) {
						t.Errorf("%s/386 callback returning %s does not load F0 with %s:\n%s", goos, ret, op, src)
					}
				}
			}

			// Callbacks returning complex64 load both halves into
			// AX:DX, and trampolines calling a function returning
			// complex64 store both halves without popping a hidden
			// pointer.
			var c64CB, c64Tramp bytes.Buffer
			cfg.writeCallbackTrampoline(&c64CB, BindingFunc{Name: "CBC64", Params: []string{"int32"}, Ret: "complex64"}, typeMap)
			for _, want := range []string{"), AX\n", "), DX\n"} {
				if src := c64CB.String(); !strings.Contains(src, want) {
					t.Errorf("%s/386 callback returning complex64 missing %q:\n%s", goos, want, src)
				}
			}
			cfg.writeTrampoline(&c64Tramp, BindingFunc{Name: "TC64", CName: "c_c64", Params: []string{"int32"}, Ret: "complex64"}, typeMap)
			src := c64Tramp.String()
			for _, want := range []string{"MOVL\tAX, ", "MOVL\tDX, "} {
				if !strings.Contains(src, want) {
					t.Errorf("%s/386 trampoline returning complex64 missing %q:\n%s", goos, want, src)
				}
			}
			if strings.Contains(src, "SUBL\t$4, SP") {
				t.Errorf("%s/386 trampoline returning complex64 must not expect the callee to pop a hidden pointer:\n%s", goos, src)
			}

			// Trampolines store register-returned composites from AX:DX
			// or F0 into the Go result.
			if goos == "freebsd" || goos == "openbsd" {
				for ret, want := range map[string]string{"S8": "MOVL\tDX, ", "SD": "FMOVDP\tF0, ", "SF": "FMOVFP\tF0, "} {
					var tr bytes.Buffer
					cfg.writeTrampoline(&tr, BindingFunc{Name: "T" + ret, CName: "c_" + ret, Params: []string{"int32"}, Ret: ret}, typeMap)
					if src := tr.String(); !strings.Contains(src, want) {
						t.Errorf("%s/386 trampoline returning %s missing %q:\n%s", goos, ret, want, src)
					}
				}
			}
		})
	}
}

func TestLayout386AndWindowsARM64(t *testing.T) {
	typeMap := map[string]BindingType{
		"Pair8":   {Name: "Pair8", Def: "struct {\n\ta int32\n\tb int32\n}", Size: 8, Align: 4},
		"Double2": {Name: "Double2", Def: "struct {\n\tx float64\n\ty float64\n}", Size: 16, Align: 8},
		"Empty":   {Name: "Empty", Def: "struct {}", Size: 0, Align: 1},
	}

	// 386 Linux vs Windows small struct return (IndirectOnStack vs AX:DX).
	lin386 := getABI("linux", "386").layout(BindingFunc{Name: "F", Params: []string{"int32"}, Ret: "Pair8"}, typeMap)
	if !lin386.Ret.IndirectOnStack || lin386.Params[0].StackOff != 4 {
		t.Errorf("linux/386 sret layout = %+v", lin386)
	}
	win386 := getABI("windows", "386").layout(BindingFunc{Name: "F", Params: []string{"int32"}, Ret: "Pair8"}, typeMap)
	if win386.Ret.Indirect || !slices.Equal(win386.Ret.Regs, []string{"AX", "DX"}) || win386.Params[0].StackOff != 0 {
		t.Errorf("windows/386 8B struct ret layout = %+v", win386)
	}

	// Zero-sized struct return on 386 must NOT allocate a hidden sret slot.
	emptyLay := getABI("linux", "386").layout(BindingFunc{Name: "E", Params: []string{"int32"}, Ret: "Empty"}, typeMap)
	if emptyLay.Ret.Indirect || emptyLay.Ret.IndirectOnStack || emptyLay.Params[0].StackOff != 0 {
		t.Errorf("linux/386 empty struct ret layout = %+v, want no sret", emptyLay)
	}

	// GoOffset on 32-bit targets must cap alignment to PtrSize=4 even when C Align=8.
	d2Lay := getABI("windows", "386").layout(BindingFunc{Name: "D", Params: []string{"int32", "Double2"}, Ret: "Double2"}, typeMap)
	if d2Lay.Params[1].GoOffset != 4 || d2Lay.Ret.GoOffset != 20 {
		t.Errorf("windows/386 (int32, Double2) GoOffsets = %d, %d; want 4, 20", d2Lay.Params[1].GoOffset, d2Lay.Ret.GoOffset)
	}

	// 386 trampolines and callbacks: x87 FMOV, SysV RET $4 vs Windows RET, PE IAT, BX preservation.
	var l386Tramp, l386CB, w386Tramp, w386CB bytes.Buffer
	getABI("linux", "386").writeTrampoline(&l386Tramp, BindingFunc{Name: "Sret", CName: "c_sret", Params: []string{"float64"}, Ret: "Pair8", MayCallback: true}, typeMap)
	for _, want := range []string{"MOVL\tBX, -4(BP)", "MOVL\t-4(BP), BX", "CALL\tc_sret(SB)", "SUBL\t$4, SP", "CALL\t_cgo_topofstack(SB)"} {
		if !strings.Contains(l386Tramp.String(), want) {
			t.Errorf("linux/386 sret trampoline missing %q:\n%s", want, l386Tramp.String())
		}
	}
	getABI("linux", "386").writeCallbackTrampoline(&l386CB, BindingFunc{Name: "CBSret", Params: []string{"int32"}, Ret: "Pair8"}, typeMap)
	l386CBAsm := l386CB.String()
	for _, want := range []string{"MOVL\tBX, -4(BP)", "MOVL\t-4(BP), BX", "BYTE\t$0xc2\n\tWORD\t$4"} {
		if !strings.Contains(l386CBAsm, want) {
			t.Errorf("linux/386 sret callback missing %q:\n%s", want, l386CBAsm)
		}
	}
	if iSave, iCall, iRest := strings.Index(l386CBAsm, "MOVL\tBX, -4(BP)"), strings.Index(l386CBAsm, "CALL\t_cgo_wait_runtime_init_done(SB)"), strings.Index(l386CBAsm, "MOVL\t-4(BP), BX"); !(iSave >= 0 && iSave < iCall && iCall < iRest) {
		t.Errorf("linux/386 callback must save BX before calls and restore after:\n%s", l386CBAsm)
	}
	if testenv.HasGoBuild() {
		dir := t.TempDir()
		sfile := filepath.Join(dir, "cb386.s")
		ofile := filepath.Join(dir, "cb386.o")
		if err := os.WriteFile(sfile, []byte("#include \"textflag.h\"\n\n"+l386CBAsm), 0o666); err != nil {
			t.Fatal(err)
		}
		cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-dynlink", "-p", "main",
			"-I", filepath.Join(testenv.GOROOT(t), "src", "runtime"), "-o", ofile, sfile)
		cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH=386")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("go tool asm -dynlink (linux/386): %v\n%s", err, out)
		}
		out, err := testenv.Command(t, testenv.GoToolPath(t), "tool", "objdump", ofile).CombinedOutput()
		if err != nil {
			t.Fatalf("go tool objdump (linux/386): %v\n%s", err, out)
		}
		var bxUses []string
		for _, line := range strings.Split(string(out), "\n") {
			if strings.Contains(line, "BX") {
				if f := strings.Fields(line); len(f) >= 4 {
					bxUses = append(bxUses, strings.Join(f[3:], " "))
				}
			}
		}
		if len(bxUses) < 3 || bxUses[0] != "MOVL BX, -0x4(BP)" || bxUses[len(bxUses)-1] != "MOVL -0x4(BP), BX" {
			t.Errorf("BX uses in linux/386 -dynlink callback = %q, want save first and restore last\n%s", bxUses, out)
		}
	}
	getABI("windows", "386").writeTrampoline(&w386Tramp, BindingFunc{Name: "Flt", CName: "c_flt", Params: []string{"float32"}, Ret: "float64"}, nil)
	for _, want := range []string{"MOVL\tc_flt(SB), CX\n\tCALL\tCX", "FMOVDP\tF0,"} {
		if !strings.Contains(w386Tramp.String(), want) {
			t.Errorf("windows/386 float trampoline missing %q:\n%s", want, w386Tramp.String())
		}
	}
	getABI("windows", "386").writeCallbackTrampoline(&w386CB, BindingFunc{Name: "CBFlt", Params: []string{"int64"}, Ret: "float32"}, nil)
	for _, want := range []string{"MOVL\tBX, -4(BP)", "MOVL\t-4(BP), BX", "FMOVF\t", "\tRET\n"} {
		if !strings.Contains(w386CB.String(), want) {
			t.Errorf("windows/386 float callback missing %q:\n%s", want, w386CB.String())
		}
	}

	var wArmBuf bytes.Buffer
	getABI("windows", "arm64").writeTrampoline(&wArmBuf, BindingFunc{Name: "_Add", CName: "Add", Params: []string{"int32"}, Ret: "int32"}, nil)
	if !strings.Contains(wArmBuf.String(), "MOVD\tAdd(SB), R12\n\tCALL\t(R12)") {
		t.Errorf("windows/arm64 trampoline missing PE indirect call:\n%s", wArmBuf.String())
	}
}

func TestLayoutARM32(t *testing.T) {
	// The hard-float ABI is used only when GOARM does not select soft float.
	t.Setenv("GOARM", "7")
	typeMap := map[string]BindingType{
		"Double2": {Name: "Double2", Def: "struct {\n\ta float64\n\tb float64\n}", Size: 16, Align: 8},
		"Pair8":   {Name: "Pair8", Def: "struct {\n\ta int32\n\tb int32\n}", Size: 8, Align: 4},
		"Big32":   {Name: "Big32", Def: "struct {\n\ta [8]int32\n}", Size: 32, Align: 4},
	}

	// ARM32 VFP backfill: float32 -> S0, float64 -> F1 (S2..S3), float32 -> S1.
	armCfg := getABI("linux", "arm")
	vfpLay := armCfg.layout(BindingFunc{
		Name:   "VFPBackfill",
		Params: []string{"float32", "float64", "float32", "int32", "int64"},
		Ret:    "float32",
	}, nil)
	if !slices.Equal(vfpLay.Params[0].Regs, []string{"S0"}) ||
		!slices.Equal(vfpLay.Params[1].Regs, []string{"F1"}) ||
		!slices.Equal(vfpLay.Params[2].Regs, []string{"S1"}) ||
		!slices.Equal(vfpLay.Params[3].Regs, []string{"R0"}) ||
		!slices.Equal(vfpLay.Params[4].Regs, []string{"R2", "R3"}) ||
		!slices.Equal(vfpLay.Ret.Regs, []string{"S0"}) {
		t.Errorf("ARM32 VFP backfill + even-pair layout = %+v", vfpLay)
	}

	// ARM32 HFA Double2 param/return in F0..F1 and 8B int struct param in R0..R1 (sret return).
	hfaLay := armCfg.layout(BindingFunc{Name: "HFA", Params: []string{"Double2", "Pair8"}, Ret: "Double2"}, typeMap)
	if !slices.Equal(hfaLay.Params[0].Regs, []string{"F0", "F1"}) ||
		!slices.Equal(hfaLay.Params[1].Regs, []string{"R0", "R1"}) ||
		!slices.Equal(hfaLay.Ret.Regs, []string{"F0", "F1"}) {
		t.Errorf("ARM32 HFA + Pair8 layout = %+v", hfaLay)
	}
	var hfaAsm bytes.Buffer
	armCfg.writeTrampoline(&hfaAsm, BindingFunc{Name: "HFA", CName: "hfa", Params: []string{"Double2"}, Ret: "Double2"}, typeMap)
	if !strings.Contains(hfaAsm.String(), "MOVD\t0(R4), F0\n\tMOVD\t8(R4), F1") ||
		!strings.Contains(hfaAsm.String(), "MOVD\tF0, 16(R4)\n\tMOVD\tF1, 24(R4)") {
		t.Errorf("ARM32 Double2 HFA trampoline = \n%s", hfaAsm.String())
	}

	// ARM32 odd-S register WORD VMOV emission (S1 -> 0xee00ca90 load, 0xee10ca90 store).
	var armOddS bytes.Buffer
	armCfg.writeTrampoline(&armOddS, BindingFunc{Name: "F2", CName: "f2", Params: []string{"float32", "float32"}, Ret: "float32"}, nil)
	if !strings.Contains(armOddS.String(), "WORD\t$0xee00ca90") {
		t.Errorf("ARM32 S1 load missing VMOV WORD:\n%s", armOddS.String())
	}
	var armOddSCb bytes.Buffer
	armCfg.writeCallbackTrampoline(&armOddSCb, BindingFunc{Name: "CbF2", Params: []string{"float32", "float32"}, Ret: "float32"}, nil)
	if !strings.Contains(armOddSCb.String(), "WORD\t$0xee10ca90") {
		t.Errorf("ARM32 callback S1 store missing VMOV WORD:\n%s", armOddSCb.String())
	}
	if !strings.Contains(armOddSCb.String(), "MOVW\tR8, 4(R13)") || !strings.Contains(armOddSCb.String(), "MOVW\t4(R13), R8") {
		t.Errorf("ARM32 callback must save/restore callee-saved R8:\n%s", armOddSCb.String())
	}
	if !strings.Contains(armOddSCb.String(), "MOVW\t$·_cgo_fn_CbF2(SB), R0\n\tMOVW\t0(R0), R0\n\tMOVW\t0(R0), R0") {
		t.Errorf("ARM32 callback must load _cgo_fn via $sym + double deref to avoid -dynlink R9 clobber:\n%s", armOddSCb.String())
	}
	if testenv.HasGoBuild() {
		dir := t.TempDir()
		sfile := filepath.Join(dir, "cbarm.s")
		ofile := filepath.Join(dir, "cbarm.o")
		if err := os.WriteFile(sfile, []byte("#include \"textflag.h\"\n\n"+armOddSCb.String()), 0o666); err != nil {
			t.Fatal(err)
		}
		cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-dynlink", "-p", "main",
			"-I", filepath.Join(testenv.GOROOT(t), "src", "runtime"), "-o", ofile, sfile)
		cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH=arm")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("go tool asm -dynlink (linux/arm): %v\n%s", err, out)
		}
		out, err := testenv.Command(t, testenv.GoToolPath(t), "tool", "objdump", ofile).CombinedOutput()
		if err != nil {
			t.Fatalf("go tool objdump (linux/arm): %v\n%s", err, out)
		}
		for _, line := range strings.Split(string(out), "\n") {
			if f := strings.Fields(line); len(f) >= 4 && strings.Contains(strings.Join(f[3:], " "), "R9") {
				t.Errorf("ARM32 -dynlink callback clobbers C callee-saved R9: %s\n%s", line, out)
			}
		}
	}

	// ARM32 split reg+stack composites (AAPCS32 Rule C.5) and GoOffset alignment for Double2.
	splitLay := armCfg.layout(BindingFunc{Name: "Split", Params: []string{"int32", "int32", "int32", "Pair8"}}, typeMap)
	if !slices.Equal(splitLay.Params[3].Regs, []string{"R3"}) || splitLay.Params[3].StackBytes != 4 || splitLay.Params[3].StackOff != 0 {
		t.Errorf("ARM32 straddling Pair8 = %+v, want Regs=[R3] StackBytes=4 StackOff=0", splitLay.Params[3])
	}
	bigLay := armCfg.layout(BindingFunc{Name: "Big", Params: []string{"Big32"}}, typeMap)
	if !slices.Equal(bigLay.Params[0].Regs, []string{"R0", "R1", "R2", "R3"}) || bigLay.Params[0].StackBytes != 16 || bigLay.Params[0].StackOff != 0 {
		t.Errorf("ARM32 Big32 = %+v, want Regs=[R0..R3] StackBytes=16 StackOff=0", bigLay.Params[0])
	}
	goOffLay := armCfg.layout(BindingFunc{Name: "GoOff", Params: []string{"int32", "Double2"}}, typeMap)
	if goOffLay.Params[1].GoOffset != 4 {
		t.Errorf("ARM32 (int32, Double2) GoOffset = %d, want 4", goOffLay.Params[1].GoOffset)
	}

	// Android/arm soft-float passes floats in GPRs.
	andArm := getABI("android", "arm")
	softLay := andArm.layout(BindingFunc{
		Name:   "SoftFloat",
		Params: []string{"float32", "float64"},
		Ret:    "float64",
	}, nil)
	if !slices.Equal(softLay.Params[0].Regs, []string{"R0"}) ||
		!slices.Equal(softLay.Params[1].Regs, []string{"R2", "R3"}) ||
		!slices.Equal(softLay.Ret.Regs, []string{"R0", "R1"}) {
		t.Errorf("android/arm softfloat layout = %+v", softLay)
	}

	// AAPCS32 Rule C.5 splits a composite between core registers and the
	// stack only while no argument has been placed on the stack yet (NSAA ==
	// SP). Once a VFP argument has spilled, a composite that does not fit in
	// the remaining core registers goes wholly on the stack (Rules C.6-C.8),
	// and later core-register arguments follow it on the stack.
	floats := func(n int, typ string) []string {
		return slices.Repeat([]string{typ}, n)
	}
	c5Types := map[string]BindingType{
		"Big32":     typeMap["Big32"],
		"Pair8":     typeMap["Pair8"],
		"Align8x24": {Name: "Align8x24", Def: "struct {\n\ta int64\n\tb int64\n\tc int64\n}", Size: 24, Align: 8},
		"Align8x16": {Name: "Align8x16", Def: "struct {\n\ta int64\n\tb int64\n}", Size: 16, Align: 8},
	}
	for _, tc := range []struct {
		name      string
		params    []string
		idx       int // index of the composite
		regs      []string
		stackOff  int64
		nextOff   int64 // stack offset of the int32 after the composite, or -1
		nextIsReg bool
	}{
		// The 9th float64 spills to 0(SP), so Big32 cannot split.
		{"Float64Spill_Big32", append(floats(9, "float64"), "Big32", "int32"), 9, nil, 8, 40, false},
		// The 17th float32 spills to 0(SP); the 8-byte-aligned composite
		// goes wholly on the stack at the next 8-byte boundary.
		{"Float32Spill_Align8x24", append(floats(17, "float32"), "Align8x24", "int32"), 17, nil, 8, 32, false},
		// A composite that fits in core registers still uses them (Rule C.4).
		{"Float64Spill_Pair8", append(floats(9, "float64"), "Pair8"), 9, []string{"R0", "R1"}, 0, -1, false},
		// Rule C.3 rounds the NCRN up to R4, so the composite goes wholly on the stack.
		{"Int32x3_Align8x16", []string{"int32", "int32", "int32", "Align8x16"}, 3, nil, 0, -1, false},
	} {
		lay := armCfg.layout(BindingFunc{Name: "C5", Params: tc.params}, c5Types)
		p := lay.Params[tc.idx]
		if !slices.Equal(p.Regs, tc.regs) || p.StackBytes != 0 || (tc.regs == nil && p.StackOff != tc.stackOff) {
			t.Errorf("ARM32 %s: composite = %+v, want Regs=%v StackBytes=0 StackOff=%d", tc.name, p, tc.regs, tc.stackOff)
		}
		if tc.nextOff >= 0 {
			if next := lay.Params[tc.idx+1]; len(next.Regs) != 0 || next.StackOff != tc.nextOff {
				t.Errorf("ARM32 %s: following int32 = %+v, want on stack at %d", tc.name, next, tc.nextOff)
			}
		}
	}
}

// TestARM32SoftFloat checks that the soft-float (softfp) ARM C ABI, which
// passes and returns floating-point values in core registers, is used on
// android/arm and openbsd/arm, and on other targets when GOARM selects
// soft float (GOARM=5 or a ",softfloat" suffix).
func TestARM32SoftFloat(t *testing.T) {
	typeMap := map[string]BindingType{
		"Double2": {Name: "Double2", Def: "struct {\n\ta float64\n\tb float64\n}", Size: 16, Align: 8},
	}
	fn := BindingFunc{Name: "SoftFP", CName: "softfp", Params: []string{"float32", "float64", "Double2"}, Ret: "Double2"}
	for _, tc := range []struct {
		goos, goarm string
		soft        bool
	}{
		{"linux", "7", false},
		{"linux", "6", false},
		{"linux", "5", true},
		{"linux", "7,softfloat", true},
		{"linux", "5,hardfloat", false},
		{"openbsd", "7", true},
		{"android", "7", true},
	} {
		t.Run(tc.goos+"_"+tc.goarm, func(t *testing.T) {
			t.Setenv("GOARM", tc.goarm)
			cfg := getABI(tc.goos, "arm")
			lay := cfg.layout(fn, typeMap)
			var tramp, cb bytes.Buffer
			cfg.writeTrampoline(&tramp, fn, typeMap)
			cfg.writeCallbackTrampoline(&cb, BindingFunc{Name: "CbSoftFP", Params: fn.Params, Ret: "float64"}, typeMap)
			if !tc.soft {
				if !slices.Equal(lay.Params[0].Regs, []string{"S0"}) || !slices.Equal(lay.Params[1].Regs, []string{"F1"}) {
					t.Errorf("%s/arm GOARM=%s: hard-float layout = %+v", tc.goos, tc.goarm, lay)
				}
				return
			}
			// The hidden sret pointer takes R0, the float32 R1, the
			// float64 R2:R3, and the 8-byte-aligned Double2 the stack.
			if !lay.Ret.Indirect ||
				!slices.Equal(lay.Params[0].Regs, []string{"R1"}) || lay.Params[0].IsFloat ||
				!slices.Equal(lay.Params[1].Regs, []string{"R2", "R3"}) || lay.Params[1].IsFloat ||
				len(lay.Params[2].Regs) != 0 || lay.Params[2].StackOff != 0 {
				t.Errorf("%s/arm GOARM=%s: soft-float layout = %+v", tc.goos, tc.goarm, lay)
			}
			for name, out := range map[string]string{"trampoline": tramp.String(), "callback": cb.String()} {
				for _, op := range []string{"MOVF\t", "MOVD\t", "WORD\t"} {
					if strings.Contains(out, op) {
						t.Errorf("%s/arm GOARM=%s: soft-float %s uses VFP instruction %q:\n%s", tc.goos, tc.goarm, name, op, out)
					}
				}
			}
		})
	}
}

func TestLayoutMIPS32(t *testing.T) {
	// The hard-float ABI is used only when GOMIPS does not select soft float.
	t.Setenv("GOMIPS", "hardfloat")
	typeMap := map[string]BindingType{
		"Pair8": {Name: "Pair8", Def: "struct {\n\ta int32\n\tb int32\n}", Size: 8, Align: 4},
		"Odd5":  {Name: "Odd5", Def: "struct {\n\ta [5]byte\n}", Size: 5, Align: 1},
	}

	// MIPS32 O32 float rule: int32 first disables FPRs for subsequent float64.
	mipsCfg := getABI("linux", "mipsle")
	mipsLay := mipsCfg.layout(BindingFunc{Name: "M", Params: []string{"int32", "float64"}}, nil)
	if !slices.Equal(mipsLay.Params[0].Regs, []string{"R4"}) || !slices.Equal(mipsLay.Params[1].Regs, []string{"R6", "R7"}) {
		t.Errorf("mipsle O32 int+double layout = %+v", mipsLay)
	}
	// MIPS32 O32 (float64, float32, int32) -> (F12, F14, R7).
	mipsDF := mipsCfg.layout(BindingFunc{Name: "DF", Params: []string{"float64", "float32", "int32"}}, nil)
	if !slices.Equal(mipsDF.Params[0].Regs, []string{"F12"}) ||
		!slices.Equal(mipsDF.Params[1].Regs, []string{"F14"}) ||
		!slices.Equal(mipsDF.Params[2].Regs, []string{"R7"}) {
		t.Errorf("mipsle O32 (double, float, int) layout = %+v", mipsDF)
	}
	// MIPS32 O32: all structs return via hidden sret in R4, and sret disables FPRs.
	mipsRet := mipsCfg.layout(BindingFunc{Name: "RetPair", Params: []string{"float32", "float64"}, Ret: "Pair8"}, typeMap)
	if !mipsRet.Ret.Indirect ||
		!slices.Equal(mipsRet.Params[0].Regs, []string{"R5"}) ||
		!slices.Equal(mipsRet.Params[1].Regs, []string{"R6", "R7"}) {
		t.Errorf("mipsle O32 sret+floats layout = %+v", mipsRet)
	}
	// MIPS32 O32 split reg+stack aggregate.
	mipsSplit := mipsCfg.layout(BindingFunc{Name: "MSplit", Params: []string{"int32", "int32", "int32", "Pair8"}}, typeMap)
	if !slices.Equal(mipsSplit.Params[3].Regs, []string{"R7"}) || mipsSplit.Params[3].StackBytes != 4 || mipsSplit.Params[3].StackOff != 16 {
		t.Errorf("mipsle O32 straddling Pair8 = %+v, want Regs=[R7] StackBytes=4 StackOff=16", mipsSplit.Params[3])
	}
	// Big-endian mips: left-justified partial struct in R5, strict-align byte loads on Odd5 at GoOffset=1, PIC R25 call.
	mipsBE := getABI("linux", "mips")
	var mipsBEAsm bytes.Buffer
	mipsBE.writeTrampoline(&mipsBEAsm, BindingFunc{Name: "MBE", CName: "mbe", Params: []string{"int8", "Odd5"}}, typeMap)
	if out := mipsBEAsm.String(); !strings.Contains(out, "SLL\t$24, R6, R6") ||
		!strings.Contains(out, "MOVBU\t1(R16), R5") ||
		!strings.Contains(out, "MOVW\t$mbe(SB), R25\n\tCALL\t(R25)") {
		t.Errorf("mips BE trampoline unexpected:\n%s", out)
	}

	// O32 returns _Complex float and _Complex double in F0 and F2 (MOVF or
	// MOVD per half) with hard float, unlike structs of two floats or
	// doubles, which are returned through a hidden pointer in R4. Complex
	// arguments are passed in GPRs like structs.
	typeMap["Float2"] = BindingType{Name: "Float2", Def: "struct {\n\ta float32\n\tb float32\n}", Size: 8, Align: 4}
	typeMap["Double2"] = BindingType{Name: "Double2", Def: "struct {\n\ta float64\n\tb float64\n}", Size: 16, Align: 8}
	for _, arch := range []string{"mips", "mipsle"} {
		cfg := getABI("linux", arch)
		// The result follows the float32 and float64 arguments at Go offset 12.
		for _, tc := range []struct {
			ret, op  string
			store    []string
			indirect bool
		}{
			{"complex64", "MOVF", []string{"MOVF\tF0, 12(R16)\n", "MOVF\tF2, 16(R16)\n"}, false},
			{"complex128", "MOVD", []string{"MOVD\tF0, 12(R16)\n", "MOVD\tF2, 20(R16)\n"}, false},
			{"Float2", "", nil, true},
			{"Double2", "", nil, true},
		} {
			fn := BindingFunc{Name: "C" + tc.ret, CName: "c_" + tc.ret, Params: []string{"float32", "float64"}, Ret: tc.ret}
			lay := cfg.layout(fn, typeMap)
			if tc.indirect {
				if !lay.Ret.Indirect || !slices.Equal(lay.Params[0].Regs, []string{"R5"}) {
					t.Errorf("%s O32 %s return = %+v, want hidden pointer in R4 and float32 in R5", arch, tc.ret, lay)
				}
				continue
			}
			if lay.Ret.Indirect || !lay.Ret.IsFloat || !slices.Equal(lay.Ret.Regs, []string{"F0", "F2"}) ||
				!slices.Equal(lay.Params[0].Regs, []string{"F12"}) || !slices.Equal(lay.Params[1].Regs, []string{"F14"}) {
				t.Errorf("%s O32 %s return = %+v, want F0, F2 with float args in F12, F14", arch, tc.ret, lay)
			}
			var tramp, cb bytes.Buffer
			cfg.writeTrampoline(&tramp, fn, typeMap)
			cbFn := BindingFunc{Name: "Cb" + tc.ret, Params: fn.Params, Ret: tc.ret}
			cfg.writeCallbackTrampoline(&cb, cbFn, typeMap)
			for _, want := range tc.store {
				if !strings.Contains(tramp.String(), want) {
					t.Errorf("%s O32 trampoline returning %s missing %q:\n%s", arch, tc.ret, want, tramp.String())
				}
			}
			for _, want := range []string{"), F0\n", "), F2\n"} {
				if !strings.Contains(cb.String(), tc.op+"\t") || !strings.Contains(cb.String(), want) {
					t.Errorf("%s O32 callback returning %s missing %s load %q:\n%s", arch, tc.ret, tc.op, want, cb.String())
				}
			}
			if testenv.HasGoBuild() {
				dir := t.TempDir()
				sfile := filepath.Join(dir, "c.s")
				ofile := filepath.Join(dir, "c.o")
				src := "#include \"textflag.h\"\n\n" + tramp.String() + cb.String()
				if err := os.WriteFile(sfile, []byte(src), 0o666); err != nil {
					t.Fatal(err)
				}
				cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-p", "main",
					"-I", filepath.Join(testenv.GOROOT(t), "src", "runtime"), "-o", ofile, sfile)
				cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH="+arch)
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Errorf("go tool asm (linux/%s) returning %s: %v\n%s\n%s", arch, tc.ret, err, out, src)
				}
			}
		}
		argLay := cfg.layout(BindingFunc{Name: "CArgs", Params: []string{"complex64", "float64"}}, typeMap)
		if !slices.Equal(argLay.Params[0].Regs, []string{"R4", "R5"}) || argLay.Params[0].IsFloat ||
			!slices.Equal(argLay.Params[1].Regs, []string{"R6", "R7"}) {
			t.Errorf("%s O32 complex64 argument layout = %+v, want R4, R5 then float64 in R6, R7", arch, argLay)
		}
	}

	// GOMIPS=softfloat passes floats in GPRs.
	t.Setenv("GOMIPS", "softfloat")
	mipsSF := getABI("linux", "mipsle")
	sfLay := mipsSF.layout(BindingFunc{Name: "SF", Params: []string{"float32", "float64"}, Ret: "float64"}, nil)
	if !slices.Equal(sfLay.Params[0].Regs, []string{"R4"}) ||
		!slices.Equal(sfLay.Params[1].Regs, []string{"R6", "R7"}) ||
		!slices.Equal(sfLay.Ret.Regs, []string{"R2", "R3"}) {
		t.Errorf("GOMIPS=softfloat layout = %+v", sfLay)
	}
	// With soft float, complex64 is returned in R2, R3 and complex128 in
	// R2..R5, not through a hidden pointer.
	for _, arch := range []string{"mips", "mipsle"} {
		cfg := getABI("linux", arch)
		for ret, want := range map[string][]string{"complex64": {"R2", "R3"}, "complex128": {"R2", "R3", "R4", "R5"}} {
			fn := BindingFunc{Name: "SF" + ret, CName: "sf_" + ret, Params: []string{"int32"}, Ret: ret}
			lay := cfg.layout(fn, nil)
			if lay.Ret.Indirect || !slices.Equal(lay.Ret.Regs, want) || !slices.Equal(lay.Params[0].Regs, []string{"R4"}) {
				t.Errorf("%s GOMIPS=softfloat %s return = %+v, want %v", arch, ret, lay, want)
			}
			var tramp, cb bytes.Buffer
			cfg.writeTrampoline(&tramp, fn, nil)
			cfg.writeCallbackTrampoline(&cb, BindingFunc{Name: "CbSF" + ret, Params: fn.Params, Ret: ret}, nil)
			for name, out := range map[string]string{"trampoline": tramp.String(), "callback": cb.String()} {
				if strings.Contains(out, "MOVF\t") || strings.Contains(out, "MOVD\t") {
					t.Errorf("%s GOMIPS=softfloat %s returning %s uses FPU instructions:\n%s", arch, name, ret, out)
				}
			}
		}
	}
}

func TestLayoutAndTrampolineS390X(t *testing.T) {
	typeMap := map[string]BindingType{
		"Pair16":  {Name: "Pair16", Def: "struct {\n\ta int64\n\tb int64\n}", Size: 16, Align: 8},
		"Odd3":    {Name: "Odd3", Def: "struct {\n\tb [3]uint8\n}", Size: 3, Align: 1},
		"SingleF": {Name: "SingleF", Def: "struct {\n\tf float32\n}", Size: 4, Align: 4},
		"Box4":    {Name: "Box4", Def: "struct {\n\tx int32\n}", Size: 4, Align: 4},
		"Box8":    {Name: "Box8", Def: "struct {\n\tx int64\n}", Size: 8, Align: 8},
		"Empty":   {Name: "Empty", Def: "struct{}", Size: 0, Align: 1},
	}

	// s390x returns all structs via sret, passes single-float structs in FPRs,
	// passes non-{1,2,4,8}-byte structs by hidden pointer, skips empty structs,
	// right-justifies spilled 1/2/4B structs (including SingleF), and avoids R10/R11.
	s390 := getABI("linux", "s390x")
	s390Lay := s390.layout(BindingFunc{Name: "S", Params: []string{"Pair16", "Odd3", "SingleF", "Box8", "Empty"}, Ret: "Box8"}, typeMap)
	if !s390Lay.Params[0].PassByPtr || !slices.Equal(s390Lay.Params[0].Regs, []string{"R3"}) ||
		!s390Lay.Params[1].PassByPtr || !slices.Equal(s390Lay.Params[1].Regs, []string{"R4"}) ||
		!s390Lay.Params[2].IsFloat || !slices.Equal(s390Lay.Params[2].Regs, []string{"F0"}) ||
		s390Lay.Params[3].PassByPtr || !slices.Equal(s390Lay.Params[3].Regs, []string{"R5"}) ||
		s390Lay.Params[4].PassByPtr || len(s390Lay.Params[4].Regs) != 0 || s390Lay.Params[4].Size != 0 ||
		!s390Lay.Ret.Indirect || s390.IndirectRetReg != "R2" {
		t.Errorf("s390x struct layout = %+v", s390Lay)
	}
	var s390Tramp, s390Cb bytes.Buffer
	s390Fn := BindingFunc{
		Name:   "S390Spill",
		CName:  "s390_spill",
		Params: []string{"float64", "float64", "float64", "float64", "SingleF", "int64", "int64", "int64", "int64", "int64", "Box4"},
		Ret:    "int32",
	}
	s390.writeTrampoline(&s390Tramp, s390Fn, typeMap)
	s390.writeCallbackTrampoline(&s390Cb, s390Fn, typeMap)
	for _, nameOut := range [][2]string{{"trampoline", s390Tramp.String()}, {"callback", s390Cb.String()}} {
		if strings.Contains(nameOut[1], "R10") || strings.Contains(nameOut[1], "R11") {
			t.Errorf("s390x %s must not touch assembler scratch R10/R11:\n%s", nameOut[0], nameOut[1])
		}
	}
	// Spilled SingleF (stackOff=160) and Box4 (stackOff=168) are right-justified (MOVWZ -> MOVD in trampoline, +4 offset in callback).
	if out := s390Tramp.String(); !strings.Contains(out, "MOVWZ\t32(R8), R1\n\tMOVD\tR1, 160(R15)") ||
		!strings.Contains(out, "MOVWZ\t80(R8), R1\n\tMOVD\tR1, 168(R15)") {
		t.Errorf("s390x spilled SingleF/Box4 trampoline not right-justified:\n%s", out)
	}
	// In callback, callerSPOff = 256; SingleF (stackOff=160) is read at 256+160+4=420 and Box4 (stackOff=168) at 256+168+4=428.
	if out := s390Cb.String(); !strings.Contains(out, "MOVWZ\t420(R15), R1") ||
		!strings.Contains(out, "MOVWZ\t428(R15), R1") {
		t.Errorf("s390x callback spilled SingleF/Box4 expected right-justified stack offsets 420 and 428:\n%s", out)
	}

	// Only a struct of 4 or 8 bytes whose only member (not counting
	// structs.HostLayout) is a float, a double, or such a struct is passed
	// in an FPR. One-member arrays, unions (represented as arrays), structs
	// with another (even empty) member, and structs holding a complex
	// number are passed in a GPR. _Complex float, _Complex double, and
	// other sizes are passed by reference.
	fpTypes := map[string]BindingType{
		"SingleF":     typeMap["SingleF"],
		"SingleD":     {Name: "SingleD", Def: "struct {\n\td float64\n}", Size: 8, Align: 8},
		"SingleFHost": {Name: "SingleFHost", Def: "struct {\n\t_ structs.HostLayout\n\tf float32\n}", Size: 4, Align: 4},
		"NestedF":     {Name: "NestedF", Def: "struct {\n\tinner SingleF\n}", Size: 4, Align: 4},
		"AlignedF8":   {Name: "AlignedF8", Def: "struct {\n\tf float32\n}", Size: 8, Align: 8},
		"AlignedF16":  {Name: "AlignedF16", Def: "struct {\n\tf float32\n}", Size: 16, Align: 16},
		"ArrF1":       {Name: "ArrF1", Def: "struct {\n\ta [1]float32\n}", Size: 4, Align: 4},
		"ArrD1":       {Name: "ArrD1", Def: "struct {\n\ta [1]float64\n}", Size: 8, Align: 8},
		"NestedArrF1": {Name: "NestedArrF1", Def: "struct {\n\tinner struct {\n\t\ta [1]float32\n\t}\n}", Size: 4, Align: 4},
		"ArrSingleF1": {Name: "ArrSingleF1", Def: "struct {\n\tinner [1]SingleF\n}", Size: 4, Align: 4},
		"UnionF":      {Name: "UnionF", Def: "[1]float32", Size: 4, Align: 4},
		"EmptyAndF":   {Name: "EmptyAndF", Def: "struct {\n\te struct{}\n\tf float32\n}", Size: 4, Align: 4},
		"WrapC64":     {Name: "WrapC64", Def: "struct {\n\tc complex64\n}", Size: 8, Align: 4},
	}
	const (
		fpr = "FPR"
		gpr = "GPR"
		ref = "ref"
	)
	for typ, want := range map[string]string{
		"SingleF": fpr, "SingleD": fpr, "SingleFHost": fpr, "NestedF": fpr, "AlignedF8": fpr,
		"ArrF1": gpr, "ArrD1": gpr, "NestedArrF1": gpr, "ArrSingleF1": gpr, "UnionF": gpr, "EmptyAndF": gpr, "WrapC64": gpr,
		"complex64": ref, "complex128": ref, "AlignedF16": ref,
	} {
		lay := s390.layout(BindingFunc{Name: "P", Params: []string{typ}}, fpTypes)
		p := lay.Params[0]
		var got string
		switch {
		case p.PassByPtr && slices.Equal(p.Regs, []string{"R2"}):
			got = ref
		case !p.PassByPtr && p.IsFloat && slices.Equal(p.Regs, []string{"F0"}):
			got = fpr
		case !p.PassByPtr && !p.IsFloat && slices.Equal(p.Regs, []string{"R2"}):
			got = gpr
		}
		if got != want {
			t.Errorf("s390x %s argument = %+v, want %s", typ, p, want)
		}
	}
	// A complex64 is passed by reference on the stack once R2..R6 are used.
	spillLay := s390.layout(BindingFunc{Name: "P", Params: []string{"int64", "int64", "int64", "int64", "int64", "complex64"}}, nil)
	if p := spillLay.Params[5]; !p.PassByPtr || len(p.Regs) != 0 || p.StackOff != 160 {
		t.Errorf("s390x spilled complex64 argument = %+v, want reference at 160(R15)", p)
	}
	// Every composite, including complex numbers and single-float structs,
	// is returned through a hidden pointer in R2, moving the first argument
	// to R3.
	for _, ret := range []string{"complex64", "complex128", "SingleF"} {
		lay := s390.layout(BindingFunc{Name: "R", Params: []string{"int64"}, Ret: ret}, fpTypes)
		if !lay.Ret.Indirect || !slices.Equal(lay.Params[0].Regs, []string{"R3"}) {
			t.Errorf("s390x returning %s = %+v, want hidden pointer in R2 and int64 in R3", ret, lay)
		}
	}
	var c64Tramp, c64Cb bytes.Buffer
	c64Fn := BindingFunc{Name: "C64", CName: "c64", Params: []string{"complex64"}, Ret: "complex64"}
	s390.writeTrampoline(&c64Tramp, c64Fn, nil)
	s390.writeCallbackTrampoline(&c64Cb, BindingFunc{Name: "CbC64", Params: c64Fn.Params, Ret: c64Fn.Ret}, nil)
	if out := c64Tramp.String(); !strings.Contains(out, "ADD\t$8, R8, R2\n") || !strings.Contains(out, "ADD\t$0, R8, R3\n") {
		t.Errorf("s390x trampoline for complex64 must pass the result buffer in R2 and the argument by reference in R3:\n%s", out)
	}
	for name, out := range map[string]string{"trampoline": c64Tramp.String(), "callback": c64Cb.String()} {
		if strings.Contains(out, "R10") || strings.Contains(out, "R11") || strings.Contains(out, "FMOV") {
			t.Errorf("s390x %s for complex64 must not use R10, R11, or FPRs:\n%s", name, out)
		}
	}
}

// allABITargets lists every supported OS/arch ABI target.
var allABITargets = [][2]string{
	{"linux", "amd64"}, {"windows", "amd64"},
	{"linux", "arm64"}, {"darwin", "arm64"}, {"windows", "arm64"},
	{"linux", "386"}, {"windows", "386"},
	{"linux", "arm"}, {"android", "arm"}, {"openbsd", "arm"}, {"linux", "mips"}, {"linux", "mipsle"},
	{"linux", "s390x"},
}

// TestIndirectAndStaticTrampolines checks that //cgo:dynload methods call
// through the function address at offset 0 of the argument struct, and that
// static symbols are called directly, on every supported target.
func TestIndirectAndStaticTrampolines(t *testing.T) {
	for _, target := range allABITargets {
		cfg := getABI(target[0], target[1])
		var ind bytes.Buffer
		cfg.writeTrampoline(&ind, BindingFunc{Name: "dyn_L_F", Params: []string{"int32", "int32"}, Ret: "int32", Indirect: true}, nil)
		if out := ind.String(); strings.Count(out, "(SB)") != 1 {
			t.Errorf("%s/%s: indirect trampoline references a symbol other than itself:\n%s", target[0], target[1], out)
		}
		// A static symbol is called like any other external symbol, except
		// on Windows, where other symbols are called through import slots.
		var st, dyn bytes.Buffer
		cfg.writeTrampoline(&st, BindingFunc{Name: "S", CName: "s_fn", Params: []string{"int32"}, Ret: "int32", Static: true}, nil)
		cfg.writeTrampoline(&dyn, BindingFunc{Name: "S", CName: "s_fn", Params: []string{"int32"}, Ret: "int32"}, nil)
		if target[0] == "windows" {
			if out := st.String(); !strings.Contains(out, "CALL\ts_fn(SB)") {
				t.Errorf("%s/%s: static trampoline does not call s_fn directly:\n%s", target[0], target[1], out)
			}
		} else if st.String() != dyn.String() {
			t.Errorf("%s/%s: static trampoline\n%s\ndiffers from dynamic trampoline\n%s", target[0], target[1], st.String(), dyn.String())
		}
	}
}
