// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
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
		{"[]byte", 24, 8, false, false, false},
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

	// Odd-sized struct loads/stores must decompose without 8-byte overreads.
	checkOdd5No8ByteMoves(t, "amd64")
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
}
