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
