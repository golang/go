// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"encoding/xml"
	"reflect"
	"strings"
	"testing"

	"simd/archsimd/_gen/unify"

	"golang.org/x/arch/arm64/instgen/xmlspec"
)

// sizeTable is the XML explanation table shared by ADD/FADD/etc. that maps the
// <T> arrangement symbol to the element specifiers B/H/S/D.
const sizeTable = `
  <explanations>
    <explanation>
      <symbol link="t">&lt;T&gt;</symbol>
      <definition>
        <table><tgroup><tbody>
          <row><entry class="symbol">B</entry></row>
          <row><entry class="symbol">H</entry></row>
          <row><entry class="symbol">S</entry></row>
          <row><entry class="symbol">D</entry></row>
        </tbody></tgroup></table>
      </definition>
    </explanation>
  </explanations>`

// addUnpred is ADD (vectors, unpredicated): ADD <Zd>.<T>, <Zn>.<T>, <Zm>.<T>.
const addUnpred = `<instructionsection id="add_z_zz" title="ADD (vectors, unpredicated)" type="instruction">
  <docvars>
    <docvar key="instr-class" value="sve"/>
    <docvar key="mnemonic" value="ADD"/>
  </docvars>
  <desc><authored><para>Add active elements of the second source to the first.</para></authored></desc>
  <classes><iclass><encoding name="add_z_zz">
    <asmtemplate><text>ADD  </text><a link="zd">&lt;Zd&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="zm">&lt;Zm&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
  </encoding></iclass></classes>` + sizeTable + `</instructionsection>`

// addPred is ADD (vectors, predicated): ADD <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>.
// It must be skipped by the current draft (governing predicate).
const addPred = `<instructionsection id="add_z_p_zz" title="ADD (vectors, predicated)" type="instruction">
  <docvars>
    <docvar key="instr-class" value="sve"/>
    <docvar key="mnemonic" value="ADD"/>
  </docvars>
  <classes><iclass><encoding name="add_z_p_zz">
    <asmtemplate><text>ADD  </text><a link="zdn">&lt;Zdn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="pg">&lt;Pg&gt;</a><text>/M, </text><a link="zdn">&lt;Zdn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="zm">&lt;Zm&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
  </encoding></iclass></classes>` + sizeTable + `</instructionsection>`

// faddUnpred is FADD (vectors, unpredicated): FADD <Zd>.<T>, <Zn>.<T>, <Zm>.<T>.
// Its size table only lists H/S/D.
const faddUnpred = `<instructionsection id="fadd_z_zz" title="FADD (vectors, unpredicated)" type="instruction">
  <desc><brief><para>Floating-point add (unpredicated)</para></brief></desc>
  <docvars>
    <docvar key="instr-class" value="sve"/>
    <docvar key="mnemonic" value="FADD"/>
  </docvars>
  <classes><iclass><encoding name="fadd_z_zz">
    <asmtemplate><text>FADD  </text><a link="zd">&lt;Zd&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="zm">&lt;Zm&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
  </encoding></iclass></classes>
  <explanations>
    <explanation>
      <symbol link="t">&lt;T&gt;</symbol>
      <definition><table><tgroup><tbody>
        <row><entry class="symbol">H</entry></row>
        <row><entry class="symbol">S</entry></row>
        <row><entry class="symbol">D</entry></row>
      </tbody></tgroup></table></definition>
    </explanation>
  </explanations>
</instructionsection>`

func parse(t *testing.T, x string) *Instruction {
	t.Helper()
	var ip xmlspec.InstructionParsed
	if err := xml.Unmarshal([]byte(x), &ip); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return &Instruction{Instruction: ip.Instruction}
}

// briefInst builds a minimal SVE instruction with the given mnemonic and brief
// description, for testing signedness classification.
func briefInst(t *testing.T, mnemonic, brief string) *Instruction {
	t.Helper()
	return parse(t, `<instructionsection id="x" title="x" type="instruction">
	  <desc><brief><para>`+brief+`</para></brief></desc>
	  <classes><iclass>
	    <docvars><docvar key="instr-class" value="sve"/><docvar key="mnemonic" value="`+mnemonic+`"/></docvars>
	  </iclass></classes>
	</instructionsection>`)
}

func TestSignedness(t *testing.T) {
	cases := []struct{ mn, brief, want string }{
		// Agnostic: no "signed"/"unsigned" in the brief.
		{"ADD", "Add (predicated)", ""},
		{"MUL", "Multiply (unpredicated)", ""},
		{"EOR", "Bitwise exclusive-OR (predicated)", ""},
		// A signed/unsigned *immediate* is about the immediate, not the lane.
		{"DUP", "Move signed integer immediate to vector elements", ""},
		// "sign bits" is not the word "signed".
		{"CLS", "Count leading sign bits (predicated)", ""},
		// Signedness-specific from the brief.
		{"SMAX", "Signed maximum (predicated)", "int"},
		{"UMAX", "Unsigned maximum (predicated)", "uint"},
		{"SQDMULH", "Signed saturating doubling multiply high (unpredicated)", "int"},
		{"SCVTF", "Signed integer convert to floating-point (predicated)", "int"},
		// Shift family and FLOGB name signedness differently -> explicit.
		{"ASR", "Arithmetic shift right (predicated)", "int"},
		{"LSR", "Logical shift right (predicated)", "uint"},
		{"FLOGB", "Floating-point base 2 logarithm as integer (predicated)", "int"},
	}
	for _, c := range cases {
		if got := briefInst(t, c.mn, c.brief).signedness(); got != c.want {
			t.Errorf("%s (%q): signedness=%q, want %q", c.mn, c.brief, got, c.want)
		}
	}
}

func TestIsFloatBrief(t *testing.T) {
	cases := []struct {
		brief string
		want  bool
	}{
		{"Floating-point add (predicated)", true},
		{"Double-precision convert to single-precision, rounding to odd", true}, // FCVTX
		{"Half-precision multiply-add to single-precision", true},               // FMLALT
		{"8-bit floating-point convert to BFloat16", true},                      // BF1CVT
		{"Add (predicated)", false},                                             // ADD
		{"Multiply (unpredicated)", false},                                      // MUL
		{"Scalar index of first true predicate element (predicated)", false},    // FIRSTP: F-prefixed but integer
		{"Count leading sign bits (predicated)", false},                         // CLS
	}
	for _, c := range cases {
		if got := isFloatBrief(c.brief); got != c.want {
			t.Errorf("isFloatBrief(%q) = %v, want %v", c.brief, got, c.want)
		}
	}
}

func TestMnemonicAndClass(t *testing.T) {
	inst := parse(t, addUnpred)
	if got := inst.mnemonic(); got != "ADD" {
		t.Errorf("mnemonic = %q, want ADD", got)
	}
	if !inst.isSVE() {
		t.Errorf("isSVE = false, want true")
	}
	if got := inst.cpuFeature(); got != "SVE" {
		t.Errorf("cpuFeature = %q, want SVE", got)
	}
}

func bitsOf(rows []arngRow) []int {
	var b []int
	for _, r := range rows {
		b = append(b, r.bits)
	}
	return b
}

func TestArrangements(t *testing.T) {
	if got := bitsOf(parse(t, addUnpred).resolveArrangementTable("t")); !reflect.DeepEqual(got, []int{8, 16, 32, 64}) {
		t.Errorf("ADD <T> domain = %v, want [8 16 32 64]", got)
	}
	if got := bitsOf(parse(t, faddUnpred).resolveArrangementTable("t")); !reflect.DeepEqual(got, []int{16, 32, 64}) {
		t.Errorf("FADD <T> domain = %v, want [16 32 64]", got)
	}
}

func TestOperands(t *testing.T) {
	ops := parse(t, addUnpred).operands()
	var got []string
	for _, op := range ops {
		got = append(got, op.Type.String()+":"+op.role)
	}
	want := []string{"ZReg:destination", "ZReg:op0", "ZReg:op1"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("operands = %v, want %v", got, want)
	}
}

func TestEmitAllUnpredicated(t *testing.T) {
	// ADD: int|uint × {8,16,32,64} = 8 defs.
	defs := parse(t, addUnpred).emitAll()
	if len(defs) != 8 {
		t.Fatalf("ADD emitAll = %d defs, want 8", len(defs))
	}
	s := defs[0].String()
	for _, want := range []string{"ZADD", "arm64", "SVE", "elemBits"} {
		if !strings.Contains(s, want) {
			t.Errorf("emitted def missing %q:\n%s", want, s)
		}
	}

	// FADD: float × {16,32,64} = 3 defs.
	if got := len(parse(t, faddUnpred).emitAll()); got != 3 {
		t.Errorf("FADD emitAll = %d defs, want 3", got)
	}
}

func TestOperandsPredicated(t *testing.T) {
	// ADD <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>: the first <Zdn> is the
	// destination, the governing predicate becomes a mask, and the repeated
	// <Zdn> is the in-place source input.
	ops := parse(t, addPred).operands()
	var got []string
	for _, op := range ops {
		got = append(got, op.Class+":"+opRole(op))
	}
	want := []string{"vreg:destination", "mask:governing", "vreg:op0", "vreg:op1"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("predicated operands = %v, want %v", got, want)
	}
	if !ops[0].resultInArg0() {
		t.Errorf("expected <Zdn> destination to be result-in-arg0")
	}
}

// maskPredications returns the predication qualifier of every mask operand in
// the def's inputs.
func maskPredications(t *testing.T, d *unify.Value) []string {
	t.Helper()
	var op struct {
		In []struct {
			Class       string
			Predication *string
		} `unify:"in"`
	}
	if err := d.Decode(&op); err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, in := range op.In {
		if in.Class == "mask" && in.Predication != nil {
			got = append(got, *in.Predication)
		}
	}
	return got
}

func TestEmitAllPredicated(t *testing.T) {
	// The governing-predicate ADD enumerates the same int|uint × {8,16,32,64}
	// arrangements, and each def carries the predicate as a mandatory merging
	// (/M) mask *input* (not an inVariant).
	defs := parse(t, addPred).emitAll()
	if len(defs) != 8 {
		t.Fatalf("predicated ADD emitAll = %d defs, want 8", len(defs))
	}
	for _, d := range defs {
		if got := maskPredications(t, d); !reflect.DeepEqual(got, []string{"M"}) {
			t.Errorf("want one mask input with predication M, got %v", got)
		}
	}
}

// fabsMZ mimics FABS: one iclass with two encodings, /M (merging) and /Z
// (zeroing), so both predication variants must be emitted.
const fabsMZ = `<instructionsection id="fabs_z_p_z" title="FABS -- A64" type="instruction">
  <desc><brief><para>Floating-point absolute value (predicated)</para></brief></desc>
  <classes><iclass>
    <docvars><docvar key="instr-class" value="sve"/><docvar key="mnemonic" value="FABS"/></docvars>
    <encoding name="fabs_z_p_z_m">
      <asmtemplate><text>FABS  </text><a link="zd">&lt;Zd&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="pg">&lt;Pg&gt;</a><text>/M, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
    </encoding>
    <encoding name="fabs_z_p_z_z">
      <asmtemplate><text>FABS  </text><a link="zd">&lt;Zd&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="pg">&lt;Pg&gt;</a><text>/Z, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
    </encoding>
  </iclass></classes>
  <explanations><explanation>
    <symbol link="t">&lt;T&gt;</symbol>
    <definition><table><tgroup><tbody>
      <row><entry class="bitfield">01</entry><entry class="symbol">H</entry></row>
      <row><entry class="bitfield">10</entry><entry class="symbol">S</entry></row>
      <row><entry class="bitfield">11</entry><entry class="symbol">D</entry></row>
    </tbody></tgroup></table></definition>
  </explanation></explanations>
</instructionsection>`

func TestPredicationMergingAndZeroing(t *testing.T) {
	// FABS has two encodings in one iclass (/M and /Z); both forms are emitted.
	// float × {16,32,64} × {M,Z} = 6 defs.
	defs := parse(t, fabsMZ).emitAll()
	if len(defs) != 6 {
		t.Fatalf("FABS emitAll = %d defs, want 6", len(defs))
	}
	seen := map[string]int{}
	for _, d := range defs {
		for _, p := range maskPredications(t, d) {
			seen[p]++
		}
	}
	if seen["M"] != 3 || seen["Z"] != 3 {
		t.Errorf("want 3 M and 3 Z variants, got %v", seen)
	}
}

// movprfxZM mimics MOVPRFX <Zd>.<T>, <Pg>/<ZM>, <Zn>.<T>: a single encoding whose
// predication bit selects merging or zeroing, so both are emitted.
const movprfxZM = `<instructionsection id="movprfx_z_p_z" title="MOVPRFX -- A64" type="instruction">
  <desc><brief><para>Move prefix (predicated)</para></brief></desc>
  <classes><iclass>
    <docvars><docvar key="instr-class" value="sve"/><docvar key="mnemonic" value="MOVPRFX"/></docvars>
    <encoding name="movprfx_z_p_z_">
      <asmtemplate><text>MOVPRFX  </text><a link="zd">&lt;Zd&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="pg">&lt;Pg&gt;</a><text>/</text><a link="zm">&lt;ZM&gt;</a><text>, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
    </encoding>
  </iclass></classes>
  <explanations><explanation>
    <symbol link="t">&lt;T&gt;</symbol>
    <definition><table><tgroup><tbody>
      <row><entry class="bitfield">00</entry><entry class="symbol">B</entry></row>
      <row><entry class="bitfield">01</entry><entry class="symbol">H</entry></row>
      <row><entry class="bitfield">10</entry><entry class="symbol">S</entry></row>
      <row><entry class="bitfield">11</entry><entry class="symbol">D</entry></row>
    </tbody></tgroup></table></definition>
  </explanation></explanations>
</instructionsection>`

func TestPredicationZM(t *testing.T) {
	// MOVPRFX is agnostic (a move), so: int|uint × {B,H,S,D} × {M,Z} = 16 defs.
	defs := parse(t, movprfxZM).emitAll()
	if len(defs) != 16 {
		t.Fatalf("MOVPRFX emitAll = %d defs, want 16", len(defs))
	}
	seen := map[string]int{}
	for _, d := range defs {
		for _, p := range maskPredications(t, d) {
			seen[p]++
		}
	}
	if seen["M"] != 8 || seen["Z"] != 8 {
		t.Errorf("want 8 M and 8 Z variants, got %v", seen)
	}
}

// sunpkhi mimics SUNPKHI <Zd>.<T>, <Zn>.<Tb>: the destination is one element
// size wider than the source, and both select on the same "size" field. The two
// symbols use bitfield-keyed tables so they line up by size.
const sunpkhi = `<instructionsection id="sunpkhi_z_z" title="SUNPKHI -- A64" type="instruction">
  <docvars>
    <docvar key="instr-class" value="sve"/>
    <docvar key="mnemonic" value="SUNPKHI"/>
  </docvars>
  <classes><iclass><encoding name="sunpkhi_z_z">
    <asmtemplate><text>SUNPKHI  </text><a link="zd">&lt;Zd&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="tb">&lt;Tb&gt;</a></asmtemplate>
  </encoding></iclass></classes>
  <explanations>
    <explanation><symbol link="t">&lt;T&gt;</symbol><definition><table><tgroup><tbody>
      <row><entry class="bitfield">01</entry><entry class="symbol">H</entry></row>
      <row><entry class="bitfield">10</entry><entry class="symbol">S</entry></row>
      <row><entry class="bitfield">11</entry><entry class="symbol">D</entry></row>
    </tbody></tgroup></table></definition></explanation>
    <explanation><symbol link="tb">&lt;Tb&gt;</symbol><definition><table><tgroup><tbody>
      <row><entry class="bitfield">01</entry><entry class="symbol">B</entry></row>
      <row><entry class="bitfield">10</entry><entry class="symbol">H</entry></row>
      <row><entry class="bitfield">11</entry><entry class="symbol">S</entry></row>
    </tbody></tgroup></table></definition></explanation>
  </explanations>
</instructionsection>`

func TestNonUniformArrangement(t *testing.T) {
	// int|uint × {H/B, S/H, D/S} = 6 defs, each with the destination one size
	// wider than the source.
	inst := parse(t, sunpkhi)
	defs := inst.emitAll()
	if len(defs) != 6 {
		t.Fatalf("SUNPKHI emitAll = %d defs, want 6", len(defs))
	}
	// In every def the (single) out elemBits must be double the (single) in.
	sawWiden := false
	for _, d := range defs {
		var op struct {
			In  []struct{ ElemBits int } `unify:"in"`
			Out []struct{ ElemBits int } `unify:"out"`
		}
		if err := d.Decode(&op); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(op.In) != 1 || len(op.Out) != 1 {
			t.Fatalf("want 1 in + 1 out, got in=%d out=%d", len(op.In), len(op.Out))
		}
		if op.Out[0].ElemBits != 2*op.In[0].ElemBits {
			t.Errorf("out elemBits %d, want 2×in elemBits %d", op.Out[0].ElemBits, op.In[0].ElemBits)
		}
		if op.Out[0].ElemBits == 16 && op.In[0].ElemBits == 8 {
			sawWiden = true
		}
	}
	if !sawWiden {
		t.Errorf("expected an H<-B widening variant")
	}
}

// saddv mimics SADDV <Dd>, <Pg>, <Zn>.<T>: a horizontal reduction whose scalar
// result <Dd> is a (special, opaque) destination, not an input.
const saddv = `<instructionsection id="saddv_r_p_z" title="SADDV -- A64" type="instruction">
  <desc><brief><para>Signed add reduction to scalar</para></brief></desc>
  <classes><iclass>
    <docvars><docvar key="instr-class" value="sve"/><docvar key="mnemonic" value="SADDV"/></docvars>
    <encoding name="saddv_r_p_z_">
      <asmtemplate><text>SADDV  </text><a link="dd">&lt;Dd&gt;</a><text>, </text><a link="pg">&lt;Pg&gt;</a><text>, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
    </encoding>
  </iclass></classes>
  <explanations><explanation>
    <symbol link="t">&lt;T&gt;</symbol>
    <definition><table><tgroup><tbody>
      <row><entry class="bitfield">00</entry><entry class="symbol">B</entry></row>
      <row><entry class="bitfield">01</entry><entry class="symbol">H</entry></row>
      <row><entry class="bitfield">10</entry><entry class="symbol">S</entry></row>
    </tbody></tgroup></table></definition>
  </explanation></explanations>
</instructionsection>`

func TestReductionOutput(t *testing.T) {
	ops := parse(t, saddv).operands()
	var got []string
	for _, op := range ops {
		got = append(got, op.Class+":"+opRole(op))
	}
	// The scalar result <Dd> is a SIMD&FP register destination, not an input.
	want := []string{"vreg:destination", "mask:governing", "vreg:op0"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("SADDV operands = %v, want %v", got, want)
	}
	// <Dd> is a fixed 64-bit SIMD&FP scalar output, not the scalable source.
	for _, d := range parse(t, saddv).emitAll() {
		var op struct {
			Out []struct {
				Class string
				Bits  string
				Lanes string
			} `unify:"out"`
		}
		if err := d.Decode(&op); err != nil {
			t.Fatal(err)
		}
		if len(op.Out) != 1 || op.Out[0].Class != "vreg" || op.Out[0].Bits != "64" || op.Out[0].Lanes != "1" {
			t.Errorf("SADDV out = %+v, want one vreg bits=64 lanes=1", op.Out)
		}
	}
}

// st1b mimics ST1B { <Zt>.<T> }, <Pg>, [<Xn|SP>{, #<imm>, MUL VL}]: a store
// whose single-register list is the data source and whose memory operand (last)
// is the destination.
const st1b = `<instructionsection id="st1b_z_p_bi" title="ST1B -- A64" type="instruction">
  <desc><brief><para>Contiguous store bytes from vector (immediate index)</para></brief></desc>
  <classes><iclass>
    <docvars><docvar key="instr-class" value="sve"/><docvar key="mnemonic" value="ST1B"/></docvars>
    <encoding name="st1b_z_p_bi_">
      <asmtemplate><text>ST1B  { </text><a link="zt">&lt;Zt&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text> }, </text><a link="pg">&lt;Pg&gt;</a><text>, [</text><a link="xn">&lt;Xn|SP&gt;</a><text>{, #</text><a link="imm">&lt;imm&gt;</a><text>, MUL VL}]</text></asmtemplate>
    </encoding>
  </iclass></classes>
  <explanations><explanation>
    <symbol link="t">&lt;T&gt;</symbol>
    <definition><table><tgroup><tbody>
      <row><entry class="bitfield">00</entry><entry class="symbol">B</entry></row>
      <row><entry class="bitfield">01</entry><entry class="symbol">H</entry></row>
      <row><entry class="bitfield">10</entry><entry class="symbol">S</entry></row>
      <row><entry class="bitfield">11</entry><entry class="symbol">D</entry></row>
    </tbody></tgroup></table></definition>
  </explanation></explanations>
</instructionsection>`

func TestStoreReglist(t *testing.T) {
	ops := parse(t, st1b).operands()
	var got []string
	for _, op := range ops {
		got = append(got, op.Class+":"+opRole(op))
	}
	// The single-register list unwraps to a vreg (the data source); the memory
	// operand is the store destination. Order follows the source template. The
	// predicate is <Pg>: a governing predicate (role "mask"), even though a
	// store writes no /Z or /M qualifier.
	want := []string{"vreg:op0", "mask:governing", "mem:destination"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ST1B operands = %v, want %v", got, want)
	}
	for _, d := range parse(t, st1b).emitAll() {
		var op struct {
			In []struct {
				Class      string
				ListNumber *string
			} `unify:"in"`
			Out []struct{ Class string } `unify:"out"`
		}
		if err := d.Decode(&op); err != nil {
			t.Fatal(err)
		}
		if len(op.Out) != 1 || op.Out[0].Class != "mem" {
			t.Errorf("ST1B out = %+v, want one mem", op.Out)
		}
		// The data vreg is a register list (distinct assembler encoding); the
		// mask is not.
		for _, in := range op.In {
			isList := in.ListNumber != nil
			if want := in.Class == "vreg"; isList != want {
				t.Errorf("in %q listNumber present = %v, want %v", in.Class, isList, want)
			}
		}
	}
}

func TestMemoryOperandClassified(t *testing.T) {
	// A bracketed operand is classified as a single "mem" class (not mistaken
	// for a scalable vector); no addressing-mode sub-classification.
	ops := operands("LD1B  <Zt>.<T>, <Pg>/Z, [<Xn|SP>, #<imm>, MUL VL]")
	if !hasClass(ops, "mem") {
		t.Fatalf("expected a mem operand, got %v", ops)
	}
	for _, op := range ops {
		if op.Class == "vreg" && strings.Contains(op.regName, "Xn") {
			t.Errorf("memory address misclassified as vreg: %+v", op)
		}
	}
}

// opRole renders an operand's partition for test expectations: its role, or
// "governing" for the governing predicate, which has no numbered role.
func opRole(op Operand) string {
	if op.governing {
		return "governing"
	}
	return op.role
}

// absExplanations is the explanation block for the ABS fixtures: the size table
// plus a <Pg> explanation using the spec's "governing scalable predicate
// register" wording, so parsing exercises the explanation-driven governing
// classification rather than the syntactic fallback.
const absExplanations = `
  <explanations>
    <explanation>
      <symbol link="t">&lt;T&gt;</symbol>
      <definition>
        <table><tgroup><tbody>
          <row><entry class="symbol">B</entry></row>
          <row><entry class="symbol">H</entry></row>
          <row><entry class="symbol">S</entry></row>
          <row><entry class="symbol">D</entry></row>
        </tbody></tgroup></table>
      </definition>
    </explanation>
    <explanation>
      <symbol link="pg">&lt;Pg&gt;</symbol>
      <account encodedin="Pg"><intro><para>Is the name of the governing scalable predicate register, encoded in the "Pg" field.</para></intro></account>
    </explanation>
  </explanations>`

// absSection builds one encoding of ABS, a predicated-only operation: both
// encodings are titled plain "ABS" (nothing in the title says "predicated"),
// and they differ only in the governing predicate's qualifier.
func absSection(id, qual string) string {
	return `<instructionsection id="` + id + `" title="ABS" type="instruction">
  <docvars>
    <docvar key="instr-class" value="sve"/>
    <docvar key="mnemonic" value="ABS"/>
  </docvars>
  <desc><brief><para>Absolute value of the signed integer in each active element.</para></brief></desc>
  <classes><iclass><encoding name="` + id + `">
    <asmtemplate><text>ABS  </text><a link="zd">&lt;Zd&gt;</a><text>.</text><a link="t">&lt;T&gt;</a><text>, </text><a link="pg">&lt;Pg&gt;</a><text>/` + qual + `, </text><a link="zn">&lt;Zn&gt;</a><text>.</text><a link="t">&lt;T&gt;</a></asmtemplate>
  </encoding></iclass></classes>` + absExplanations + `</instructionsection>`
}

// TestGroupPredicatedOnly covers the emission path of an operation with no
// unpredicated encoding at all: the merging encoding carries the operation,
// keeping its governing predicate as an implicit-all-true input, and the group
// becomes an inVariant on it; the sibling encoding is covered and not emitted.
func TestGroupPredicatedOnly(t *testing.T) {
	m := parse(t, absSection("abs_m", "M"))
	z := parse(t, absSection("abs_z", "Z"))
	covered := groupPredicationForms([]*Instruction{m, z})
	if covered[m] || !covered[z] {
		t.Fatalf("covered[m]=%v covered[z]=%v, want the merging carrier kept and the zeroing sibling covered", covered[m], covered[z])
	}
	if len(m.predVariants) != 1 || m.predVariants[0].quals != "M" {
		t.Fatalf("carrier predVariants = %+v, want one variant with quals M", m.predVariants)
	}

	defs := m.emitAll()
	if len(defs) == 0 {
		t.Fatal("carrier emitAll returned no defs")
	}
	for _, d := range defs {
		var op struct {
			In []struct {
				Class       string
				Predication *string
				Governing   *bool
				RegName     *string   `unify:"regName"`
				PredRegName *[]string `unify:"predRegName"`
			} `unify:"in"`
			InVariant []struct {
				Class       string
				Predication *string
			} `unify:"inVariant"`
			Out []struct {
				PredRegName *[]string `unify:"predRegName"`
			} `unify:"out"`
		}
		if err := d.Decode(&op); err != nil {
			t.Fatal(err)
		}
		var sawGoverning, sawVreg bool
		for _, in := range op.In {
			switch in.Class {
			case "mask":
				if in.Governing == nil || !*in.Governing {
					t.Errorf("mask input not marked governing: %+v", in)
				}
				if in.Predication == nil || *in.Predication != "M" {
					t.Errorf("governing predicate predication = %v, want M", in.Predication)
				}
				sawGoverning = true
			case "vreg":
				if in.PredRegName == nil || !reflect.DeepEqual(*in.PredRegName, []string{"Zn"}) {
					t.Errorf("vreg input predRegName = %v, want [Zn]", in.PredRegName)
				}
				sawVreg = true
			}
		}
		if !sawGoverning || !sawVreg {
			t.Errorf("inputs missing governing mask or vreg: %+v", op.In)
		}
		if len(op.InVariant) != 1 || op.InVariant[0].Predication == nil || *op.InVariant[0].Predication != "M" {
			t.Errorf("inVariant = %+v, want one mask with predication M", op.InVariant)
		}
		if len(op.Out) != 1 || op.Out[0].PredRegName == nil || !reflect.DeepEqual(*op.Out[0].PredRegName, []string{"Zd"}) {
			t.Errorf("out predRegName = %+v, want [Zd]", op.Out)
		}
	}
}
