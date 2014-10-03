// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"bytes"
	"strings"
	"testing"
)

func TestXed32Manual(t *testing.T)   { testXed32(t, hexCases(t, xedManualTests)) }
func TestXed32Testdata(t *testing.T) { testXed32(t, concat(basicPrefixes, testdataCases(t))) }
func TestXed32ModRM(t *testing.T)    { testXed32(t, concat(basicPrefixes, enumModRM)) }
func TestXed32OneByte(t *testing.T)  { testBasic(t, testXed32) }
func TestXed320F(t *testing.T)       { testBasic(t, testXed32, 0x0F) }
func TestXed320F38(t *testing.T)     { testBasic(t, testXed32, 0x0F, 0x38) }
func TestXed320F3A(t *testing.T)     { testBasic(t, testXed32, 0x0F, 0x3A) }
func TestXed32Prefix(t *testing.T)   { testPrefix(t, testXed32) }

func TestXed64Manual(t *testing.T)   { testXed64(t, hexCases(t, xedManualTests)) }
func TestXed64Testdata(t *testing.T) { testXed64(t, concat(basicPrefixes, testdataCases(t))) }
func TestXed64ModRM(t *testing.T)    { testXed64(t, concat(basicPrefixes, enumModRM)) }
func TestXed64OneByte(t *testing.T)  { testBasic(t, testXed64) }
func TestXed640F(t *testing.T)       { testBasic(t, testXed64, 0x0F) }
func TestXed640F38(t *testing.T)     { testBasic(t, testXed64, 0x0F, 0x38) }
func TestXed640F3A(t *testing.T)     { testBasic(t, testXed64, 0x0F, 0x3A) }
func TestXed64Prefix(t *testing.T)   { testPrefix(t, testXed64) }

func TestXed64REXTestdata(t *testing.T) {
	testXed64(t, filter(concat3(basicPrefixes, rexPrefixes, testdataCases(t)), isValidREX))
}
func TestXed64REXModRM(t *testing.T)   { testXed64(t, concat3(basicPrefixes, rexPrefixes, enumModRM)) }
func TestXed64REXOneByte(t *testing.T) { testBasicREX(t, testXed64) }
func TestXed64REX0F(t *testing.T)      { testBasicREX(t, testXed64, 0x0F) }
func TestXed64REX0F38(t *testing.T)    { testBasicREX(t, testXed64, 0x0F, 0x38) }
func TestXed64REX0F3A(t *testing.T)    { testBasicREX(t, testXed64, 0x0F, 0x3A) }
func TestXed64REXPrefix(t *testing.T)  { testPrefixREX(t, testXed64) }

// xedManualTests holds test cases that will be run by TestXedManual32 and TestXedManual64.
// If you are debugging a few cases that turned up in a longer run, it can be useful
// to list them here and then use -run=XedManual, particularly with tracing enabled.
var xedManualTests = `
6690
`

// allowedMismatchXed reports whether the mismatch between text and dec
// should be allowed by the test.
func allowedMismatchXed(text string, size int, inst *Inst, dec ExtInst) bool {
	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, "GENERAL_ERROR", "INSTR_TOO_LONG", "BAD_LOCK_PREFIX") {
		return true
	}

	if contains(dec.text, "BAD_LOCK_PREFIX") && countExactPrefix(inst, PrefixLOCK|PrefixInvalid) > 0 {
		return true
	}

	if contains(dec.text, "BAD_LOCK_PREFIX", "GENERAL_ERROR") && countExactPrefix(inst, PrefixLOCK|PrefixImplicit) > 0 {
		return true
	}

	if text == "lock" && size == 1 && contains(dec.text, "BAD_LOCK_PREFIX") {
		return true
	}

	// Instructions not known to us.
	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, unsupported...) {
		return true
	}

	// Instructions not known to xed.
	if contains(text, xedUnsupported...) && contains(dec.text, "ERROR") {
		return true
	}

	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, "shl ") && (inst.Opcode>>16)&0xEC38 == 0xC030 {
		return true
	}

	// 82 11 22: xed says 'adc byte ptr [ecx], 0x22' but there is no justification in the manuals for that.
	// C0 30 11: xed says 'shl byte ptr [eax], 0x11' but there is no justification in the manuals for that.
	// F6 08 11: xed says 'test byte ptr [eax], 0x11' but there is no justification in the manuals for that.
	if (contains(text, "error:") || isPrefix(text) && size == 1) && hasByte(dec.enc[:dec.nenc], 0x82, 0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3, 0xF6, 0xF7) {
		return true
	}

	// F3 11 22 and many others: xed allows and drops misused rep/repn prefix.
	if (text == "rep" && dec.enc[0] == 0xF3 || (text == "repn" || text == "repne") && dec.enc[0] == 0xF2) && (!contains(dec.text, "ins", "outs", "movs", "lods", "cmps", "scas") || contains(dec.text, "xmm")) {
		return true
	}

	// 0F C7 30: xed says vmptrld qword ptr [eax]; we say rdrand eax.
	// TODO(rsc): Fix, since we are probably wrong, but we don't have vmptrld in the manual.
	if contains(text, "rdrand") && contains(dec.text, "vmptrld", "vmxon", "vmclear") {
		return true
	}

	// F3 0F AE 00: we say 'rdfsbase dword ptr [eax]' but RDFSBASE needs a register.
	// Also, this is a 64-bit only instruction.
	// TODO(rsc): Fix to reject this encoding.
	if contains(text, "rdfsbase", "rdgsbase", "wrfsbase", "wrgsbase") && contains(dec.text, "ERROR") {
		return true
	}

	// 0F 01 F8: we say swapgs but that's only valid in 64-bit mode.
	// TODO(rsc): Fix.
	if contains(text, "swapgs") {
		return true
	}

	// 0F 24 11: 'mov ecx, tr2' except there is no TR2.
	// Or maybe the MOV to TR registers doesn't use RMF.
	if contains(text, "cr1", "cr5", "cr6", "cr7", "tr0", "tr1", "tr2", "tr3", "tr4", "tr5", "tr6", "tr7") && contains(dec.text, "ERROR") {
		return true
	}

	// 0F 19 11, 0F 1C 11, 0F 1D 11, 0F 1E 11, 0F 1F 11: xed says nop,
	// but the Intel manuals say that the only NOP there is 0F 1F /0.
	// Perhaps xed is reporting an older encoding.
	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, "nop ") && (inst.Opcode>>8)&0xFFFF38 != 0x0F1F00 {
		return true
	}

	// 66 0F AE 38: clflushopt but we only know clflush
	if contains(text, "clflush") && contains(dec.text, "clflushopt") {
		return true
	}

	// 0F 20 04 11: MOV SP, CR0 but has mod!=3 despite register argument.
	// (This encoding ignores the mod bits.) The decoder sees the non-register
	// mod and reads farther ahead to decode the memory reference that
	// isn't really there, causing the size to be too large.
	// TODO(rsc): Fix.
	if text == dec.text && size > dec.nenc && contains(text, " cr", " dr", " tr") {
		return true
	}

	// 0F AE E9: xed says lfence, which is wrong (only 0F AE E8 is lfence). And so on.
	if contains(dec.text, "fence") && hasByte(dec.enc[:dec.nenc], 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF) {
		return true
	}

	// DD C9, DF C9: xed says 'fxch st0, st1' but that instruction is D9 C9.
	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, "fxch ") && hasByte(dec.enc[:dec.nenc], 0xDD, 0xDF) {
		return true
	}

	// DC D4: xed says 'fcom st0, st4' but that instruction is D8 D4.
	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, "fcom ") && hasByte(dec.enc[:dec.nenc], 0xD8, 0xDC) {
		return true
	}

	// DE D4: xed says 'fcomp st0, st4' but that instruction is D8 D4.
	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, "fcomp ") && hasByte(dec.enc[:dec.nenc], 0xDC, 0xDE) {
		return true
	}

	// DF D4: xed says 'fstp st4, st0' but that instruction is DD D4.
	if (contains(text, "error:") || isPrefix(text) && size == 1) && contains(dec.text, "fstp ") && hasByte(dec.enc[:dec.nenc], 0xDF) {
		return true
	}

	return false
}

func countExactPrefix(inst *Inst, target Prefix) int {
	n := 0
	for _, p := range inst.Prefix {
		if p == target {
			n++
		}
	}
	return n
}

func hasByte(src []byte, target ...byte) bool {
	for _, b := range target {
		if bytes.IndexByte(src, b) >= 0 {
			return true
		}
	}
	return false
}

// Instructions known to us but not to xed.
var xedUnsupported = strings.Fields(`
	xrstor
	xsave
	xsave
	ud1
	xgetbv
	xsetbv
	fxsave
	fxrstor
	clflush
	lfence
	mfence
	sfence
	rsqrtps
	rcpps
	emms
	ldmxcsr
	stmxcsr
	movhpd
	movnti
	rdrand
	movbe
	movlpd
	sysret
`)
