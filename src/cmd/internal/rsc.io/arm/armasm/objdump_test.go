// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armasm

import (
	"encoding/binary"
	"strings"
	"testing"
)

func TestObjdumpARMTestdata(t *testing.T) { testObjdumpARM(t, testdataCases(t)) }
func TestObjdumpARMManual(t *testing.T)   { testObjdumpARM(t, hexCases(t, objdumpManualTests)) }
func TestObjdumpARMCond(t *testing.T)     { testObjdumpARM(t, condCases(t)) }
func TestObjdumpARMUncond(t *testing.T)   { testObjdumpARM(t, uncondCases(t)) }
func TestObjdumpARMVFP(t *testing.T)      { testObjdumpARM(t, vfpCases(t)) }

// objdumpManualTests holds test cases that will be run by TestObjdumpARMManual.
// If you are debugging a few cases that turned up in a longer run, it can be useful
// to list them here and then use -run=Manual, particularly with tracing enabled.
// Note that these are byte sequences, so they must be reversed from the usual
// word presentation.
var objdumpManualTests = `
00000000
`

// allowedMismatchObjdump reports whether the mismatch between text and dec
// should be allowed by the test.
func allowedMismatchObjdump(text string, size int, inst *Inst, dec ExtInst) bool {
	if hasPrefix(text, "error:") {
		if hasPrefix(dec.text, unsupported...) || strings.Contains(dec.text, "invalid:") || strings.HasSuffix(dec.text, "^") || strings.Contains(dec.text, "f16.f64") || strings.Contains(dec.text, "f64.f16") {
			return true
		}
		// word 4320F02C: libopcodes says 'nopmi {44}'.
		if hasPrefix(dec.text, "nop") && strings.Contains(dec.text, "{") {
			return true
		}
	}

	if hasPrefix(dec.text, "error:") && text == "undef" && inst.Enc == 0xf7fabcfd {
		return true
	}

	// word 00f02053: libopcodes says 'noppl {0}'.
	if hasPrefix(dec.text, "nop") && hasPrefix(text, "nop") && dec.text == text+" {0}" {
		return true
	}

	// word F57FF04F. we say 'dsb #15', libopcodes says 'dsb sy'.
	if hasPrefix(text, "dsb") && hasPrefix(dec.text, "dsb") {
		return true
	}
	// word F57FF06F. we say 'isb #15', libopcodes says 'isb sy'.
	if hasPrefix(text, "isb") && hasPrefix(dec.text, "isb") {
		return true
	}
	// word F57FF053. we say 'dmb #3', libopcodes says 'dmb osh'.
	if hasPrefix(text, "dmb") && hasPrefix(dec.text, "dmb") {
		return true
	}

	// word 992D0000. push/stmdb with no registers (undefined).
	// we say 'stmdbls sp!, {}', libopcodes says 'pushls {}'.
	if hasPrefix(text, "stmdb") && hasPrefix(dec.text, "push") && strings.Contains(text, "{}") && strings.Contains(dec.text, "{}") {
		return true
	}

	// word 28BD0000. pop/ldm with no registers (undefined).
	// we say 'ldmcs sp!, {}', libopcodes says 'popcs {}'.
	if hasPrefix(text, "ldm") && hasPrefix(dec.text, "pop") && strings.Contains(text, "{}") && strings.Contains(dec.text, "{}") {
		return true
	}

	// word 014640F0.
	// libopcodes emits #-0 for negative zero; we don't.
	if strings.Replace(dec.text, "#-0", "#0", -1) == text || strings.Replace(dec.text, ", #-0", "", -1) == text {
		return true
	}

	// word 91EF90F0. we say 'strdls r9, [pc, #0]!' but libopcodes says 'strdls r9, [pc]'.
	// word D16F60F0. we say 'strdle r6, [pc, #0]!' but libopcodes says 'strdle r6, [pc, #-0]'.
	if strings.Replace(text, ", #0]!", "]", -1) == strings.Replace(dec.text, ", #-0]", "]", -1) {
		return true
	}

	// word 510F4000. we say apsr, libopcodes says CPSR.
	if strings.Replace(dec.text, "CPSR", "apsr", -1) == text {
		return true
	}

	// word 06A4B059.
	// for ssat and usat, libopcodes decodes asr #0 as asr #0 but the manual seems to say it should be asr #32.
	// There is never an asr #0.
	if strings.Replace(dec.text, ", asr #0", ", asr #32", -1) == text {
		return true
	}

	if len(dec.enc) >= 4 {
		raw := binary.LittleEndian.Uint32(dec.enc[:4])

		// word 21FFF0B5.
		// the manual is clear that this is pre-indexed mode (with !) but libopcodes generates post-index (without !).
		if raw&0x01200000 == 0x01200000 && strings.Replace(text, "!", "", -1) == dec.text {
			return true
		}

		// word C100543E: libopcodes says tst, but no evidence for that.
		if strings.HasPrefix(dec.text, "tst") && raw&0x0ff00000 != 0x03100000 && raw&0x0ff00000 != 0x01100000 {
			return true
		}

		// word C3203CE8: libopcodes says teq, but no evidence for that.
		if strings.HasPrefix(dec.text, "teq") && raw&0x0ff00000 != 0x03300000 && raw&0x0ff00000 != 0x01300000 {
			return true
		}

		// word D14C552E: libopcodes says cmp but no evidence for that.
		if strings.HasPrefix(dec.text, "cmp") && raw&0x0ff00000 != 0x03500000 && raw&0x0ff00000 != 0x01500000 {
			return true
		}

		// word 2166AA4A: libopcodes says cmn but no evidence for that.
		if strings.HasPrefix(dec.text, "cmn") && raw&0x0ff00000 != 0x03700000 && raw&0x0ff00000 != 0x01700000 {
			return true
		}

		// word E70AEEEF: libopcodes says str but no evidence for that.
		if strings.HasPrefix(dec.text, "str") && len(dec.text) >= 5 && (dec.text[3] == ' ' || dec.text[5] == ' ') && raw&0x0e500018 != 0x06000000 && raw&0x0e500000 != 0x0400000 {
			return true
		}

		// word B0AF48F4: libopcodes says strd but P=0,W=1 which is unpredictable.
		if hasPrefix(dec.text, "ldr", "str") && raw&0x01200000 == 0x00200000 {
			return true
		}

		// word B6CC1C76: libopcodes inexplicably says 'uxtab16lt r1, ip, r6, ROR #24' instead of 'uxtab16lt r1, ip, r6, ror #24'
		if strings.ToLower(dec.text) == text {
			return true
		}

		// word F410FDA1: libopcodes says PLDW but the manual is clear that PLDW is F5/F7, not F4.
		// word F7D0FB17: libopcodes says PLDW but the manual is clear that PLDW has 0x10 clear
		if hasPrefix(dec.text, "pld") && raw&0xfd000010 != 0xf5000000 {
			return true
		}

		// word F650FE14: libopcodes says PLI but the manual is clear that PLI has 0x10 clear
		if hasPrefix(dec.text, "pli") && raw&0xff000010 != 0xf6000000 {
			return true
		}
	}

	return false
}

// Instructions known to libopcodes (or xed) but not to us.
// Most of these are floating point coprocessor instructions.
var unsupported = strings.Fields(`
	abs
	acs
	adf
	aes
	asn
	atn
	cdp
	cf
	cmf
	cnf
	cos
	cps
	crc32
	dvf
	eret
	exp
	fadd
	fcmp
	fcpy
	fcvt
	fdiv
	fdv
	fix
	fld
	flt
	fmac
	fmd
	fml
	fmr
	fms
	fmul
	fmx
	fneg
	fnm
	frd
	fsit
	fsq
	fst
	fsu
	fto
	fui
	hlt
	hvc
	lda
	ldc
	ldf
	lfm
	lgn
	log
	mar
	mcr
	mcrr
	mia
	mnf
	mra
	mrc
	mrrc
	mrs
	msr
	msr
	muf
	mvf
	nrm
	pol
	pow
	rdf
	rfc
	rfe
	rfs
	rmf
	rnd
	rpw
	rsf
	sdiv
	sev
	sfm
	sha1
	sha256
	sin
	smc
	sqt
	srs
	stc
	stf
	stl
	suf
	tan
	udf
	udiv
	urd
	vfma
	vfms
	vfnma
	vfnms
	vrint
	wfc
	wfs
`)
