// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64asm

import (
	"encoding/binary"
	"strings"
	"testing"
)

func TestObjdumpPowerTestdata(t *testing.T) { testObjdump(t, testdataCases(t)) }
func TestObjdumpPowerManual(t *testing.T)   { testObjdump(t, hexCases(t, objdumpManualTests)) }

// Disable this for now since generating all possible bit combinations within a word
// generates lots of ppc64x instructions not possible with golang so not worth supporting..
//func TestObjdumpPowerRandom(t *testing.T)   { testObjdump(t, randomCases(t)) }

// objdumpManualTests holds test cases that will be run by TestObjdumpARMManual.
// If you are debugging a few cases that turned up in a longer run, it can be useful
// to list them here and then use -run=Manual, particularly with tracing enabled.
// Note that these are byte sequences, so they must be reversed from the usual
// word presentation.
var objdumpManualTests = `
6d746162
4c040000
88000017
`

// allowedMismatchObjdump reports whether the mismatch between text and dec
// should be allowed by the test.
func allowedMismatchObjdump(text string, size int, inst *Inst, dec ExtInst) bool {
	if hasPrefix(dec.text, deleted...) {
		return true
	}

	// we support more instructions than binutils
	if strings.Contains(dec.text, ".long") {
		return true
	}

	if hasPrefix(text, "error:") {
		if hasPrefix(dec.text, unsupported...) {
			return true
		}
	}

	switch inst.Op {
	case BC, BCA, BL, BLA, BCL, BCLA, TDI, TWI, TW, TD:
		return true // TODO(minux): we lack the support for extended opcodes here
	case RLWNM, RLWNMCC, RLDICL, RLDICLCC, RLWINM, RLWINMCC, RLDCL, RLDCLCC:
		return true // TODO(minux): we lack the support for extended opcodes here
	case DCBTST, DCBT:
		return true // objdump uses the embedded argument order, we use the server argument order
	case MTFSF, MTFSFCC: // objdump doesn't show the last two arguments
		return true
	case VSPLTB, VSPLTH, VSPLTW: // objdump generates unreasonable result "vspltw v6,v19,4" for 10c49a8c, the last 4 should be 0.
		return true
	}
	if hasPrefix(text, "evm", "evl", "efs") { // objdump will disassemble them wrong (e.g. evmhoumia as vsldoi)
		return true
	}

	if len(dec.enc) >= 4 {
		_ = binary.BigEndian.Uint32(dec.enc[:4])
	}

	return false
}

// Instructions known to libopcodes (or xed) but not to us.
// TODO(minux): those single precision instructions are missing from ppc64.csv
// those data cache instructions are deprecated, but must be treated as no-ops, see 4.3.2.1 pg. 774.
var unsupported = strings.Fields(`
fmsubs
fmsubs.
fnmadds
fnmadds.
fnmsubs
fnmsubs.
fmuls
fmuls.
fdivs
fdivs.
fadds
fadds.
fsubs
fsubs.
dst
dstst
dssall
`)

// Instructions explicitly dropped in Power ISA that were in POWER architecture.
// See A.30 Deleted Instructions and A.31 Discontiued Opcodes
var deleted = strings.Fields(`
abs
clcs
clf
cli
dclst
div
divs
doz
dozi
lscbx
maskg
maskir
mfsri
mul
nabs
rac
rfi
rfsvc
rlmi
rrib
sle
sleq
sliq
slliq
sllq
slq
sraiq
sraq
sre
srea
sreq
sriq
srliq
srlq
srq
maskg`)
