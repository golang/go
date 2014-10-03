// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"bytes"
	"strings"
	"testing"
)

func TestObjdump32Manual(t *testing.T)   { testObjdump32(t, hexCases(t, objdumpManualTests)) }
func TestObjdump32Testdata(t *testing.T) { testObjdump32(t, concat(basicPrefixes, testdataCases(t))) }
func TestObjdump32ModRM(t *testing.T)    { testObjdump32(t, concat(basicPrefixes, enumModRM)) }
func TestObjdump32OneByte(t *testing.T)  { testBasic(t, testObjdump32) }
func TestObjdump320F(t *testing.T)       { testBasic(t, testObjdump32, 0x0F) }
func TestObjdump320F38(t *testing.T)     { testBasic(t, testObjdump32, 0x0F, 0x38) }
func TestObjdump320F3A(t *testing.T)     { testBasic(t, testObjdump32, 0x0F, 0x3A) }
func TestObjdump32Prefix(t *testing.T)   { testPrefix(t, testObjdump32) }

func TestObjdump64Manual(t *testing.T)   { testObjdump64(t, hexCases(t, objdumpManualTests)) }
func TestObjdump64Testdata(t *testing.T) { testObjdump64(t, concat(basicPrefixes, testdataCases(t))) }
func TestObjdump64ModRM(t *testing.T)    { testObjdump64(t, concat(basicPrefixes, enumModRM)) }
func TestObjdump64OneByte(t *testing.T)  { testBasic(t, testObjdump64) }
func TestObjdump640F(t *testing.T)       { testBasic(t, testObjdump64, 0x0F) }
func TestObjdump640F38(t *testing.T)     { testBasic(t, testObjdump64, 0x0F, 0x38) }
func TestObjdump640F3A(t *testing.T)     { testBasic(t, testObjdump64, 0x0F, 0x3A) }
func TestObjdump64Prefix(t *testing.T)   { testPrefix(t, testObjdump64) }

func TestObjdump64REXTestdata(t *testing.T) {
	testObjdump64(t, filter(concat3(basicPrefixes, rexPrefixes, testdataCases(t)), isValidREX))
}
func TestObjdump64REXModRM(t *testing.T) {
	testObjdump64(t, concat3(basicPrefixes, rexPrefixes, enumModRM))
}
func TestObjdump64REXOneByte(t *testing.T) { testBasicREX(t, testObjdump64) }
func TestObjdump64REX0F(t *testing.T)      { testBasicREX(t, testObjdump64, 0x0F) }
func TestObjdump64REX0F38(t *testing.T)    { testBasicREX(t, testObjdump64, 0x0F, 0x38) }
func TestObjdump64REX0F3A(t *testing.T)    { testBasicREX(t, testObjdump64, 0x0F, 0x3A) }
func TestObjdump64REXPrefix(t *testing.T)  { testPrefixREX(t, testObjdump64) }

// objdumpManualTests holds test cases that will be run by TestObjdumpManual.
// If you are debugging a few cases that turned up in a longer run, it can be useful
// to list them here and then use -run=ObjdumpManual, particularly with tracing enabled.
var objdumpManualTests = `
F390
`

// allowedMismatchObjdump reports whether the mismatch between text and dec
// should be allowed by the test.
func allowedMismatchObjdump(text string, size int, inst *Inst, dec ExtInst) bool {
	if size == 15 && dec.nenc == 15 && contains(text, "truncated") && contains(dec.text, "(bad)") {
		return true
	}

	if i := strings.LastIndex(dec.text, " "); isPrefix(dec.text[i+1:]) && size == 1 && isPrefix(text) {
		return true
	}

	if size == dec.nenc && contains(dec.text, "movupd") && contains(dec.text, "data32") {
		s := strings.Replace(dec.text, "data32 ", "", -1)
		if text == s {
			return true
		}
	}

	// Simplify our invalid instruction text.
	if text == "error: unrecognized instruction" {
		text = "BAD"
	}

	// Invalid instructions for which libopcodes prints %? register.
	// FF E8 11 22 33 44:
	// Invalid instructions for which libopcodes prints "internal disassembler error".
	// Invalid instructions for which libopcodes prints 8087 only (e.g., DB E0)
	// or prints 287 only (e.g., DB E4).
	if contains(dec.text, "%?", "<internal disassembler error>", "(8087 only)", "(287 only)") {
		dec.text = "(bad)"
	}

	// 0F 19 11, 0F 1C 11, 0F 1D 11, 0F 1E 11, 0F 1F 11: libopcodes says nop,
	// but the Intel manuals say that the only NOP there is 0F 1F /0.
	// Perhaps libopcodes is reporting an older encoding.
	i := bytes.IndexByte(dec.enc[:], 0x0F)
	if contains(dec.text, "nop") && i >= 0 && i+2 < len(dec.enc) && dec.enc[i+1]&^7 == 0x18 && (dec.enc[i+1] != 0x1F || (dec.enc[i+2]>>3)&7 != 0) {
		dec.text = "(bad)"
	}

	// Any invalid instruction.
	if text == "BAD" && contains(dec.text, "(bad)") {
		return true
	}

	// Instructions libopcodes knows but we do not (e.g., 0F 19 11).
	if (text == "BAD" || size == 1 && isPrefix(text)) && hasPrefix(dec.text, unsupported...) {
		return true
	}

	// Instructions we know but libopcodes does not (e.g., 0F D0 11).
	if (contains(dec.text, "(bad)") || dec.nenc == 1 && isPrefix(dec.text)) && hasPrefix(text, libopcodesUnsupported...) {
		return true
	}

	// Libopcodes rejects F2 90 as NOP. Not sure why.
	if (contains(dec.text, "(bad)") || dec.nenc == 1 && isPrefix(dec.text)) && inst.Opcode>>24 == 0x90 && countPrefix(inst, 0xF2) > 0 {
		return true
	}

	// 0F 20 11, 0F 21 11, 0F 22 11, 0F 23 11, 0F 24 11:
	// Moves into and out of some control registers seem to be unsupported by libopcodes.
	// TODO(rsc): Are they invalid somehow?
	if (contains(dec.text, "(bad)") || dec.nenc == 1 && isPrefix(dec.text)) && contains(text, "%cr", "%db", "%tr") {
		return true
	}

	if contains(dec.text, "fwait") && dec.nenc == 1 && dec.enc[0] != 0x9B {
		return true
	}

	// 9B D9 11: libopcodes reports FSTSW instead of FWAIT + FNSTSW.
	// This is correct in that FSTSW is a pseudo-op for the pair, but it really
	// is a pair of instructions: execution can stop between them.
	// Our decoder chooses to separate them.
	if (text == "fwait" || strings.HasSuffix(text, " fwait")) && dec.nenc >= len(strings.Fields(text)) && dec.enc[len(strings.Fields(text))-1] == 0x9B {
		return true
	}

	// 0F 18 77 11:
	// Invalid instructions for which libopcodes prints "nop/reserved".
	// Perhaps libopcodes is reporting an older encoding.
	if text == "BAD" && contains(dec.text, "nop/reserved") {
		return true
	}

	// 0F C7 B0 11 22 33 44: libopcodes says vmptrld 0x44332211(%eax); we say rdrand %eax.
	// TODO(rsc): Fix, since we are probably wrong, but we don't have vmptrld in the manual.
	if contains(text, "rdrand") && contains(dec.text, "vmptrld", "vmxon", "vmclear") {
		return true
	}

	// DD C8: libopcodes says FNOP but the Intel manual is clear FNOP is only D9 D0.
	// Perhaps libopcodes is reporting an older encoding.
	if text == "BAD" && contains(dec.text, "fnop") && (dec.enc[0] != 0xD9 || dec.enc[1] != 0xD0) {
		return true
	}

	// 66 90: libopcodes says xchg %ax,%ax; we say 'data16 nop'.
	// The 16-bit swap will preserve the high bits of the register,
	// so they are the same.
	if contains(text, "nop") && contains(dec.text, "xchg %ax,%ax") {
		return true
	}

	// If there are multiple prefixes, allow libopcodes to use an alternate name.
	if size == 1 && dec.nenc == 1 && prefixByte[text] > 0 && prefixByte[text] == prefixByte[dec.text] {
		return true
	}

	// 26 9B: libopcodes reports "fwait"/1, ignoring segment prefix.
	// https://sourceware.org/bugzilla/show_bug.cgi?id=16891
	// F0 82: Decode="lock"/1 but libopcodes="lock (bad)"/2.
	if size == 1 && dec.nenc >= 1 && prefixByte[text] == dec.enc[0] && contains(dec.text, "(bad)", "fwait", "fnop") {
		return true
	}

	// libopcodes interprets 660f801122 as taking a rel16 but
	// truncating the address at 16 bits. Not sure what is correct.
	if contains(text, ".+0x2211", ".+0x11") && contains(dec.text, " .-") {
		return true
	}

	// 66 F3 0F D6 C5, 66 F2 0F D6 C0: libopcodes reports use of XMM register instead of MMX register,
	// but only when the instruction has a 66 prefix. Maybe they know something we don't.
	if countPrefix(inst, 0x66) > 0 && contains(dec.text, "movdq2q", "movq2dq") && !contains(dec.text, "%mm") {
		return true
	}

	// 0F 01 F8, 0F 05, 0F 07: these are 64-bit instructions but libopcodes accepts them.
	if (text == "BAD" || size == 1 && isPrefix(text)) && contains(dec.text, "swapgs", "syscall", "sysret", "rdfsbase", "rdgsbase", "wrfsbase", "wrgsbase") {
		return true
	}

	return false
}

// Instructions known to libopcodes (or xed) but not to us.
// Most of these come from supplementary manuals of one form or another.
var unsupported = strings.Fields(`
	bndc
	bndl
	bndm
	bnds
	clac
	clgi
	femms
	fldln
	fldz
	getsec
	invlpga
	kmov
	montmul
	pavg
	pf2i
	pfacc
	pfadd
	pfcmp
	pfmax
	pfmin
	pfmul
	pfna
	pfpnac
	pfrc
	pfrs
	pfsub
	phadd
	phsub
	pi2f
	pmulhr
	prefetch
	pswap
	ptest
	rdseed
	sha1
	sha256
	skinit
	stac
	stgi
	vadd
	vand
	vcmp
	vcomis
	vcvt
	vcvt
	vdiv
	vhadd
	vhsub
	vld
	vmax
	vmcall
	vmfunc
	vmin
	vmlaunch
	vmload
	vmmcall
	vmov
	vmov
	vmov
	vmptrld
	vmptrst
	vmread
	vmresume
	vmrun
	vmsave
	vmul
	vmwrite
	vmxoff
	vor
	vpack
	vpadd
	vpand
	vpavg
	vpcmp
	vpcmp
	vpins
	vpmadd
	vpmax
	vpmin
	vpmul
	vpmul
	vpor
	vpsad
	vpshuf
	vpsll
	vpsra
	vpsrad
	vpsrl
	vpsub
	vpunp
	vpxor
	vrcp
	vrsqrt
	vshuf
	vsqrt
	vsub
	vucomis
	vunp
	vxor
	vzero
	xcrypt
	xsha1
	xsha256
	xstore-rng
	insertq
	extrq
	vmclear
	invvpid
	adox
	vmxon
	invept
	adcx
	vmclear
	prefetchwt1
	enclu
	encls
	salc
	fstpnce
	fdisi8087_nop
	fsetpm287_nop
	feni8087_nop
	syscall
	sysret
`)

// Instructions known to us but not to libopcodes (at least in binutils 2.24).
var libopcodesUnsupported = strings.Fields(`
	addsubps
	aes
	blend
	cvttpd2dq
	dpp
	extract
	haddps
	hsubps
	insert
	invpcid
	lddqu
	movmsk
	movnt
	movq2dq
	mps
	pack
	pblend
	pclmul
	pcmp
	pext
	phmin
	pins
	pmax
	pmin
	pmov
	pmovmsk
	pmul
	popcnt
	pslld
	psllq
	psllw
	psrad
	psraw
	psrl
	ptest
	punpck
	round
	xrstor
	xsavec
	xsaves
	comis
	ucomis
	movhps
	movntps
	rsqrt
	rcpp
	puncpck
	bsf
	movq2dq
	cvttpd2dq
	movq
	hsubpd
	movdqa
	movhpd
	addsubpd
	movd
	haddpd
	cvtps2dq
	bsr
	cvtdq2ps
	rdrand
	maskmov
	movq2dq
	movlhps
	movbe
	movlpd
`)
