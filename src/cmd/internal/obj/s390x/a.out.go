// Based on cmd/internal/obj/ppc64/a.out.go.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package s390x

import "cmd/internal/obj"

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p s390x

const (
	NSNAME = 8
	NSYM   = 50
	NREG   = 16 // number of general purpose registers
	NFREG  = 16 // number of floating point registers
)

const (
	// General purpose registers (GPRs).
	REG_R0 = obj.RBaseS390X + iota
	REG_R1
	REG_R2
	REG_R3
	REG_R4
	REG_R5
	REG_R6
	REG_R7
	REG_R8
	REG_R9
	REG_R10
	REG_R11
	REG_R12
	REG_R13
	REG_R14
	REG_R15

	// Floating point registers (FPRs).
	REG_F0
	REG_F1
	REG_F2
	REG_F3
	REG_F4
	REG_F5
	REG_F6
	REG_F7
	REG_F8
	REG_F9
	REG_F10
	REG_F11
	REG_F12
	REG_F13
	REG_F14
	REG_F15

	// Vector registers (VRs) - only available when the vector
	// facility is installed.
	// V0-V15 are aliases for F0-F15.
	// We keep them in a separate space to make printing etc. easier
	// If the code generator ever emits vector instructions it will
	// need to take into account the aliasing.
	REG_V0
	REG_V1
	REG_V2
	REG_V3
	REG_V4
	REG_V5
	REG_V6
	REG_V7
	REG_V8
	REG_V9
	REG_V10
	REG_V11
	REG_V12
	REG_V13
	REG_V14
	REG_V15
	REG_V16
	REG_V17
	REG_V18
	REG_V19
	REG_V20
	REG_V21
	REG_V22
	REG_V23
	REG_V24
	REG_V25
	REG_V26
	REG_V27
	REG_V28
	REG_V29
	REG_V30
	REG_V31

	// Access registers (ARs).
	// The thread pointer is typically stored in the register pair
	// AR0 and AR1.
	REG_AR0
	REG_AR1
	REG_AR2
	REG_AR3
	REG_AR4
	REG_AR5
	REG_AR6
	REG_AR7
	REG_AR8
	REG_AR9
	REG_AR10
	REG_AR11
	REG_AR12
	REG_AR13
	REG_AR14
	REG_AR15

	REG_RESERVED // end of allocated registers

	REGZERO = REG_R0  // set to zero
	REGARG  = -1      // -1 disables passing the first argument in register
	REGRT1  = REG_R3  // used during zeroing of the stack - not reserved
	REGRT2  = REG_R4  // used during zeroing of the stack - not reserved
	REGTMP  = REG_R10 // scratch register used in the assembler and linker
	REGTMP2 = REG_R11 // scratch register used in the assembler and linker
	REGCTXT = REG_R12 // context for closures
	REGG    = REG_R13 // G
	REG_LR  = REG_R14 // link register
	REGSP   = REG_R15 // stack pointer
)

const (
	BIG    = 32768 - 8
	DISP12 = 4096
	DISP16 = 65536
	DISP20 = 1048576
)

const (
	// mark flags
	LABEL   = 1 << 0
	LEAF    = 1 << 1
	FLOAT   = 1 << 2
	BRANCH  = 1 << 3
	LOAD    = 1 << 4
	FCMP    = 1 << 5
	SYNC    = 1 << 6
	LIST    = 1 << 7
	FOLL    = 1 << 8
	NOSCHED = 1 << 9
)

const ( // comments from func aclass in asmz.go
	C_NONE     = iota
	C_REG      // general-purpose register (64-bit)
	C_FREG     // floating-point register (64-bit)
	C_VREG     // vector register (128-bit)
	C_AREG     // access register (32-bit)
	C_ZCON     // constant == 0
	C_SCON     // 0 <= constant <= 0x7fff (positive int16)
	C_UCON     // constant & 0xffff == 0 (int16 or uint16)
	C_ADDCON   // 0 > constant >= -0x8000 (negative int16)
	C_ANDCON   // constant <= 0xffff
	C_LCON     // constant (int32 or uint32)
	C_DCON     // constant (int64 or uint64)
	C_SACON    // computed address, 16-bit displacement, possibly SP-relative
	C_LACON    // computed address, 32-bit displacement, possibly SP-relative
	C_DACON    // computed address, 64-bit displacment?
	C_SBRA     // short branch
	C_LBRA     // long branch
	C_SAUTO    // short auto
	C_LAUTO    // long auto
	C_ZOREG    // heap address, register-based, displacement == 0
	C_SOREG    // heap address, register-based, int16 displacement
	C_LOREG    // heap address, register-based, int32 displacement
	C_TLS_LE   // TLS - local exec model (for executables)
	C_TLS_IE   // TLS - initial exec model (for shared libraries loaded at program startup)
	C_GOK      // general address
	C_ADDR     // relocation for extern or static symbols (loads and stores)
	C_SYMADDR  // relocation for extern or static symbols (address taking)
	C_GOTADDR  // GOT slot for a symbol in -dynlink mode
	C_TEXTSIZE // text size
	C_ANY
	C_NCLASS // must be the last
)

const (
	// integer arithmetic
	AADD = obj.ABaseS390X + obj.A_ARCHSPECIFIC + iota
	AADDC
	AADDE
	AADDW
	ADIVW
	ADIVWU
	ADIVD
	ADIVDU
	AMODW
	AMODWU
	AMODD
	AMODDU
	AMULLW
	AMULLD
	AMULHD
	AMULHDU
	ASUB
	ASUBC
	ASUBV
	ASUBE
	ASUBW
	ANEG
	ANEGW

	// integer moves
	AMOVWBR
	AMOVB
	AMOVBZ
	AMOVH
	AMOVHBR
	AMOVHZ
	AMOVW
	AMOVWZ
	AMOVD
	AMOVDBR

	// conditional moves
	AMOVDEQ
	AMOVDGE
	AMOVDGT
	AMOVDLE
	AMOVDLT
	AMOVDNE

	// find leftmost one
	AFLOGR

	// integer bitwise
	AAND
	AANDW
	AOR
	AORW
	AXOR
	AXORW
	ASLW
	ASLD
	ASRW
	ASRAW
	ASRD
	ASRAD
	ARLL
	ARLLG

	// floating point
	AFABS
	AFADD
	AFADDS
	AFCMPO
	AFCMPU
	ACEBR
	AFDIV
	AFDIVS
	AFMADD
	AFMADDS
	AFMOVD
	AFMOVS
	AFMSUB
	AFMSUBS
	AFMUL
	AFMULS
	AFNABS
	AFNEG
	AFNEGS
	AFNMADD
	AFNMADDS
	AFNMSUB
	AFNMSUBS
	ALEDBR
	ALDEBR
	AFSUB
	AFSUBS
	AFSQRT
	AFSQRTS
	AFIEBR
	AFIDBR

	// convert from int32/int64 to float/float64
	ACEFBRA
	ACDFBRA
	ACEGBRA
	ACDGBRA

	// convert from float/float64 to int32/int64
	ACFEBRA
	ACFDBRA
	ACGEBRA
	ACGDBRA

	// convert from uint32/uint64 to float/float64
	ACELFBR
	ACDLFBR
	ACELGBR
	ACDLGBR

	// convert from float/float64 to uint32/uint64
	ACLFEBR
	ACLFDBR
	ACLGEBR
	ACLGDBR

	// compare
	ACMP
	ACMPU
	ACMPW
	ACMPWU

	// compare and swap
	ACS
	ACSG

	// serialize
	ASYNC

	// branch
	ABC
	ABCL
	ABEQ
	ABGE
	ABGT
	ABLE
	ABLT
	ABLEU
	ABLTU
	ABNE
	ABVC
	ABVS
	ASYSCALL

	// compare and branch
	ACMPBEQ
	ACMPBGE
	ACMPBGT
	ACMPBLE
	ACMPBLT
	ACMPBNE
	ACMPUBEQ
	ACMPUBGE
	ACMPUBGT
	ACMPUBLE
	ACMPUBLT
	ACMPUBNE

	// storage-and-storage
	AMVC
	ACLC
	AXC
	AOC
	ANC

	// load
	AEXRL
	ALARL
	ALA
	ALAY

	// interlocked load and op
	ALAA
	ALAAG
	ALAAL
	ALAALG
	ALAN
	ALANG
	ALAX
	ALAXG
	ALAO
	ALAOG

	// load/store multiple
	ALMY
	ALMG
	ASTMY
	ASTMG

	// store clock
	ASTCK
	ASTCKC
	ASTCKE
	ASTCKF

	// macros
	ACLEAR

	// vector
	AVA
	AVAB
	AVAH
	AVAF
	AVAG
	AVAQ
	AVACC
	AVACCB
	AVACCH
	AVACCF
	AVACCG
	AVACCQ
	AVAC
	AVACQ
	AVACCC
	AVACCCQ
	AVN
	AVNC
	AVAVG
	AVAVGB
	AVAVGH
	AVAVGF
	AVAVGG
	AVAVGL
	AVAVGLB
	AVAVGLH
	AVAVGLF
	AVAVGLG
	AVCKSM
	AVCEQ
	AVCEQB
	AVCEQH
	AVCEQF
	AVCEQG
	AVCEQBS
	AVCEQHS
	AVCEQFS
	AVCEQGS
	AVCH
	AVCHB
	AVCHH
	AVCHF
	AVCHG
	AVCHBS
	AVCHHS
	AVCHFS
	AVCHGS
	AVCHL
	AVCHLB
	AVCHLH
	AVCHLF
	AVCHLG
	AVCHLBS
	AVCHLHS
	AVCHLFS
	AVCHLGS
	AVCLZ
	AVCLZB
	AVCLZH
	AVCLZF
	AVCLZG
	AVCTZ
	AVCTZB
	AVCTZH
	AVCTZF
	AVCTZG
	AVEC
	AVECB
	AVECH
	AVECF
	AVECG
	AVECL
	AVECLB
	AVECLH
	AVECLF
	AVECLG
	AVERIM
	AVERIMB
	AVERIMH
	AVERIMF
	AVERIMG
	AVERLL
	AVERLLB
	AVERLLH
	AVERLLF
	AVERLLG
	AVERLLV
	AVERLLVB
	AVERLLVH
	AVERLLVF
	AVERLLVG
	AVESLV
	AVESLVB
	AVESLVH
	AVESLVF
	AVESLVG
	AVESL
	AVESLB
	AVESLH
	AVESLF
	AVESLG
	AVESRA
	AVESRAB
	AVESRAH
	AVESRAF
	AVESRAG
	AVESRAV
	AVESRAVB
	AVESRAVH
	AVESRAVF
	AVESRAVG
	AVESRL
	AVESRLB
	AVESRLH
	AVESRLF
	AVESRLG
	AVESRLV
	AVESRLVB
	AVESRLVH
	AVESRLVF
	AVESRLVG
	AVX
	AVFAE
	AVFAEB
	AVFAEH
	AVFAEF
	AVFAEBS
	AVFAEHS
	AVFAEFS
	AVFAEZB
	AVFAEZH
	AVFAEZF
	AVFAEZBS
	AVFAEZHS
	AVFAEZFS
	AVFEE
	AVFEEB
	AVFEEH
	AVFEEF
	AVFEEBS
	AVFEEHS
	AVFEEFS
	AVFEEZB
	AVFEEZH
	AVFEEZF
	AVFEEZBS
	AVFEEZHS
	AVFEEZFS
	AVFENE
	AVFENEB
	AVFENEH
	AVFENEF
	AVFENEBS
	AVFENEHS
	AVFENEFS
	AVFENEZB
	AVFENEZH
	AVFENEZF
	AVFENEZBS
	AVFENEZHS
	AVFENEZFS
	AVFA
	AVFADB
	AWFADB
	AWFK
	AWFKDB
	AVFCE
	AVFCEDB
	AVFCEDBS
	AWFCEDB
	AWFCEDBS
	AVFCH
	AVFCHDB
	AVFCHDBS
	AWFCHDB
	AWFCHDBS
	AVFCHE
	AVFCHEDB
	AVFCHEDBS
	AWFCHEDB
	AWFCHEDBS
	AWFC
	AWFCDB
	AVCDG
	AVCDGB
	AWCDGB
	AVCDLG
	AVCDLGB
	AWCDLGB
	AVCGD
	AVCGDB
	AWCGDB
	AVCLGD
	AVCLGDB
	AWCLGDB
	AVFD
	AVFDDB
	AWFDDB
	AVLDE
	AVLDEB
	AWLDEB
	AVLED
	AVLEDB
	AWLEDB
	AVFM
	AVFMDB
	AWFMDB
	AVFMA
	AVFMADB
	AWFMADB
	AVFMS
	AVFMSDB
	AWFMSDB
	AVFPSO
	AVFPSODB
	AWFPSODB
	AVFLCDB
	AWFLCDB
	AVFLNDB
	AWFLNDB
	AVFLPDB
	AWFLPDB
	AVFSQ
	AVFSQDB
	AWFSQDB
	AVFS
	AVFSDB
	AWFSDB
	AVFTCI
	AVFTCIDB
	AWFTCIDB
	AVGFM
	AVGFMB
	AVGFMH
	AVGFMF
	AVGFMG
	AVGFMA
	AVGFMAB
	AVGFMAH
	AVGFMAF
	AVGFMAG
	AVGEF
	AVGEG
	AVGBM
	AVZERO
	AVONE
	AVGM
	AVGMB
	AVGMH
	AVGMF
	AVGMG
	AVISTR
	AVISTRB
	AVISTRH
	AVISTRF
	AVISTRBS
	AVISTRHS
	AVISTRFS
	AVL
	AVLR
	AVLREP
	AVLREPB
	AVLREPH
	AVLREPF
	AVLREPG
	AVLC
	AVLCB
	AVLCH
	AVLCF
	AVLCG
	AVLEH
	AVLEF
	AVLEG
	AVLEB
	AVLEIH
	AVLEIF
	AVLEIG
	AVLEIB
	AVFI
	AVFIDB
	AWFIDB
	AVLGV
	AVLGVB
	AVLGVH
	AVLGVF
	AVLGVG
	AVLLEZ
	AVLLEZB
	AVLLEZH
	AVLLEZF
	AVLLEZG
	AVLM
	AVLP
	AVLPB
	AVLPH
	AVLPF
	AVLPG
	AVLBB
	AVLVG
	AVLVGB
	AVLVGH
	AVLVGF
	AVLVGG
	AVLVGP
	AVLL
	AVMX
	AVMXB
	AVMXH
	AVMXF
	AVMXG
	AVMXL
	AVMXLB
	AVMXLH
	AVMXLF
	AVMXLG
	AVMRH
	AVMRHB
	AVMRHH
	AVMRHF
	AVMRHG
	AVMRL
	AVMRLB
	AVMRLH
	AVMRLF
	AVMRLG
	AVMN
	AVMNB
	AVMNH
	AVMNF
	AVMNG
	AVMNL
	AVMNLB
	AVMNLH
	AVMNLF
	AVMNLG
	AVMAE
	AVMAEB
	AVMAEH
	AVMAEF
	AVMAH
	AVMAHB
	AVMAHH
	AVMAHF
	AVMALE
	AVMALEB
	AVMALEH
	AVMALEF
	AVMALH
	AVMALHB
	AVMALHH
	AVMALHF
	AVMALO
	AVMALOB
	AVMALOH
	AVMALOF
	AVMAL
	AVMALB
	AVMALHW
	AVMALF
	AVMAO
	AVMAOB
	AVMAOH
	AVMAOF
	AVME
	AVMEB
	AVMEH
	AVMEF
	AVMH
	AVMHB
	AVMHH
	AVMHF
	AVMLE
	AVMLEB
	AVMLEH
	AVMLEF
	AVMLH
	AVMLHB
	AVMLHH
	AVMLHF
	AVMLO
	AVMLOB
	AVMLOH
	AVMLOF
	AVML
	AVMLB
	AVMLHW
	AVMLF
	AVMO
	AVMOB
	AVMOH
	AVMOF
	AVNO
	AVNOT
	AVO
	AVPK
	AVPKH
	AVPKF
	AVPKG
	AVPKLS
	AVPKLSH
	AVPKLSF
	AVPKLSG
	AVPKLSHS
	AVPKLSFS
	AVPKLSGS
	AVPKS
	AVPKSH
	AVPKSF
	AVPKSG
	AVPKSHS
	AVPKSFS
	AVPKSGS
	AVPERM
	AVPDI
	AVPOPCT
	AVREP
	AVREPB
	AVREPH
	AVREPF
	AVREPG
	AVREPI
	AVREPIB
	AVREPIH
	AVREPIF
	AVREPIG
	AVSCEF
	AVSCEG
	AVSEL
	AVSL
	AVSLB
	AVSLDB
	AVSRA
	AVSRAB
	AVSRL
	AVSRLB
	AVSEG
	AVSEGB
	AVSEGH
	AVSEGF
	AVST
	AVSTEH
	AVSTEF
	AVSTEG
	AVSTEB
	AVSTM
	AVSTL
	AVSTRC
	AVSTRCB
	AVSTRCH
	AVSTRCF
	AVSTRCBS
	AVSTRCHS
	AVSTRCFS
	AVSTRCZB
	AVSTRCZH
	AVSTRCZF
	AVSTRCZBS
	AVSTRCZHS
	AVSTRCZFS
	AVS
	AVSB
	AVSH
	AVSF
	AVSG
	AVSQ
	AVSCBI
	AVSCBIB
	AVSCBIH
	AVSCBIF
	AVSCBIG
	AVSCBIQ
	AVSBCBI
	AVSBCBIQ
	AVSBI
	AVSBIQ
	AVSUMG
	AVSUMGH
	AVSUMGF
	AVSUMQ
	AVSUMQF
	AVSUMQG
	AVSUM
	AVSUMB
	AVSUMH
	AVTM
	AVUPH
	AVUPHB
	AVUPHH
	AVUPHF
	AVUPLH
	AVUPLHB
	AVUPLHH
	AVUPLHF
	AVUPLL
	AVUPLLB
	AVUPLLH
	AVUPLLF
	AVUPL
	AVUPLB
	AVUPLHW
	AVUPLF

	// binary
	ABYTE
	AWORD
	ADWORD

	// end marker
	ALAST

	// aliases
	ABR = obj.AJMP
	ABL = obj.ACALL
)
