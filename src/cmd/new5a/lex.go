// Inferno utils/5a/lex.c
// http://code.google.com/p/inferno-os/source/browse/utils/5a/lex.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.	All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

package main

const (
	Plan9   = 1 << 0
	Unix    = 1 << 1
	Windows = 1 << 2
)

func systemtype(sys int) int {
	return sys & Windows

	return sys & Plan9
}

func Lconv(fp *obj.Fmt) int {
	return obj.Linklinefmt(Ctxt, fp)
}

func dodef(p string) {
	if nDlist%8 == 0 {
		Dlist = asm.Allocn(Dlist, nDlist*sizeof(string), 8*sizeof(string)).(*string)
	}
	Dlist[nDlist] = p
	nDlist++
}

func usage() {
	fmt.Printf("usage: %ca [options] file.c...\n", Thechar)
	main.Flagprint(1)
	asm.Errorexit()
}

func main(argc int, argv [XXX]string) {
	var p string

	Thechar = '5'
	thestring = "arm"

	Ctxt = obj.Linknew(&arm.Linkarm)
	Ctxt.Diag = asm.Yyerror
	Ctxt.Bso = &Bstdout
	Ctxt.Enforce_data_order = 1
	obj.Binit(&Bstdout, 1, main.OWRITE)
	arm.Listinit5()
	obj.Fmtinstall('L', Lconv)

	// Allow GOARCH=thestring or GOARCH=thestringsuffix,
	// but not other values.
	p = Getgoarch()

	if !strings.HasPrefix(p, thestring) {
		log.Fatalf("cannot use %cc with GOARCH=%s", Thechar, p)
	}

	asm.Ensuresymb(NSYMB)
	Debug = [256]int{}
	cinit()
	Outfile = ""
	asm.Setinclude(".")

	main.Flagfn1("D", "name[=value]: add #define", dodef)
	main.Flagfn1("I", "dir: add dir to include path", asm.Setinclude)
	main.Flagcount("S", "print assembly and machine code", &Debug['S'])
	main.Flagcount("m", "debug preprocessor macros", &Debug['m'])
	main.Flagstr("o", "file: set output file", &Outfile)
	main.Flagstr("trimpath", "prefix: remove prefix from recorded source file paths", &Ctxt.Trimpath)

	main.Flagparse(&argc, (**string)(&argv), usage)
	Ctxt.Debugasm = int32(Debug['S'])

	if argc < 1 {
		usage()
	}
	if argc > 1 {
		fmt.Printf("can't assemble multiple files\n")
		asm.Errorexit()
	}

	if assemble(argv[0]) != 0 {
		asm.Errorexit()
	}
	obj.Bflush(&Bstdout)
	if Nerrors > 0 {
		asm.Errorexit()
	}
	main.Exits("")
}

func assemble(file string) int {
	var ofile string
	var p string
	var i int
	var of int

	ofile = asm.Alloc(int32(len(file)) + 3).(string) // +3 for .x\0 (x=thechar)
	ofile = file
	p = main.Utfrrune(ofile, '/')
	if p != "" {
		Include[0] = ofile
		p = ""
		p = p[1:]
	} else {

		p = ofile
	}
	if Outfile == "" {
		Outfile = p
		if Outfile != "" {
			p = main.Utfrrune(Outfile, '.')
			if p != "" {
				if p[1] == 's' && p[2] == 0 {
					p = ""
				}
			}
			p = main.Utfrune(Outfile, 0)
			p[0] = '.'
			p[1] = byte(Thechar)
			p[2] = 0
		} else {

			Outfile = "/dev/null"
		}
	}

	of = main.Create(Outfile, main.OWRITE, 0664)
	if of < 0 {
		asm.Yyerror("%ca: cannot create %s", Thechar, Outfile)
		asm.Errorexit()
	}

	obj.Binit(&obuf, of, main.OWRITE)
	fmt.Fprintf(&obuf, "go object %s %s %s\n", main.Getgoos(), main.Getgoarch(), main.Getgoversion())
	fmt.Fprintf(&obuf, "!\n")

	for pass = 1; pass <= 2; pass++ {
		asm.Pinit(file)
		for i = 0; i < nDlist; i++ {
			asm.Dodefine(Dlist[i])
		}
		yyparse()
		cclean()
		if Nerrors != 0 {
			return Nerrors
		}
	}

	obj.Writeobj(Ctxt, &obuf)
	obj.Bflush(&obuf)
	return 0
}

var itab = []struct {
	name  string
	type_ uint16
	value uint16
}{
	{"SP", LSP, arm.D_AUTO},
	{"SB", LSB, arm.D_EXTERN},
	{"FP", LFP, arm.D_PARAM},
	{"PC", LPC, arm.D_BRANCH},
	{"R", LR, 0},
	{"R0", LREG, 0},
	{"R1", LREG, 1},
	{"R2", LREG, 2},
	{"R3", LREG, 3},
	{"R4", LREG, 4},
	{"R5", LREG, 5},
	{"R6", LREG, 6},
	{"R7", LREG, 7},
	{"R8", LREG, 8},
	{"R9", LREG, 9},
	{"g", LREG, 10}, // avoid unintentionally clobber g using R10
	{"R11", LREG, 11},
	{"R12", LREG, 12},
	{"R13", LREG, 13},
	{"R14", LREG, 14},
	{"R15", LREG, 15},
	{"F", LF, 0},
	{"F0", LFREG, 0},
	{"F1", LFREG, 1},
	{"F2", LFREG, 2},
	{"F3", LFREG, 3},
	{"F4", LFREG, 4},
	{"F5", LFREG, 5},
	{"F6", LFREG, 6},
	{"F7", LFREG, 7},
	{"F8", LFREG, 8},
	{"F9", LFREG, 9},
	{"F10", LFREG, 10},
	{"F11", LFREG, 11},
	{"F12", LFREG, 12},
	{"F13", LFREG, 13},
	{"F14", LFREG, 14},
	{"F15", LFREG, 15},
	{"C", LC, 0},
	{"C0", LCREG, 0},
	{"C1", LCREG, 1},
	{"C2", LCREG, 2},
	{"C3", LCREG, 3},
	{"C4", LCREG, 4},
	{"C5", LCREG, 5},
	{"C6", LCREG, 6},
	{"C7", LCREG, 7},
	{"C8", LCREG, 8},
	{"C9", LCREG, 9},
	{"C10", LCREG, 10},
	{"C11", LCREG, 11},
	{"C12", LCREG, 12},
	{"C13", LCREG, 13},
	{"C14", LCREG, 14},
	{"C15", LCREG, 15},
	{"CPSR", LPSR, 0},
	{"SPSR", LPSR, 1},
	{"FPSR", LFCR, 0},
	{"FPCR", LFCR, 1},
	{".EQ", LCOND, 0},
	{".NE", LCOND, 1},
	{".CS", LCOND, 2},
	{".HS", LCOND, 2},
	{".CC", LCOND, 3},
	{".LO", LCOND, 3},
	{".MI", LCOND, 4},
	{".PL", LCOND, 5},
	{".VS", LCOND, 6},
	{".VC", LCOND, 7},
	{".HI", LCOND, 8},
	{".LS", LCOND, 9},
	{".GE", LCOND, 10},
	{".LT", LCOND, 11},
	{".GT", LCOND, 12},
	{".LE", LCOND, 13},
	{".AL", LCOND, Always},
	{".U", LS, arm.C_UBIT},
	{".S", LS, arm.C_SBIT},
	{".W", LS, arm.C_WBIT},
	{".P", LS, arm.C_PBIT},
	{".PW", LS, arm.C_WBIT | arm.C_PBIT},
	{".WP", LS, arm.C_WBIT | arm.C_PBIT},
	{".F", LS, arm.C_FBIT},
	{".IBW", LS, arm.C_WBIT | arm.C_PBIT | arm.C_UBIT},
	{".IAW", LS, arm.C_WBIT | arm.C_UBIT},
	{".DBW", LS, arm.C_WBIT | arm.C_PBIT},
	{".DAW", LS, arm.C_WBIT},
	{".IB", LS, arm.C_PBIT | arm.C_UBIT},
	{".IA", LS, arm.C_UBIT},
	{".DB", LS, arm.C_PBIT},
	{".DA", LS, 0},
	{"@", LAT, 0},
	{"AND", LTYPE1, arm.AAND},
	{"EOR", LTYPE1, arm.AEOR},
	{"SUB", LTYPE1, arm.ASUB},
	{"RSB", LTYPE1, arm.ARSB},
	{"ADD", LTYPE1, arm.AADD},
	{"ADC", LTYPE1, arm.AADC},
	{"SBC", LTYPE1, arm.ASBC},
	{"RSC", LTYPE1, arm.ARSC},
	{"ORR", LTYPE1, arm.AORR},
	{"BIC", LTYPE1, arm.ABIC},
	{"SLL", LTYPE1, arm.ASLL},
	{"SRL", LTYPE1, arm.ASRL},
	{"SRA", LTYPE1, arm.ASRA},
	{"MUL", LTYPE1, arm.AMUL},
	{"MULA", LTYPEN, arm.AMULA},
	{"DIV", LTYPE1, arm.ADIV},
	{"MOD", LTYPE1, arm.AMOD},
	{"MULL", LTYPEM, arm.AMULL},
	{"MULAL", LTYPEM, arm.AMULAL},
	{"MULLU", LTYPEM, arm.AMULLU},
	{"MULALU", LTYPEM, arm.AMULALU},
	{"MVN", LTYPE2, arm.AMVN}, /* op2 ignored */
	{"MOVB", LTYPE3, arm.AMOVB},
	{"MOVBU", LTYPE3, arm.AMOVBU},
	{"MOVH", LTYPE3, arm.AMOVH},
	{"MOVHU", LTYPE3, arm.AMOVHU},
	{"MOVW", LTYPE3, arm.AMOVW},
	{"MOVD", LTYPE3, arm.AMOVD},
	{"MOVDF", LTYPE3, arm.AMOVDF},
	{"MOVDW", LTYPE3, arm.AMOVDW},
	{"MOVF", LTYPE3, arm.AMOVF},
	{"MOVFD", LTYPE3, arm.AMOVFD},
	{"MOVFW", LTYPE3, arm.AMOVFW},
	{"MOVWD", LTYPE3, arm.AMOVWD},
	{"MOVWF", LTYPE3, arm.AMOVWF},
	{"LDREX", LTYPE3, arm.ALDREX},
	{"LDREXD", LTYPE3, arm.ALDREXD},
	{"STREX", LTYPE9, arm.ASTREX},
	{"STREXD", LTYPE9, arm.ASTREXD},

	/*
		{"NEGF",		LTYPEI, ANEGF},
		{"NEGD",		LTYPEI, ANEGD},
		{"SQTF",		LTYPEI,	ASQTF},
		{"SQTD",		LTYPEI,	ASQTD},
		{"RNDF",		LTYPEI,	ARNDF},
		{"RNDD",		LTYPEI,	ARNDD},
		{"URDF",		LTYPEI,	AURDF},
		{"URDD",		LTYPEI,	AURDD},
		{"NRMF",		LTYPEI,	ANRMF},
		{"NRMD",		LTYPEI,	ANRMD},
	*/
	{"ABSF", LTYPEI, arm.AABSF},
	{"ABSD", LTYPEI, arm.AABSD},
	{"SQRTF", LTYPEI, arm.ASQRTF},
	{"SQRTD", LTYPEI, arm.ASQRTD},
	{"CMPF", LTYPEL, arm.ACMPF},
	{"CMPD", LTYPEL, arm.ACMPD},
	{"ADDF", LTYPEK, arm.AADDF},
	{"ADDD", LTYPEK, arm.AADDD},
	{"SUBF", LTYPEK, arm.ASUBF},
	{"SUBD", LTYPEK, arm.ASUBD},
	{"MULF", LTYPEK, arm.AMULF},
	{"MULD", LTYPEK, arm.AMULD},
	{"DIVF", LTYPEK, arm.ADIVF},
	{"DIVD", LTYPEK, arm.ADIVD},
	{"B", LTYPE4, arm.AB},
	{"BL", LTYPE4, arm.ABL},
	{"BX", LTYPEBX, arm.ABX},
	{"BEQ", LTYPE5, arm.ABEQ},
	{"BNE", LTYPE5, arm.ABNE},
	{"BCS", LTYPE5, arm.ABCS},
	{"BHS", LTYPE5, arm.ABHS},
	{"BCC", LTYPE5, arm.ABCC},
	{"BLO", LTYPE5, arm.ABLO},
	{"BMI", LTYPE5, arm.ABMI},
	{"BPL", LTYPE5, arm.ABPL},
	{"BVS", LTYPE5, arm.ABVS},
	{"BVC", LTYPE5, arm.ABVC},
	{"BHI", LTYPE5, arm.ABHI},
	{"BLS", LTYPE5, arm.ABLS},
	{"BGE", LTYPE5, arm.ABGE},
	{"BLT", LTYPE5, arm.ABLT},
	{"BGT", LTYPE5, arm.ABGT},
	{"BLE", LTYPE5, arm.ABLE},
	{"BCASE", LTYPE5, arm.ABCASE},
	{"SWI", LTYPE6, arm.ASWI},
	{"CMP", LTYPE7, arm.ACMP},
	{"TST", LTYPE7, arm.ATST},
	{"TEQ", LTYPE7, arm.ATEQ},
	{"CMN", LTYPE7, arm.ACMN},
	{"MOVM", LTYPE8, arm.AMOVM},
	{"SWPBU", LTYPE9, arm.ASWPBU},
	{"SWPW", LTYPE9, arm.ASWPW},
	{"RET", LTYPEA, arm.ARET},
	{"RFE", LTYPEA, arm.ARFE},
	{"TEXT", LTYPEB, arm.ATEXT},
	{"GLOBL", LTYPEB, arm.AGLOBL},
	{"DATA", LTYPEC, arm.ADATA},
	{"CASE", LTYPED, arm.ACASE},
	{"END", LTYPEE, arm.AEND},
	{"WORD", LTYPEH, arm.AWORD},
	{"NOP", LTYPEI, arm.ANOP},
	{"MCR", LTYPEJ, 0},
	{"MRC", LTYPEJ, 1},
	{"PLD", LTYPEPLD, arm.APLD},
	{"UNDEF", LTYPEE, arm.AUNDEF},
	{"CLZ", LTYPE2, arm.ACLZ},
	{"MULWT", LTYPE1, arm.AMULWT},
	{"MULWB", LTYPE1, arm.AMULWB},
	{"MULAWT", LTYPEN, arm.AMULAWT},
	{"MULAWB", LTYPEN, arm.AMULAWB},
	{"USEFIELD", LTYPEN, arm.AUSEFIELD},
	{"PCDATA", LTYPEPC, arm.APCDATA},
	{"FUNCDATA", LTYPEF, arm.AFUNCDATA},
}

func cinit() {
	var s *Sym
	var i int

	Nullgen.Type_ = arm.D_NONE
	Nullgen.Name = arm.D_NONE
	Nullgen.Reg = arm.NREG

	Nerrors = 0
	Iostack = nil
	Iofree = nil
	Peekc = IGN
	nhunk = 0
	for i = 0; i < NHASH; i++ {
		Hash[i] = nil
	}
	for i = 0; itab[i].name != ""; i++ {
		s = asm.Slookup(itab[i].name)
		s.Type_ = itab[i].type_
		s.Value = int32(itab[i].value)
	}
}

func Syminit(s *Sym) {
	s.Type_ = LNAME
	s.Value = 0
}

func isreg(g *obj.Addr) int {
	return 1
}

func cclean() {
	outcode(arm.AEND, Always, &Nullgen, arm.NREG, &Nullgen)
}

var bcode = []int{
	arm.ABEQ,
	arm.ABNE,
	arm.ABCS,
	arm.ABCC,
	arm.ABMI,
	arm.ABPL,
	arm.ABVS,
	arm.ABVC,
	arm.ABHI,
	arm.ABLS,
	arm.ABGE,
	arm.ABLT,
	arm.ABGT,
	arm.ABLE,
	arm.AB,
	arm.ANOP,
}

var lastpc *obj.Prog

func outcode(a int, scond int, g1 *obj.Addr, reg int, g2 *obj.Addr) {
	var p *obj.Prog
	var pl *obj.Plist

	/* hack to make B.NE etc. work: turn it into the corresponding conditional */
	if a == arm.AB {

		a = bcode[scond&0xf]
		scond = scond&^0xf | Always
	}

	if pass == 1 {
		goto out
	}

	p = new(obj.Prog)
	*p = obj.Prog{}
	p.As = int16(a)
	p.Lineno = stmtline
	p.Scond = uint8(scond)
	p.From = *g1
	p.Reg = uint8(reg)
	p.To = *g2
	p.Pc = int64(Pc)

	if lastpc == nil {
		pl = obj.Linknewplist(Ctxt)
		pl.Firstpc = p
	} else {

		lastpc.Link = p
	}
	lastpc = p

out:
	if a != arm.AGLOBL && a != arm.ADATA {
		Pc++
	}
}
