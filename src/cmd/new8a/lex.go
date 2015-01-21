// Inferno utils/8a/lex.c
// http://code.google.com/p/inferno-os/source/browse/utils/8a/lex.c
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

func pathchar() int {
	return '/'
}

func Lconv(fp *obj.Fmt) int {
	return obj.Linklinefmt(ctxt, fp)
}

func dodef(p string) {
	if nDlist%8 == 0 {
		Dlist = allocn(Dlist, nDlist*sizeof(string), 8*sizeof(string)).(*string)
	}
	Dlist[nDlist] = p
	nDlist++
}

func usage() {
	fmt.Printf("usage: %ca [options] file.c...\n", thechar)
	main.Flagprint(1)
	errorexit()
}

func main(argc int, argv [XXX]string) {
	var p string

	thechar = '8'
	thestring = "386"

	ctxt = obj.Linknew(&i386.Link386)
	ctxt.Diag = yyerror
	ctxt.Bso = &bstdout
	ctxt.Enforce_data_order = 1
	obj.Binit(&bstdout, 1, main.OWRITE)
	i386.Listinit8()
	obj.Fmtinstall('L', Lconv)

	// Allow GOARCH=thestring or GOARCH=thestringsuffix,
	// but not other values.
	p = Getgoarch()

	if !strings.HasPrefix(p, thestring) {
		log.Fatalf("cannot use %cc with GOARCH=%s", thechar, p)
	}

	ensuresymb(NSYMB)
	debug = [256]int{}
	cinit()
	outfile = ""
	setinclude(".")

	main.Flagfn1("D", "name[=value]: add #define", dodef)
	main.Flagfn1("I", "dir: add dir to include path", setinclude)
	main.Flagcount("S", "print assembly and machine code", &debug['S'])
	main.Flagcount("m", "debug preprocessor macros", &debug['m'])
	main.Flagstr("o", "file: set output file", &outfile)
	main.Flagstr("trimpath", "prefix: remove prefix from recorded source file paths", &ctxt.Trimpath)

	main.Flagparse(&argc, (**string)(&argv), usage)
	ctxt.Debugasm = int32(debug['S'])

	if argc < 1 {
		usage()
	}
	if argc > 1 {
		fmt.Printf("can't assemble multiple files\n")
		errorexit()
	}

	if assemble(argv[0]) != 0 {
		errorexit()
	}
	obj.Bflush(&bstdout)
	if nerrors > 0 {
		errorexit()
	}
	main.Exits("")
}

func assemble(file string) int {
	var ofile string
	var p string
	var i int
	var of int

	ofile = alloc(int32(len(file)) + 3).(string) // +3 for .x\0 (x=thechar)
	ofile = file
	p = main.Utfrrune(ofile, uint(pathchar()))
	if p != "" {
		include[0] = ofile
		p = ""
		p = p[1:]
	} else {

		p = ofile
	}
	if outfile == "" {
		outfile = p
		if outfile != "" {
			p = main.Utfrrune(outfile, '.')
			if p != "" {
				if p[1] == 's' && p[2] == 0 {
					p = ""
				}
			}
			p = main.Utfrune(outfile, 0)
			p[0] = '.'
			p[1] = byte(thechar)
			p[2] = 0
		} else {

			outfile = "/dev/null"
		}
	}

	of = main.Create(outfile, main.OWRITE, 0664)
	if of < 0 {
		yyerror("%ca: cannot create %s", thechar, outfile)
		errorexit()
	}

	obj.Binit(&obuf, of, main.OWRITE)
	fmt.Fprintf(&obuf, "go object %s %s %s\n", main.Getgoos(), main.Getgoarch(), main.Getgoversion())
	fmt.Fprintf(&obuf, "!\n")

	for pass = 1; pass <= 2; pass++ {
		pinit(file)
		for i = 0; i < nDlist; i++ {
			dodefine(Dlist[i])
		}
		yyparse()
		cclean()
		if nerrors != 0 {
			return nerrors
		}
	}

	obj.Writeobj(ctxt, &obuf)
	obj.Bflush(&obuf)
	return 0
}

var itab = []struct {
	name  string
	type_ uint16
	value uint16
}{
	{"SP", LSP, i386.D_AUTO},
	{"SB", LSB, i386.D_EXTERN},
	{"FP", LFP, i386.D_PARAM},
	{"PC", LPC, i386.D_BRANCH},
	{"AL", LBREG, i386.D_AL},
	{"CL", LBREG, i386.D_CL},
	{"DL", LBREG, i386.D_DL},
	{"BL", LBREG, i386.D_BL},
	{"AH", LBREG, i386.D_AH},
	{"CH", LBREG, i386.D_CH},
	{"DH", LBREG, i386.D_DH},
	{"BH", LBREG, i386.D_BH},
	{"AX", LLREG, i386.D_AX},
	{"CX", LLREG, i386.D_CX},
	{"DX", LLREG, i386.D_DX},
	{"BX", LLREG, i386.D_BX},
	/*	"SP",		LLREG,	D_SP,	*/
	{"BP", LLREG, i386.D_BP},
	{"SI", LLREG, i386.D_SI},
	{"DI", LLREG, i386.D_DI},
	{"F0", LFREG, i386.D_F0 + 0},
	{"F1", LFREG, i386.D_F0 + 1},
	{"F2", LFREG, i386.D_F0 + 2},
	{"F3", LFREG, i386.D_F0 + 3},
	{"F4", LFREG, i386.D_F0 + 4},
	{"F5", LFREG, i386.D_F0 + 5},
	{"F6", LFREG, i386.D_F0 + 6},
	{"F7", LFREG, i386.D_F0 + 7},
	{"X0", LXREG, i386.D_X0 + 0},
	{"X1", LXREG, i386.D_X0 + 1},
	{"X2", LXREG, i386.D_X0 + 2},
	{"X3", LXREG, i386.D_X0 + 3},
	{"X4", LXREG, i386.D_X0 + 4},
	{"X5", LXREG, i386.D_X0 + 5},
	{"X6", LXREG, i386.D_X0 + 6},
	{"X7", LXREG, i386.D_X0 + 7},
	{"CS", LSREG, i386.D_CS},
	{"SS", LSREG, i386.D_SS},
	{"DS", LSREG, i386.D_DS},
	{"ES", LSREG, i386.D_ES},
	{"FS", LSREG, i386.D_FS},
	{"GS", LSREG, i386.D_GS},
	{"TLS", LSREG, i386.D_TLS},
	{"GDTR", LBREG, i386.D_GDTR},
	{"IDTR", LBREG, i386.D_IDTR},
	{"LDTR", LBREG, i386.D_LDTR},
	{"MSW", LBREG, i386.D_MSW},
	{"TASK", LBREG, i386.D_TASK},
	{"CR0", LBREG, i386.D_CR + 0},
	{"CR1", LBREG, i386.D_CR + 1},
	{"CR2", LBREG, i386.D_CR + 2},
	{"CR3", LBREG, i386.D_CR + 3},
	{"CR4", LBREG, i386.D_CR + 4},
	{"CR5", LBREG, i386.D_CR + 5},
	{"CR6", LBREG, i386.D_CR + 6},
	{"CR7", LBREG, i386.D_CR + 7},
	{"DR0", LBREG, i386.D_DR + 0},
	{"DR1", LBREG, i386.D_DR + 1},
	{"DR2", LBREG, i386.D_DR + 2},
	{"DR3", LBREG, i386.D_DR + 3},
	{"DR4", LBREG, i386.D_DR + 4},
	{"DR5", LBREG, i386.D_DR + 5},
	{"DR6", LBREG, i386.D_DR + 6},
	{"DR7", LBREG, i386.D_DR + 7},
	{"TR0", LBREG, i386.D_TR + 0},
	{"TR1", LBREG, i386.D_TR + 1},
	{"TR2", LBREG, i386.D_TR + 2},
	{"TR3", LBREG, i386.D_TR + 3},
	{"TR4", LBREG, i386.D_TR + 4},
	{"TR5", LBREG, i386.D_TR + 5},
	{"TR6", LBREG, i386.D_TR + 6},
	{"TR7", LBREG, i386.D_TR + 7},
	{"AAA", LTYPE0, i386.AAAA},
	{"AAD", LTYPE0, i386.AAAD},
	{"AAM", LTYPE0, i386.AAAM},
	{"AAS", LTYPE0, i386.AAAS},
	{"ADCB", LTYPE3, i386.AADCB},
	{"ADCL", LTYPE3, i386.AADCL},
	{"ADCW", LTYPE3, i386.AADCW},
	{"ADDB", LTYPE3, i386.AADDB},
	{"ADDL", LTYPE3, i386.AADDL},
	{"ADDW", LTYPE3, i386.AADDW},
	{"ADJSP", LTYPE2, i386.AADJSP},
	{"ANDB", LTYPE3, i386.AANDB},
	{"ANDL", LTYPE3, i386.AANDL},
	{"ANDW", LTYPE3, i386.AANDW},
	{"ARPL", LTYPE3, i386.AARPL},
	{"BOUNDL", LTYPE3, i386.ABOUNDL},
	{"BOUNDW", LTYPE3, i386.ABOUNDW},
	{"BSFL", LTYPE3, i386.ABSFL},
	{"BSFW", LTYPE3, i386.ABSFW},
	{"BSRL", LTYPE3, i386.ABSRL},
	{"BSRW", LTYPE3, i386.ABSRW},
	{"BSWAPL", LTYPE1, i386.ABSWAPL},
	{"BTCL", LTYPE3, i386.ABTCL},
	{"BTCW", LTYPE3, i386.ABTCW},
	{"BTL", LTYPE3, i386.ABTL},
	{"BTRL", LTYPE3, i386.ABTRL},
	{"BTRW", LTYPE3, i386.ABTRW},
	{"BTSL", LTYPE3, i386.ABTSL},
	{"BTSW", LTYPE3, i386.ABTSW},
	{"BTW", LTYPE3, i386.ABTW},
	{"BYTE", LTYPE2, i386.ABYTE},
	{"CALL", LTYPEC, i386.ACALL},
	{"CLC", LTYPE0, i386.ACLC},
	{"CLD", LTYPE0, i386.ACLD},
	{"CLI", LTYPE0, i386.ACLI},
	{"CLTS", LTYPE0, i386.ACLTS},
	{"CMC", LTYPE0, i386.ACMC},
	{"CMPB", LTYPE4, i386.ACMPB},
	{"CMPL", LTYPE4, i386.ACMPL},
	{"CMPW", LTYPE4, i386.ACMPW},
	{"CMPSB", LTYPE0, i386.ACMPSB},
	{"CMPSL", LTYPE0, i386.ACMPSL},
	{"CMPSW", LTYPE0, i386.ACMPSW},
	{"CMPXCHG8B", LTYPE1, i386.ACMPXCHG8B},
	{"CMPXCHGB", LTYPE3, i386.ACMPXCHGB},
	{"CMPXCHGL", LTYPE3, i386.ACMPXCHGL},
	{"CMPXCHGW", LTYPE3, i386.ACMPXCHGW},
	{"CPUID", LTYPE0, i386.ACPUID},
	{"DAA", LTYPE0, i386.ADAA},
	{"DAS", LTYPE0, i386.ADAS},
	{"DATA", LTYPED, i386.ADATA},
	{"DECB", LTYPE1, i386.ADECB},
	{"DECL", LTYPE1, i386.ADECL},
	{"DECW", LTYPE1, i386.ADECW},
	{"DIVB", LTYPE2, i386.ADIVB},
	{"DIVL", LTYPE2, i386.ADIVL},
	{"DIVW", LTYPE2, i386.ADIVW},
	{"END", LTYPE0, i386.AEND},
	{"ENTER", LTYPE2, i386.AENTER},
	{"GLOBL", LTYPEG, i386.AGLOBL},
	{"HLT", LTYPE0, i386.AHLT},
	{"IDIVB", LTYPE2, i386.AIDIVB},
	{"IDIVL", LTYPE2, i386.AIDIVL},
	{"IDIVW", LTYPE2, i386.AIDIVW},
	{"IMULB", LTYPE2, i386.AIMULB},
	{"IMULL", LTYPEI, i386.AIMULL},
	{"IMULW", LTYPEI, i386.AIMULW},
	{"INB", LTYPE0, i386.AINB},
	{"INL", LTYPE0, i386.AINL},
	{"INW", LTYPE0, i386.AINW},
	{"INCB", LTYPE1, i386.AINCB},
	{"INCL", LTYPE1, i386.AINCL},
	{"INCW", LTYPE1, i386.AINCW},
	{"INSB", LTYPE0, i386.AINSB},
	{"INSL", LTYPE0, i386.AINSL},
	{"INSW", LTYPE0, i386.AINSW},
	{"INT", LTYPE2, i386.AINT},
	{"INTO", LTYPE0, i386.AINTO},
	{"IRETL", LTYPE0, i386.AIRETL},
	{"IRETW", LTYPE0, i386.AIRETW},
	{"JOS", LTYPER, i386.AJOS},  /* overflow set (OF = 1) */
	{"JO", LTYPER, i386.AJOS},   /* alternate */
	{"JOC", LTYPER, i386.AJOC},  /* overflow clear (OF = 0) */
	{"JNO", LTYPER, i386.AJOC},  /* alternate */
	{"JCS", LTYPER, i386.AJCS},  /* carry set (CF = 1) */
	{"JB", LTYPER, i386.AJCS},   /* alternate */
	{"JC", LTYPER, i386.AJCS},   /* alternate */
	{"JNAE", LTYPER, i386.AJCS}, /* alternate */
	{"JLO", LTYPER, i386.AJCS},  /* alternate */
	{"JCC", LTYPER, i386.AJCC},  /* carry clear (CF = 0) */
	{"JAE", LTYPER, i386.AJCC},  /* alternate */
	{"JNB", LTYPER, i386.AJCC},  /* alternate */
	{"JNC", LTYPER, i386.AJCC},  /* alternate */
	{"JHS", LTYPER, i386.AJCC},  /* alternate */
	{"JEQ", LTYPER, i386.AJEQ},  /* equal (ZF = 1) */
	{"JE", LTYPER, i386.AJEQ},   /* alternate */
	{"JZ", LTYPER, i386.AJEQ},   /* alternate */
	{"JNE", LTYPER, i386.AJNE},  /* not equal (ZF = 0) */
	{"JNZ", LTYPER, i386.AJNE},  /* alternate */
	{"JLS", LTYPER, i386.AJLS},  /* lower or same (unsigned) (CF = 1 || ZF = 1) */
	{"JBE", LTYPER, i386.AJLS},  /* alternate */
	{"JNA", LTYPER, i386.AJLS},  /* alternate */
	{"JHI", LTYPER, i386.AJHI},  /* higher (unsigned) (CF = 0 && ZF = 0) */
	{"JA", LTYPER, i386.AJHI},   /* alternate */
	{"JNBE", LTYPER, i386.AJHI}, /* alternate */
	{"JMI", LTYPER, i386.AJMI},  /* negative (minus) (SF = 1) */
	{"JS", LTYPER, i386.AJMI},   /* alternate */
	{"JPL", LTYPER, i386.AJPL},  /* non-negative (plus) (SF = 0) */
	{"JNS", LTYPER, i386.AJPL},  /* alternate */
	{"JPS", LTYPER, i386.AJPS},  /* parity set (PF = 1) */
	{"JP", LTYPER, i386.AJPS},   /* alternate */
	{"JPE", LTYPER, i386.AJPS},  /* alternate */
	{"JPC", LTYPER, i386.AJPC},  /* parity clear (PF = 0) */
	{"JNP", LTYPER, i386.AJPC},  /* alternate */
	{"JPO", LTYPER, i386.AJPC},  /* alternate */
	{"JLT", LTYPER, i386.AJLT},  /* less than (signed) (SF != OF) */
	{"JL", LTYPER, i386.AJLT},   /* alternate */
	{"JNGE", LTYPER, i386.AJLT}, /* alternate */
	{"JGE", LTYPER, i386.AJGE},  /* greater than or equal (signed) (SF = OF) */
	{"JNL", LTYPER, i386.AJGE},  /* alternate */
	{"JLE", LTYPER, i386.AJLE},  /* less than or equal (signed) (ZF = 1 || SF != OF) */
	{"JNG", LTYPER, i386.AJLE},  /* alternate */
	{"JGT", LTYPER, i386.AJGT},  /* greater than (signed) (ZF = 0 && SF = OF) */
	{"JG", LTYPER, i386.AJGT},   /* alternate */
	{"JNLE", LTYPER, i386.AJGT}, /* alternate */
	{"JCXZL", LTYPER, i386.AJCXZL},
	{"JCXZW", LTYPER, i386.AJCXZW},
	{"JMP", LTYPEC, i386.AJMP},
	{"LAHF", LTYPE0, i386.ALAHF},
	{"LARL", LTYPE3, i386.ALARL},
	{"LARW", LTYPE3, i386.ALARW},
	{"LEAL", LTYPE3, i386.ALEAL},
	{"LEAW", LTYPE3, i386.ALEAW},
	{"LEAVEL", LTYPE0, i386.ALEAVEL},
	{"LEAVEW", LTYPE0, i386.ALEAVEW},
	{"LOCK", LTYPE0, i386.ALOCK},
	{"LODSB", LTYPE0, i386.ALODSB},
	{"LODSL", LTYPE0, i386.ALODSL},
	{"LODSW", LTYPE0, i386.ALODSW},
	{"LONG", LTYPE2, i386.ALONG},
	{"LOOP", LTYPER, i386.ALOOP},
	{"LOOPEQ", LTYPER, i386.ALOOPEQ},
	{"LOOPNE", LTYPER, i386.ALOOPNE},
	{"LSLL", LTYPE3, i386.ALSLL},
	{"LSLW", LTYPE3, i386.ALSLW},
	{"MOVB", LTYPE3, i386.AMOVB},
	{"MOVL", LTYPEM, i386.AMOVL},
	{"MOVW", LTYPEM, i386.AMOVW},
	{"MOVQ", LTYPEM, i386.AMOVQ},
	{"MOVBLSX", LTYPE3, i386.AMOVBLSX},
	{"MOVBLZX", LTYPE3, i386.AMOVBLZX},
	{"MOVBWSX", LTYPE3, i386.AMOVBWSX},
	{"MOVBWZX", LTYPE3, i386.AMOVBWZX},
	{"MOVWLSX", LTYPE3, i386.AMOVWLSX},
	{"MOVWLZX", LTYPE3, i386.AMOVWLZX},
	{"MOVSB", LTYPE0, i386.AMOVSB},
	{"MOVSL", LTYPE0, i386.AMOVSL},
	{"MOVSW", LTYPE0, i386.AMOVSW},
	{"MULB", LTYPE2, i386.AMULB},
	{"MULL", LTYPE2, i386.AMULL},
	{"MULW", LTYPE2, i386.AMULW},
	{"NEGB", LTYPE1, i386.ANEGB},
	{"NEGL", LTYPE1, i386.ANEGL},
	{"NEGW", LTYPE1, i386.ANEGW},
	{"NOP", LTYPEN, i386.ANOP},
	{"NOTB", LTYPE1, i386.ANOTB},
	{"NOTL", LTYPE1, i386.ANOTL},
	{"NOTW", LTYPE1, i386.ANOTW},
	{"ORB", LTYPE3, i386.AORB},
	{"ORL", LTYPE3, i386.AORL},
	{"ORW", LTYPE3, i386.AORW},
	{"OUTB", LTYPE0, i386.AOUTB},
	{"OUTL", LTYPE0, i386.AOUTL},
	{"OUTW", LTYPE0, i386.AOUTW},
	{"OUTSB", LTYPE0, i386.AOUTSB},
	{"OUTSL", LTYPE0, i386.AOUTSL},
	{"OUTSW", LTYPE0, i386.AOUTSW},
	{"PAUSE", LTYPEN, i386.APAUSE},
	{"PINSRD", LTYPEX, i386.APINSRD},
	{"POPAL", LTYPE0, i386.APOPAL},
	{"POPAW", LTYPE0, i386.APOPAW},
	{"POPFL", LTYPE0, i386.APOPFL},
	{"POPFW", LTYPE0, i386.APOPFW},
	{"POPL", LTYPE1, i386.APOPL},
	{"POPW", LTYPE1, i386.APOPW},
	{"PUSHAL", LTYPE0, i386.APUSHAL},
	{"PUSHAW", LTYPE0, i386.APUSHAW},
	{"PUSHFL", LTYPE0, i386.APUSHFL},
	{"PUSHFW", LTYPE0, i386.APUSHFW},
	{"PUSHL", LTYPE2, i386.APUSHL},
	{"PUSHW", LTYPE2, i386.APUSHW},
	{"RCLB", LTYPE3, i386.ARCLB},
	{"RCLL", LTYPE3, i386.ARCLL},
	{"RCLW", LTYPE3, i386.ARCLW},
	{"RCRB", LTYPE3, i386.ARCRB},
	{"RCRL", LTYPE3, i386.ARCRL},
	{"RCRW", LTYPE3, i386.ARCRW},
	{"RDTSC", LTYPE0, i386.ARDTSC},
	{"REP", LTYPE0, i386.AREP},
	{"REPN", LTYPE0, i386.AREPN},
	{"RET", LTYPE0, i386.ARET},
	{"ROLB", LTYPE3, i386.AROLB},
	{"ROLL", LTYPE3, i386.AROLL},
	{"ROLW", LTYPE3, i386.AROLW},
	{"RORB", LTYPE3, i386.ARORB},
	{"RORL", LTYPE3, i386.ARORL},
	{"RORW", LTYPE3, i386.ARORW},
	{"SAHF", LTYPE0, i386.ASAHF},
	{"SALB", LTYPE3, i386.ASALB},
	{"SALL", LTYPE3, i386.ASALL},
	{"SALW", LTYPE3, i386.ASALW},
	{"SARB", LTYPE3, i386.ASARB},
	{"SARL", LTYPE3, i386.ASARL},
	{"SARW", LTYPE3, i386.ASARW},
	{"SBBB", LTYPE3, i386.ASBBB},
	{"SBBL", LTYPE3, i386.ASBBL},
	{"SBBW", LTYPE3, i386.ASBBW},
	{"SCASB", LTYPE0, i386.ASCASB},
	{"SCASL", LTYPE0, i386.ASCASL},
	{"SCASW", LTYPE0, i386.ASCASW},
	{"SETCC", LTYPE1, i386.ASETCC}, /* see JCC etc above for condition codes */
	{"SETCS", LTYPE1, i386.ASETCS},
	{"SETEQ", LTYPE1, i386.ASETEQ},
	{"SETGE", LTYPE1, i386.ASETGE},
	{"SETGT", LTYPE1, i386.ASETGT},
	{"SETHI", LTYPE1, i386.ASETHI},
	{"SETLE", LTYPE1, i386.ASETLE},
	{"SETLS", LTYPE1, i386.ASETLS},
	{"SETLT", LTYPE1, i386.ASETLT},
	{"SETMI", LTYPE1, i386.ASETMI},
	{"SETNE", LTYPE1, i386.ASETNE},
	{"SETOC", LTYPE1, i386.ASETOC},
	{"SETOS", LTYPE1, i386.ASETOS},
	{"SETPC", LTYPE1, i386.ASETPC},
	{"SETPL", LTYPE1, i386.ASETPL},
	{"SETPS", LTYPE1, i386.ASETPS},
	{"CDQ", LTYPE0, i386.ACDQ},
	{"CWD", LTYPE0, i386.ACWD},
	{"SHLB", LTYPE3, i386.ASHLB},
	{"SHLL", LTYPES, i386.ASHLL},
	{"SHLW", LTYPES, i386.ASHLW},
	{"SHRB", LTYPE3, i386.ASHRB},
	{"SHRL", LTYPES, i386.ASHRL},
	{"SHRW", LTYPES, i386.ASHRW},
	{"STC", LTYPE0, i386.ASTC},
	{"STD", LTYPE0, i386.ASTD},
	{"STI", LTYPE0, i386.ASTI},
	{"STOSB", LTYPE0, i386.ASTOSB},
	{"STOSL", LTYPE0, i386.ASTOSL},
	{"STOSW", LTYPE0, i386.ASTOSW},
	{"SUBB", LTYPE3, i386.ASUBB},
	{"SUBL", LTYPE3, i386.ASUBL},
	{"SUBW", LTYPE3, i386.ASUBW},
	{"SYSCALL", LTYPE0, i386.ASYSCALL},
	{"TESTB", LTYPE3, i386.ATESTB},
	{"TESTL", LTYPE3, i386.ATESTL},
	{"TESTW", LTYPE3, i386.ATESTW},
	{"TEXT", LTYPET, i386.ATEXT},
	{"VERR", LTYPE2, i386.AVERR},
	{"VERW", LTYPE2, i386.AVERW},
	{"WAIT", LTYPE0, i386.AWAIT},
	{"WORD", LTYPE2, i386.AWORD},
	{"XADDB", LTYPE3, i386.AXADDB},
	{"XADDL", LTYPE3, i386.AXADDL},
	{"XADDW", LTYPE3, i386.AXADDW},
	{"XCHGB", LTYPE3, i386.AXCHGB},
	{"XCHGL", LTYPE3, i386.AXCHGL},
	{"XCHGW", LTYPE3, i386.AXCHGW},
	{"XLAT", LTYPE2, i386.AXLAT},
	{"XORB", LTYPE3, i386.AXORB},
	{"XORL", LTYPE3, i386.AXORL},
	{"XORW", LTYPE3, i386.AXORW},
	{"CMOVLCC", LTYPE3, i386.ACMOVLCC},
	{"CMOVLCS", LTYPE3, i386.ACMOVLCS},
	{"CMOVLEQ", LTYPE3, i386.ACMOVLEQ},
	{"CMOVLGE", LTYPE3, i386.ACMOVLGE},
	{"CMOVLGT", LTYPE3, i386.ACMOVLGT},
	{"CMOVLHI", LTYPE3, i386.ACMOVLHI},
	{"CMOVLLE", LTYPE3, i386.ACMOVLLE},
	{"CMOVLLS", LTYPE3, i386.ACMOVLLS},
	{"CMOVLLT", LTYPE3, i386.ACMOVLLT},
	{"CMOVLMI", LTYPE3, i386.ACMOVLMI},
	{"CMOVLNE", LTYPE3, i386.ACMOVLNE},
	{"CMOVLOC", LTYPE3, i386.ACMOVLOC},
	{"CMOVLOS", LTYPE3, i386.ACMOVLOS},
	{"CMOVLPC", LTYPE3, i386.ACMOVLPC},
	{"CMOVLPL", LTYPE3, i386.ACMOVLPL},
	{"CMOVLPS", LTYPE3, i386.ACMOVLPS},
	{"CMOVWCC", LTYPE3, i386.ACMOVWCC},
	{"CMOVWCS", LTYPE3, i386.ACMOVWCS},
	{"CMOVWEQ", LTYPE3, i386.ACMOVWEQ},
	{"CMOVWGE", LTYPE3, i386.ACMOVWGE},
	{"CMOVWGT", LTYPE3, i386.ACMOVWGT},
	{"CMOVWHI", LTYPE3, i386.ACMOVWHI},
	{"CMOVWLE", LTYPE3, i386.ACMOVWLE},
	{"CMOVWLS", LTYPE3, i386.ACMOVWLS},
	{"CMOVWLT", LTYPE3, i386.ACMOVWLT},
	{"CMOVWMI", LTYPE3, i386.ACMOVWMI},
	{"CMOVWNE", LTYPE3, i386.ACMOVWNE},
	{"CMOVWOC", LTYPE3, i386.ACMOVWOC},
	{"CMOVWOS", LTYPE3, i386.ACMOVWOS},
	{"CMOVWPC", LTYPE3, i386.ACMOVWPC},
	{"CMOVWPL", LTYPE3, i386.ACMOVWPL},
	{"CMOVWPS", LTYPE3, i386.ACMOVWPS},
	{"FMOVB", LTYPE3, i386.AFMOVB},
	{"FMOVBP", LTYPE3, i386.AFMOVBP},
	{"FMOVD", LTYPE3, i386.AFMOVD},
	{"FMOVDP", LTYPE3, i386.AFMOVDP},
	{"FMOVF", LTYPE3, i386.AFMOVF},
	{"FMOVFP", LTYPE3, i386.AFMOVFP},
	{"FMOVL", LTYPE3, i386.AFMOVL},
	{"FMOVLP", LTYPE3, i386.AFMOVLP},
	{"FMOVV", LTYPE3, i386.AFMOVV},
	{"FMOVVP", LTYPE3, i386.AFMOVVP},
	{"FMOVW", LTYPE3, i386.AFMOVW},
	{"FMOVWP", LTYPE3, i386.AFMOVWP},
	{"FMOVX", LTYPE3, i386.AFMOVX},
	{"FMOVXP", LTYPE3, i386.AFMOVXP},
	{"FCMOVCC", LTYPE3, i386.AFCMOVCC},
	{"FCMOVCS", LTYPE3, i386.AFCMOVCS},
	{"FCMOVEQ", LTYPE3, i386.AFCMOVEQ},
	{"FCMOVHI", LTYPE3, i386.AFCMOVHI},
	{"FCMOVLS", LTYPE3, i386.AFCMOVLS},
	{"FCMOVNE", LTYPE3, i386.AFCMOVNE},
	{"FCMOVNU", LTYPE3, i386.AFCMOVNU},
	{"FCMOVUN", LTYPE3, i386.AFCMOVUN},
	{"FCOMB", LTYPE3, i386.AFCOMB},
	{"FCOMBP", LTYPE3, i386.AFCOMBP},
	{"FCOMD", LTYPE3, i386.AFCOMD},
	{"FCOMDP", LTYPE3, i386.AFCOMDP},
	{"FCOMDPP", LTYPE3, i386.AFCOMDPP},
	{"FCOMF", LTYPE3, i386.AFCOMF},
	{"FCOMFP", LTYPE3, i386.AFCOMFP},
	{"FCOMI", LTYPE3, i386.AFCOMI},
	{"FCOMIP", LTYPE3, i386.AFCOMIP},
	{"FCOML", LTYPE3, i386.AFCOML},
	{"FCOMLP", LTYPE3, i386.AFCOMLP},
	{"FCOMW", LTYPE3, i386.AFCOMW},
	{"FCOMWP", LTYPE3, i386.AFCOMWP},
	{"FUCOM", LTYPE3, i386.AFUCOM},
	{"FUCOMI", LTYPE3, i386.AFUCOMI},
	{"FUCOMIP", LTYPE3, i386.AFUCOMIP},
	{"FUCOMP", LTYPE3, i386.AFUCOMP},
	{"FUCOMPP", LTYPE3, i386.AFUCOMPP},
	{"FADDW", LTYPE3, i386.AFADDW},
	{"FADDL", LTYPE3, i386.AFADDL},
	{"FADDF", LTYPE3, i386.AFADDF},
	{"FADDD", LTYPE3, i386.AFADDD},
	{"FADDDP", LTYPE3, i386.AFADDDP},
	{"FSUBDP", LTYPE3, i386.AFSUBDP},
	{"FSUBW", LTYPE3, i386.AFSUBW},
	{"FSUBL", LTYPE3, i386.AFSUBL},
	{"FSUBF", LTYPE3, i386.AFSUBF},
	{"FSUBD", LTYPE3, i386.AFSUBD},
	{"FSUBRDP", LTYPE3, i386.AFSUBRDP},
	{"FSUBRW", LTYPE3, i386.AFSUBRW},
	{"FSUBRL", LTYPE3, i386.AFSUBRL},
	{"FSUBRF", LTYPE3, i386.AFSUBRF},
	{"FSUBRD", LTYPE3, i386.AFSUBRD},
	{"FMULDP", LTYPE3, i386.AFMULDP},
	{"FMULW", LTYPE3, i386.AFMULW},
	{"FMULL", LTYPE3, i386.AFMULL},
	{"FMULF", LTYPE3, i386.AFMULF},
	{"FMULD", LTYPE3, i386.AFMULD},
	{"FDIVDP", LTYPE3, i386.AFDIVDP},
	{"FDIVW", LTYPE3, i386.AFDIVW},
	{"FDIVL", LTYPE3, i386.AFDIVL},
	{"FDIVF", LTYPE3, i386.AFDIVF},
	{"FDIVD", LTYPE3, i386.AFDIVD},
	{"FDIVRDP", LTYPE3, i386.AFDIVRDP},
	{"FDIVRW", LTYPE3, i386.AFDIVRW},
	{"FDIVRL", LTYPE3, i386.AFDIVRL},
	{"FDIVRF", LTYPE3, i386.AFDIVRF},
	{"FDIVRD", LTYPE3, i386.AFDIVRD},
	{"FXCHD", LTYPE3, i386.AFXCHD},
	{"FFREE", LTYPE1, i386.AFFREE},
	{"FLDCW", LTYPE2, i386.AFLDCW},
	{"FLDENV", LTYPE1, i386.AFLDENV},
	{"FRSTOR", LTYPE2, i386.AFRSTOR},
	{"FSAVE", LTYPE1, i386.AFSAVE},
	{"FSTCW", LTYPE1, i386.AFSTCW},
	{"FSTENV", LTYPE1, i386.AFSTENV},
	{"FSTSW", LTYPE1, i386.AFSTSW},
	{"F2XM1", LTYPE0, i386.AF2XM1},
	{"FABS", LTYPE0, i386.AFABS},
	{"FCHS", LTYPE0, i386.AFCHS},
	{"FCLEX", LTYPE0, i386.AFCLEX},
	{"FCOS", LTYPE0, i386.AFCOS},
	{"FDECSTP", LTYPE0, i386.AFDECSTP},
	{"FINCSTP", LTYPE0, i386.AFINCSTP},
	{"FINIT", LTYPE0, i386.AFINIT},
	{"FLD1", LTYPE0, i386.AFLD1},
	{"FLDL2E", LTYPE0, i386.AFLDL2E},
	{"FLDL2T", LTYPE0, i386.AFLDL2T},
	{"FLDLG2", LTYPE0, i386.AFLDLG2},
	{"FLDLN2", LTYPE0, i386.AFLDLN2},
	{"FLDPI", LTYPE0, i386.AFLDPI},
	{"FLDZ", LTYPE0, i386.AFLDZ},
	{"FNOP", LTYPE0, i386.AFNOP},
	{"FPATAN", LTYPE0, i386.AFPATAN},
	{"FPREM", LTYPE0, i386.AFPREM},
	{"FPREM1", LTYPE0, i386.AFPREM1},
	{"FPTAN", LTYPE0, i386.AFPTAN},
	{"FRNDINT", LTYPE0, i386.AFRNDINT},
	{"FSCALE", LTYPE0, i386.AFSCALE},
	{"FSIN", LTYPE0, i386.AFSIN},
	{"FSINCOS", LTYPE0, i386.AFSINCOS},
	{"FSQRT", LTYPE0, i386.AFSQRT},
	{"FTST", LTYPE0, i386.AFTST},
	{"FXAM", LTYPE0, i386.AFXAM},
	{"FXTRACT", LTYPE0, i386.AFXTRACT},
	{"FYL2X", LTYPE0, i386.AFYL2X},
	{"FYL2XP1", LTYPE0, i386.AFYL2XP1},
	{"LFENCE", LTYPE0, i386.ALFENCE},
	{"MFENCE", LTYPE0, i386.AMFENCE},
	{"SFENCE", LTYPE0, i386.ASFENCE},
	{"EMMS", LTYPE0, i386.AEMMS},
	{"PREFETCHT0", LTYPE2, i386.APREFETCHT0},
	{"PREFETCHT1", LTYPE2, i386.APREFETCHT1},
	{"PREFETCHT2", LTYPE2, i386.APREFETCHT2},
	{"PREFETCHNTA", LTYPE2, i386.APREFETCHNTA},
	{"UNDEF", LTYPE0, i386.AUNDEF},
	{"ADDPD", LTYPE3, i386.AADDPD},
	{"ADDPS", LTYPE3, i386.AADDPS},
	{"ADDSD", LTYPE3, i386.AADDSD},
	{"ADDSS", LTYPE3, i386.AADDSS},
	{"AESENC", LTYPE3, i386.AAESENC},
	{"ANDNPD", LTYPE3, i386.AANDNPD},
	{"ANDNPS", LTYPE3, i386.AANDNPS},
	{"ANDPD", LTYPE3, i386.AANDPD},
	{"ANDPS", LTYPE3, i386.AANDPS},
	{"CMPPD", LTYPEXC, i386.ACMPPD},
	{"CMPPS", LTYPEXC, i386.ACMPPS},
	{"CMPSD", LTYPEXC, i386.ACMPSD},
	{"CMPSS", LTYPEXC, i386.ACMPSS},
	{"COMISD", LTYPE3, i386.ACOMISD},
	{"COMISS", LTYPE3, i386.ACOMISS},
	{"CVTPL2PD", LTYPE3, i386.ACVTPL2PD},
	{"CVTPL2PS", LTYPE3, i386.ACVTPL2PS},
	{"CVTPD2PL", LTYPE3, i386.ACVTPD2PL},
	{"CVTPD2PS", LTYPE3, i386.ACVTPD2PS},
	{"CVTPS2PL", LTYPE3, i386.ACVTPS2PL},
	{"CVTPS2PD", LTYPE3, i386.ACVTPS2PD},
	{"CVTSD2SL", LTYPE3, i386.ACVTSD2SL},
	{"CVTSD2SS", LTYPE3, i386.ACVTSD2SS},
	{"CVTSL2SD", LTYPE3, i386.ACVTSL2SD},
	{"CVTSL2SS", LTYPE3, i386.ACVTSL2SS},
	{"CVTSS2SD", LTYPE3, i386.ACVTSS2SD},
	{"CVTSS2SL", LTYPE3, i386.ACVTSS2SL},
	{"CVTTPD2PL", LTYPE3, i386.ACVTTPD2PL},
	{"CVTTPS2PL", LTYPE3, i386.ACVTTPS2PL},
	{"CVTTSD2SL", LTYPE3, i386.ACVTTSD2SL},
	{"CVTTSS2SL", LTYPE3, i386.ACVTTSS2SL},
	{"DIVPD", LTYPE3, i386.ADIVPD},
	{"DIVPS", LTYPE3, i386.ADIVPS},
	{"DIVSD", LTYPE3, i386.ADIVSD},
	{"DIVSS", LTYPE3, i386.ADIVSS},
	{"MASKMOVOU", LTYPE3, i386.AMASKMOVOU},
	{"MASKMOVDQU", LTYPE3, i386.AMASKMOVOU}, /* syn */
	{"MAXPD", LTYPE3, i386.AMAXPD},
	{"MAXPS", LTYPE3, i386.AMAXPS},
	{"MAXSD", LTYPE3, i386.AMAXSD},
	{"MAXSS", LTYPE3, i386.AMAXSS},
	{"MINPD", LTYPE3, i386.AMINPD},
	{"MINPS", LTYPE3, i386.AMINPS},
	{"MINSD", LTYPE3, i386.AMINSD},
	{"MINSS", LTYPE3, i386.AMINSS},
	{"MOVAPD", LTYPE3, i386.AMOVAPD},
	{"MOVAPS", LTYPE3, i386.AMOVAPS},
	{"MOVO", LTYPE3, i386.AMOVO},
	{"MOVOA", LTYPE3, i386.AMOVO}, /* syn */
	{"MOVOU", LTYPE3, i386.AMOVOU},
	{"MOVHLPS", LTYPE3, i386.AMOVHLPS},
	{"MOVHPD", LTYPE3, i386.AMOVHPD},
	{"MOVHPS", LTYPE3, i386.AMOVHPS},
	{"MOVLHPS", LTYPE3, i386.AMOVLHPS},
	{"MOVLPD", LTYPE3, i386.AMOVLPD},
	{"MOVLPS", LTYPE3, i386.AMOVLPS},
	{"MOVMSKPD", LTYPE3, i386.AMOVMSKPD},
	{"MOVMSKPS", LTYPE3, i386.AMOVMSKPS},
	{"MOVNTO", LTYPE3, i386.AMOVNTO},
	{"MOVNTDQ", LTYPE3, i386.AMOVNTO}, /* syn */
	{"MOVNTPD", LTYPE3, i386.AMOVNTPD},
	{"MOVNTPS", LTYPE3, i386.AMOVNTPS},
	{"MOVSD", LTYPE3, i386.AMOVSD},
	{"MOVSS", LTYPE3, i386.AMOVSS},
	{"MOVUPD", LTYPE3, i386.AMOVUPD},
	{"MOVUPS", LTYPE3, i386.AMOVUPS},
	{"MULPD", LTYPE3, i386.AMULPD},
	{"MULPS", LTYPE3, i386.AMULPS},
	{"MULSD", LTYPE3, i386.AMULSD},
	{"MULSS", LTYPE3, i386.AMULSS},
	{"ORPD", LTYPE3, i386.AORPD},
	{"ORPS", LTYPE3, i386.AORPS},
	{"PADDQ", LTYPE3, i386.APADDQ},
	{"PAND", LTYPE3, i386.APAND},
	{"PCMPEQB", LTYPE3, i386.APCMPEQB},
	{"PMAXSW", LTYPE3, i386.APMAXSW},
	{"PMAXUB", LTYPE3, i386.APMAXUB},
	{"PMINSW", LTYPE3, i386.APMINSW},
	{"PMINUB", LTYPE3, i386.APMINUB},
	{"PMOVMSKB", LTYPE3, i386.APMOVMSKB},
	{"PSADBW", LTYPE3, i386.APSADBW},
	{"PSHUFB", LTYPE3, i386.APSHUFB},
	{"PSHUFHW", LTYPEX, i386.APSHUFHW},
	{"PSHUFL", LTYPEX, i386.APSHUFL},
	{"PSHUFLW", LTYPEX, i386.APSHUFLW},
	{"PSUBB", LTYPE3, i386.APSUBB},
	{"PSUBL", LTYPE3, i386.APSUBL},
	{"PSUBQ", LTYPE3, i386.APSUBQ},
	{"PSUBSB", LTYPE3, i386.APSUBSB},
	{"PSUBSW", LTYPE3, i386.APSUBSW},
	{"PSUBUSB", LTYPE3, i386.APSUBUSB},
	{"PSUBUSW", LTYPE3, i386.APSUBUSW},
	{"PSUBW", LTYPE3, i386.APSUBW},
	{"PUNPCKHQDQ", LTYPE3, i386.APUNPCKHQDQ},
	{"PUNPCKLQDQ", LTYPE3, i386.APUNPCKLQDQ},
	{"PXOR", LTYPE3, i386.APXOR},
	{"RCPPS", LTYPE3, i386.ARCPPS},
	{"RCPSS", LTYPE3, i386.ARCPSS},
	{"RSQRTPS", LTYPE3, i386.ARSQRTPS},
	{"RSQRTSS", LTYPE3, i386.ARSQRTSS},
	{"SQRTPD", LTYPE3, i386.ASQRTPD},
	{"SQRTPS", LTYPE3, i386.ASQRTPS},
	{"SQRTSD", LTYPE3, i386.ASQRTSD},
	{"SQRTSS", LTYPE3, i386.ASQRTSS},
	{"SUBPD", LTYPE3, i386.ASUBPD},
	{"SUBPS", LTYPE3, i386.ASUBPS},
	{"SUBSD", LTYPE3, i386.ASUBSD},
	{"SUBSS", LTYPE3, i386.ASUBSS},
	{"UCOMISD", LTYPE3, i386.AUCOMISD},
	{"UCOMISS", LTYPE3, i386.AUCOMISS},
	{"UNPCKHPD", LTYPE3, i386.AUNPCKHPD},
	{"UNPCKHPS", LTYPE3, i386.AUNPCKHPS},
	{"UNPCKLPD", LTYPE3, i386.AUNPCKLPD},
	{"UNPCKLPS", LTYPE3, i386.AUNPCKLPS},
	{"XORPD", LTYPE3, i386.AXORPD},
	{"XORPS", LTYPE3, i386.AXORPS},
	{"USEFIELD", LTYPEN, i386.AUSEFIELD},
	{"PCDATA", LTYPEPC, i386.APCDATA},
	{"FUNCDATA", LTYPEF, i386.AFUNCDATA},
}

func cinit() {
	var s *Sym
	var i int

	nullgen.Type_ = i386.D_NONE
	nullgen.Index = i386.D_NONE

	nerrors = 0
	iostack = nil
	iofree = nil
	peekc = IGN
	nhunk = 0
	for i = 0; i < NHASH; i++ {
		hash[i] = nil
	}
	for i = 0; itab[i].name != ""; i++ {
		s = slookup(itab[i].name)
		if s.type_ != LNAME {
			yyerror("double initialization %s", itab[i].name)
		}
		s.type_ = itab[i].type_
		s.value = int32(itab[i].value)
	}
}

func checkscale(scale int) {
	switch scale {
	case 1,
		2,
		4,
		8:
		return
	}

	yyerror("scale must be 1248: %d", scale)
}

func syminit(s *Sym) {
	s.type_ = LNAME
	s.value = 0
}

func cclean() {
	var g2 Addr2

	g2.from = nullgen
	g2.to = nullgen
	outcode(i386.AEND, &g2)
}

var lastpc *obj.Prog

func outcode(a int, g2 *Addr2) {
	var p *obj.Prog
	var pl *obj.Plist

	if pass == 1 {
		goto out
	}

	p = new(obj.Prog)
	*p = obj.Prog{}
	p.As = int16(a)
	p.Lineno = stmtline
	p.From = g2.from
	p.To = g2.to
	p.Pc = int64(pc)

	if lastpc == nil {
		pl = obj.Linknewplist(ctxt)
		pl.Firstpc = p
	} else {

		lastpc.Link = p
	}
	lastpc = p

out:
	if a != i386.AGLOBL && a != i386.ADATA {
		pc++
	}
}
