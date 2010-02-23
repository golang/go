/*
Derived from Inferno's utils/iyacc/yacc.c
http://code.google.com/p/inferno-os/source/browse/utils/iyacc/yacc.c

This copyright NOTICE applies to all files in this directory and
subdirectories, unless another copyright notice appears in a given
file or subdirectory.  If you take substantial code from this software to use in
other programs, you must somehow include with it an appropriate
copyright notice that includes the copyright notice and the other
notices below.  It is fine (and often tidier) to do that in a separate
file such as NOTICE, LICENCE or COPYING.

	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
	Portions Copyright © 1997-1999 Vita Nuova Limited
	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
	Portions Copyright © 2004,2006 Bruce Ellis
	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
	Portions Copyright © 2009 The Go Authors.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

package main

// yacc
// major difference is lack of stem ("y" variable)
//

import (
	"flag"
	"fmt"
	"bufio"
	"os"
)

// the following are adjustable
// according to memory size
const (
	ACTSIZE  = 30000
	NSTATES  = 2000
	TEMPSIZE = 2000

	SYMINC   = 50  // increase for non-term or term
	RULEINC  = 50  // increase for max rule length prodptr[i]
	PRODINC  = 100 // increase for productions     prodptr
	WSETINC  = 50  // increase for working sets    wsets
	STATEINC = 200 // increase for states          statemem

	NAMESIZE = 50
	NTYPES   = 63
	ISIZE    = 400

	PRIVATE = 0xE000 // unicode private use

	// relationships which must hold:
	//	TEMPSIZE >= NTERMS + NNONTERM + 1;
	//	TEMPSIZE >= NSTATES;
	//

	NTBASE     = 010000
	ERRCODE    = 8190
	ACCEPTCODE = 8191
	YYLEXUNK   = 3
	TOKSTART   = 4 //index of first defined token
)

// no, left, right, binary assoc.
const (
	NOASC = iota
	LASC
	RASC
	BASC
)

// flags for state generation
const (
	DONE = iota
	MUSTDO
	MUSTLOOKAHEAD
)

// flags for a rule having an action, and being reduced
const (
	ACTFLAG = 1 << (iota + 2)
	REDFLAG
)

// output parser flags
const YYFLAG = -1000

// parse tokens
const (
	IDENTIFIER = PRIVATE + iota
	MARK
	TERM
	LEFT
	RIGHT
	BINARY
	PREC
	LCURLY
	IDENTCOLON
	NUMBER
	START
	TYPEDEF
	TYPENAME
	UNION
)

const ENDFILE = 0
const EMPTY = 1
const WHOKNOWS = 0
const OK = 1
const NOMORE = -1000

// macros for getting associativity and precedence levels
func ASSOC(i int) int { return i & 3 }

func PLEVEL(i int) int { return (i >> 4) & 077 }

func TYPE(i int) int { return (i >> 10) & 077 }

// macros for setting associativity and precedence levels
func SETASC(i, j int) int { return i | j }

func SETPLEV(i, j int) int { return i | (j << 4) }

func SETTYPE(i, j int) int { return i | (j << 10) }

// I/O descriptors
var finput *bufio.Reader // input file
var stderr *bufio.Writer
var ftable *bufio.Writer  // y.go file
var foutput *bufio.Writer // y.output file

var oflag string // -o [y.go]		- y.go file
var vflag string // -v [y.output]	- y.output file
var lflag bool   // -l			- disable line directives

var stacksize = 200

// communication variables between various I/O routines
var infile string  // input file name
var numbval int    // value of an input number
var tokname string // input token name, slop for runes and 0
var tokflag = false

// structure declarations
type Lkset []int

type Pitem struct {
	prod   []int
	off    int // offset within the production
	first  int // first term or non-term in item
	prodno int // production number for sorting
}

type Item struct {
	pitem Pitem
	look  Lkset
}

type Symb struct {
	name  string
	value int
}

type Wset struct {
	pitem Pitem
	flag  int
	ws    Lkset
}

// storage of types
var ntypes int             // number of types defined
var typeset [NTYPES]string // pointers to type tags

// token information

var ntokens = 0 // number of tokens
var tokset []Symb
var toklev []int // vector with the precedence of the terminals

// nonterminal information

var nnonter = -1 // the number of nonterminals
var nontrst []Symb
var start int // start symbol

// state information

var nstate = 0                      // number of states
var pstate = make([]int, NSTATES+2) // index into statemem to the descriptions of the states
var statemem []Item
var tystate = make([]int, NSTATES) // contains type information about the states
var tstates []int                  // states generated by terminal gotos
var ntstates []int                 // states generated by nonterminal gotos
var mstates = make([]int, NSTATES) // chain of overflows of term/nonterm generation lists
var lastred int                    // number of last reduction of a state
var defact = make([]int, NSTATES)  // default actions of states

// lookahead set information

var lkst []Lkset
var nolook = 0  // flag to turn off lookahead computations
var tbitset = 0 // size of lookahead sets
var clset Lkset // temporary storage for lookahead computations

// working set information

var wsets []Wset
var cwp int

// storage for action table

var amem []int                   // action table storage
var memp int                     // next free action table position
var indgo = make([]int, NSTATES) // index to the stored goto table

// temporary vector, indexable by states, terms, or ntokens

var temp1 = make([]int, TEMPSIZE) // temporary storage, indexed by terms + ntokens or states
var lineno = 1                    // current input line number
var fatfl = 1                     // if on, error is fatal
var nerrors = 0                   // number of errors

// assigned token type values

var extval = 0

// grammar rule information

var nprod = 1      // number of productions
var prdptr [][]int // pointers to descriptions of productions
var levprd []int   // precedence levels for the productions
var rlines []int   // line number for this rule

// statistics collection variables

var zzgoent = 0
var zzgobest = 0
var zzacent = 0
var zzexcp = 0
var zzclose = 0
var zzrrconf = 0
var zzsrconf = 0
var zzstate = 0

// optimizer arrays

var yypgo [][]int
var optst [][]int
var ggreed []int
var pgo []int

var maxspr int // maximum spread of any entry
var maxoff int // maximum offset into a array
var maxa int

// storage for information about the nonterminals

var pres [][][]int // vector of pointers to productions yielding each nonterminal
var pfirst []Lkset
var pempty []int // vector of nonterminals nontrivially deriving e

// random stuff picked out from between functions

var indebug = 0 // debugging flag for cpfir
var pidebug = 0 // debugging flag for putitem
var gsdebug = 0 // debugging flag for stagen
var cldebug = 0 // debugging flag for closure
var pkdebug = 0 // debugging flag for apack
var g2debug = 0 // debugging for go2gen
var adb = 0     // debugging for callopt

type Resrv struct {
	name  string
	value int
}

var resrv = []Resrv{
	Resrv{"binary", BINARY},
	Resrv{"left", LEFT},
	Resrv{"nonassoc", BINARY},
	Resrv{"prec", PREC},
	Resrv{"right", RIGHT},
	Resrv{"start", START},
	Resrv{"term", TERM},
	Resrv{"token", TERM},
	Resrv{"type", TYPEDEF},
	Resrv{"union", UNION},
	Resrv{"struct", UNION},
}

var zznewstate = 0

const EOF = -1
const UTFmax = 0x3f

func main() {

	setup() // initialize and read productions

	tbitset = (ntokens + 32) / 32
	cpres()  // make table of which productions yield a given nonterminal
	cempty() // make a table of which nonterminals can match the empty string
	cpfir()  // make a table of firsts of nonterminals

	stagen() // generate the states

	yypgo = make([][]int, nnonter+1)
	optst = make([][]int, nstate)
	output() // write the states and the tables
	go2out()

	hideprod()
	summary()

	callopt()

	others()

	exit(0)
}

func setup() {
	var j, ty int

	stderr = bufio.NewWriter(os.NewFile(2, "stderr"))
	foutput = nil

	flag.StringVar(&oflag, "o", "", "parser output")
	flag.StringVar(&vflag, "v", "", "create parsing tables")
	flag.BoolVar(&lflag, "l", false, "disable line directives")

	flag.Parse()
	if flag.NArg() != 1 {
		usage()
	}
	if stacksize < 1 {
		// never set so cannot happen
		fmt.Fprintf(stderr, "yacc: stack size too small\n")
		usage()
	}
	openup()

	defin(0, "$end")
	extval = PRIVATE // tokens start in unicode 'private use'
	defin(0, "error")
	defin(1, "$accept")
	defin(0, "$unk")
	i := 0

	t := gettok()

outer:
	for {
		switch t {
		default:
			error("syntax error tok=%v", t-PRIVATE)

		case MARK, ENDFILE:
			break outer

		case ';':

		case START:
			t = gettok()
			if t != IDENTIFIER {
				error("bad %%start construction")
			}
			start = chfind(1, tokname)

		case TYPEDEF:
			t = gettok()
			if t != TYPENAME {
				error("bad syntax in %%type")
			}
			ty = numbval
			for {
				t = gettok()
				switch t {
				case IDENTIFIER:
					t = chfind(1, tokname)
					if t < NTBASE {
						j = TYPE(toklev[t])
						if j != 0 && j != ty {
							error("type redeclaration of token ",
								tokset[t].name)
						} else {
							toklev[t] = SETTYPE(toklev[t], ty)
						}
					} else {
						j = nontrst[t-NTBASE].value
						if j != 0 && j != ty {
							error("type redeclaration of nonterminal %v",
								nontrst[t-NTBASE].name)
						} else {
							nontrst[t-NTBASE].value = ty
						}
					}
					continue

				case ',':
					continue
				}
				break
			}
			continue

		case UNION:
			cpyunion()

		case LEFT, BINARY, RIGHT, TERM:
			// nonzero means new prec. and assoc.
			lev := t - TERM
			if lev != 0 {
				i++
			}
			ty = 0

			// get identifiers so defined
			t = gettok()

			// there is a type defined
			if t == TYPENAME {
				ty = numbval
				t = gettok()
			}
			for {
				switch t {
				case ',':
					t = gettok()
					continue

				case ';':
					break

				case IDENTIFIER:
					j = chfind(0, tokname)
					if j >= NTBASE {
						error("%v defined earlier as nonterminal", tokname)
					}
					if lev != 0 {
						if ASSOC(toklev[j]) != 0 {
							error("redeclaration of precedence of %v", tokname)
						}
						toklev[j] = SETASC(toklev[j], lev)
						toklev[j] = SETPLEV(toklev[j], i)
					}
					if ty != 0 {
						if TYPE(toklev[j]) != 0 {
							error("redeclaration of type of %v", tokname)
						}
						toklev[j] = SETTYPE(toklev[j], ty)
					}
					t = gettok()
					if t == NUMBER {
						tokset[j].value = numbval
						t = gettok()
					}

					continue
				}
				break
			}
			continue

		case LCURLY:
			cpycode()
		}
		t = gettok()
	}

	if t == ENDFILE {
		error("unexpected EOF before %%")
	}

	// put out non-literal terminals
	for i := TOKSTART; i <= ntokens; i++ {
		// non-literals
		c := tokset[i].name[0]
		if c != ' ' && c != '$' {
			fmt.Fprintf(ftable, "const\t%v\t= %v\n", tokset[i].name, tokset[i].value)
		}
	}

	// put out names of token names
	fmt.Fprintf(ftable, "var\tToknames\t =[]string {\n")
	for i := TOKSTART; i <= ntokens; i++ {
		fmt.Fprintf(ftable, "\t\"%v\",\n", tokset[i].name)
	}
	fmt.Fprintf(ftable, "}\n")

	// put out names of state names
	fmt.Fprintf(ftable, "var\tStatenames\t =[]string {\n")
	//	for i:=TOKSTART; i<=ntokens; i++ {
	//		fmt.Fprintf(ftable, "\t\"%v\",\n", tokset[i].name);
	//	}
	fmt.Fprintf(ftable, "}\n")

	fmt.Fprintf(ftable, "\nfunc\n")
	fmt.Fprintf(ftable, "yyrun(p int, yypt int) {\n")
	fmt.Fprintf(ftable, "switch p {\n")

	moreprod()
	prdptr[0] = []int{NTBASE, start, 1, 0}

	nprod = 1
	curprod := make([]int, RULEINC)
	t = gettok()
	if t != IDENTCOLON {
		error("bad syntax on first rule")
	}

	if start == 0 {
		prdptr[0][1] = chfind(1, tokname)
	}

	// read rules
	// put into prdptr array in the format
	// target
	// followed by id's of terminals and non-terminals
	// followd by -nprod

	for t != MARK && t != ENDFILE {
		mem := 0

		// process a rule
		rlines[nprod] = lineno
		if t == '|' {
			curprod[mem] = prdptr[nprod-1][0]
			mem++
		} else if t == IDENTCOLON {
			curprod[mem] = chfind(1, tokname)
			if curprod[mem] < NTBASE {
				error("token illegal on LHS of grammar rule")
			}
			mem++
		} else {
			error("illegal rule: missing semicolon or | ?")
		}

		// read rule body
		t = gettok()
		for {
			for t == IDENTIFIER {
				curprod[mem] = chfind(1, tokname)
				if curprod[mem] < NTBASE {
					levprd[nprod] = toklev[curprod[mem]]
				}
				mem++
				if mem >= len(curprod) {
					ncurprod := make([]int, mem+RULEINC)
					copy(ncurprod, curprod)
					curprod = ncurprod
				}
				t = gettok()
			}
			if t == PREC {
				if gettok() != IDENTIFIER {
					error("illegal %%prec syntax")
				}
				j = chfind(2, tokname)
				if j >= NTBASE {
					error("nonterminal " + nontrst[j-NTBASE].name + " illegal after %%prec")
				}
				levprd[nprod] = toklev[j]
				t = gettok()
			}
			if t != '=' {
				break
			}
			levprd[nprod] |= ACTFLAG
			fmt.Fprintf(ftable, "\ncase %v:", nprod)
			cpyact(curprod, mem)

			// action within rule...
			t = gettok()
			if t == IDENTIFIER {
				// make it a nonterminal
				j = chfind(1, fmt.Sprintf("$$%v", nprod))

				//
				// the current rule will become rule number nprod+1
				// enter null production for action
				//
				prdptr[nprod] = make([]int, 2)
				prdptr[nprod][0] = j
				prdptr[nprod][1] = -nprod

				// update the production information
				nprod++
				moreprod()
				levprd[nprod] = levprd[nprod-1] & ^ACTFLAG
				levprd[nprod-1] = ACTFLAG
				rlines[nprod] = lineno

				// make the action appear in the original rule
				curprod[mem] = j
				mem++
				if mem >= len(curprod) {
					ncurprod := make([]int, mem+RULEINC)
					copy(ncurprod, curprod)
					curprod = ncurprod
				}
			}
		}

		for t == ';' {
			t = gettok()
		}
		curprod[mem] = -nprod
		mem++

		// check that default action is reasonable
		if ntypes != 0 && (levprd[nprod]&ACTFLAG) == 0 &&
			nontrst[curprod[0]-NTBASE].value != 0 {
			// no explicit action, LHS has value
			tempty := curprod[1]
			if tempty < 0 {
				error("must return a value, since LHS has a type")
			}
			if tempty >= NTBASE {
				tempty = nontrst[tempty-NTBASE].value
			} else {
				tempty = TYPE(toklev[tempty])
			}
			if tempty != nontrst[curprod[0]-NTBASE].value {
				error("default action causes potential type clash")
			}
			fmt.Fprintf(ftable, "\ncase %v:", nprod)
			fmt.Fprintf(ftable, "\n\tYYVAL.%v = YYS[yypt-0].%v;",
				typeset[tempty], typeset[tempty])
		}
		moreprod()
		prdptr[nprod] = make([]int, mem)
		copy(prdptr[nprod], curprod)
		nprod++
		moreprod()
		levprd[nprod] = 0
	}

	//
	// end of all rules
	// dump out the prefix code
	//

	fmt.Fprintf(ftable, "\n\t}")
	fmt.Fprintf(ftable, "\n}\n")

	fmt.Fprintf(ftable, "const	YYEOFCODE	= 1\n")
	fmt.Fprintf(ftable, "const	YYERRCODE	= 2\n")
	fmt.Fprintf(ftable, "const	YYMAXDEPTH	= %v\n", stacksize)

	//
	// copy any postfix code
	//
	if t == MARK {
		if !lflag {
			fmt.Fprintf(ftable, "\n//line %v:%v\n", infile, lineno)
		}
		for {
			c := getrune(finput)
			if c == EOF {
				break
			}
			putrune(ftable, c)
		}
	}
}

//
// allocate enough room to hold another production
//
func moreprod() {
	n := len(prdptr)
	if nprod >= n {
		nn := n + PRODINC
		aprod := make([][]int, nn)
		alevprd := make([]int, nn)
		arlines := make([]int, nn)

		copy(aprod, prdptr)
		copy(alevprd, levprd)
		copy(arlines, rlines)

		prdptr = aprod
		levprd = alevprd
		rlines = arlines
	}
}

//
// define s to be a terminal if t=0
// or a nonterminal if t=1
//
func defin(nt int, s string) int {
	val := 0
	if nt != 0 {
		nnonter++
		if nnonter >= len(nontrst) {
			anontrst := make([]Symb, nnonter+SYMINC)
			copy(anontrst, nontrst)
			nontrst = anontrst
		}
		nontrst[nnonter] = Symb{s, 0}
		return NTBASE + nnonter
	}

	// must be a token
	ntokens++
	if ntokens >= len(tokset) {
		nn := ntokens + SYMINC
		atokset := make([]Symb, nn)
		atoklev := make([]int, nn)

		copy(atoklev, toklev)
		copy(atokset, tokset)

		tokset = atokset
		toklev = atoklev
	}
	tokset[ntokens].name = s
	toklev[ntokens] = 0

	// establish value for token
	// single character literal
	if s[0] == ' ' && len(s) == 1+1 {
		val = int(s[1])
	} else if s[0] == ' ' && s[1] == '\\' { // escape sequence
		if len(s) == 2+1 {
			// single character escape sequence
			switch s[2] {
			case '\'':
				val = '\''
			case '"':
				val = '"'
			case '\\':
				val = '\\'
			case 'a':
				val = '\a'
			case 'b':
				val = '\b'
			case 'n':
				val = '\n'
			case 'r':
				val = '\r'
			case 't':
				val = '\t'
			case 'v':
				val = '\v'
			default:
				error("invalid escape %v", s[1:3])
			}
		} else if s[2] == 'u' && len(s) == 2+1+4 { // \unnnn sequence
			val = 0
			s = s[3:]
			for s != "" {
				c := int(s[0])
				switch {
				case c >= '0' && c <= '9':
					c -= '0'
				case c >= 'a' && c <= 'f':
					c -= 'a' - 10
				case c >= 'A' && c <= 'F':
					c -= 'A' - 10
				default:
					error("illegal \\unnnn construction")
				}
				val = val*16 + c
				s = s[1:]
			}
			if val == 0 {
				error("'\\u0000' is illegal")
			}
		} else {
			error("unknown escape")
		}
	} else {
		val = extval
		extval++
	}

	tokset[ntokens].value = val
	return ntokens
}

var peekline = 0

func gettok() int {
	var i, match, c int

	tokname = ""
	for {
		lineno += peekline
		peekline = 0
		c = getrune(finput)
		for c == ' ' || c == '\n' || c == '\t' || c == '\v' || c == '\r' {
			if c == '\n' {
				lineno++
			}
			c = getrune(finput)
		}

		// skip comment -- fix
		if c != '/' {
			break
		}
		lineno += skipcom()
	}

	switch c {
	case EOF:
		if tokflag {
			fmt.Printf(">>> ENDFILE %v\n", lineno)
		}
		return ENDFILE

	case '{':
		ungetrune(finput, c)
		if tokflag {
			fmt.Printf(">>> ={ %v\n", lineno)
		}
		return '='

	case '<':
		// get, and look up, a type name (union member name)
		c = getrune(finput)
		for c != '>' && c != EOF && c != '\n' {
			tokname += string(c)
			c = getrune(finput)
		}

		if c != '>' {
			error("unterminated < ... > clause")
		}

		for i = 1; i <= ntypes; i++ {
			if typeset[i] == tokname {
				numbval = i
				if tokflag {
					fmt.Printf(">>> TYPENAME old <%v> %v\n", tokname, lineno)
				}
				return TYPENAME
			}
		}
		ntypes++
		numbval = ntypes
		typeset[numbval] = tokname
		if tokflag {
			fmt.Printf(">>> TYPENAME new <%v> %v\n", tokname, lineno)
		}
		return TYPENAME

	case '"', '\'':
		match = c
		tokname = " "
		for {
			c = getrune(finput)
			if c == '\n' || c == EOF {
				error("illegal or missing ' or \"")
			}
			if c == '\\' {
				tokname += string('\\')
				c = getrune(finput)
			} else if c == match {
				if tokflag {
					fmt.Printf(">>> IDENTIFIER \"%v\" %v\n", tokname, lineno)
				}
				return IDENTIFIER
			}
			tokname += string(c)
		}

	case '%':
		c = getrune(finput)
		switch c {
		case '%':
			if tokflag {
				fmt.Printf(">>> MARK %%%% %v\n", lineno)
			}
			return MARK
		case '=':
			if tokflag {
				fmt.Printf(">>> PREC %%= %v\n", lineno)
			}
			return PREC
		case '{':
			if tokflag {
				fmt.Printf(">>> LCURLY %%{ %v\n", lineno)
			}
			return LCURLY
		}

		getword(c)
		// find a reserved word
		for c = 0; c < len(resrv); c++ {
			if tokname == resrv[c].name {
				if tokflag {
					fmt.Printf(">>> %%%v %v %v\n", tokname,
						resrv[c].value-PRIVATE, lineno)
				}
				return resrv[c].value
			}
		}
		error("invalid escape, or illegal reserved word: %v", tokname)

	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		numbval = c - '0'
		for {
			c = getrune(finput)
			if !isdigit(c) {
				break
			}
			numbval = numbval*10 + c - '0'
		}
		ungetrune(finput, c)
		if tokflag {
			fmt.Printf(">>> NUMBER %v %v\n", numbval, lineno)
		}
		return NUMBER

	default:
		if isword(c) || c == '.' || c == '$' {
			getword(c)
			break
		}
		if tokflag {
			fmt.Printf(">>> OPERATOR %v %v\n", string(c), lineno)
		}
		return c
	}

	// look ahead to distinguish IDENTIFIER from IDENTCOLON
	c = getrune(finput)
	for c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\r' || c == '/' {
		if c == '\n' {
			peekline++
		}
		// look for comments
		if c == '/' {
			peekline += skipcom()
		}
		c = getrune(finput)
	}
	if c == ':' {
		if tokflag {
			fmt.Printf(">>> IDENTCOLON %v: %v\n", tokname, lineno)
		}
		return IDENTCOLON
	}

	ungetrune(finput, c)
	if tokflag {
		fmt.Printf(">>> IDENTIFIER %v %v\n", tokname, lineno)
	}
	return IDENTIFIER
}

func getword(c int) {
	tokname = ""
	for isword(c) || isdigit(c) || c == '_' || c == '.' || c == '$' {
		tokname += string(c)
		c = getrune(finput)
	}
	ungetrune(finput, c)
}

//
// determine the type of a symbol
//
func fdtype(t int) int {
	var v int
	var s string

	if t >= NTBASE {
		v = nontrst[t-NTBASE].value
		s = nontrst[t-NTBASE].name
	} else {
		v = TYPE(toklev[t])
		s = tokset[t].name
	}
	if v <= 0 {
		error("must specify type for %v", s)
	}
	return v
}

func chfind(t int, s string) int {
	if s[0] == ' ' {
		t = 0
	}
	for i := 0; i <= ntokens; i++ {
		if s == tokset[i].name {
			return i
		}
	}
	for i := 0; i <= nnonter; i++ {
		if s == nontrst[i].name {
			return NTBASE + i
		}
	}

	// cannot find name
	if t > 1 {
		error("%v should have been defined earlier", s)
	}
	return defin(t, s)
}

//
// copy the union declaration to the output, and the define file if present
//
func cpyunion() {

	if !lflag {
		fmt.Fprintf(ftable, "\n//line %v %v\n", lineno, infile)
	}
	fmt.Fprintf(ftable, "type\tYYSTYPE\tstruct")

	level := 0

out:
	for {
		c := getrune(finput)
		if c == EOF {
			error("EOF encountered while processing %%union")
		}
		putrune(ftable, c)
		switch c {
		case '\n':
			lineno++
		case '{':
			if level == 0 {
				fmt.Fprintf(ftable, "\n\tyys\tint;")
			}
			level++
		case '}':
			level--
			if level == 0 {
				break out
			}
		}
	}
	fmt.Fprintf(ftable, "\n")
	fmt.Fprintf(ftable, "var\tyylval\tYYSTYPE\n")
	fmt.Fprintf(ftable, "var\tYYVAL\tYYSTYPE\n")
	fmt.Fprintf(ftable, "var\tYYS\t[%v]YYSTYPE\n", stacksize)
}

//
// saves code between %{ and %}
//
func cpycode() {
	lno := lineno

	c := getrune(finput)
	if c == '\n' {
		c = getrune(finput)
		lineno++
	}
	if !lflag {
		fmt.Fprintf(ftable, "\n//line %v %v\n", lineno, infile)
	}
	for c != EOF {
		if c == '%' {
			c = getrune(finput)
			if c == '}' {
				return
			}
			putrune(ftable, '%')
		}
		putrune(ftable, c)
		if c == '\n' {
			lineno++
		}
		c = getrune(finput)
	}
	lineno = lno
	error("eof before %%}")
}

//func
//addcode(k int, s string)
//{
//	for i := 0; i < len(s); i++ {
//		addcodec(k, int(s[i]));
//	}
//}

//func
//addcodec(k, c int)
//{
//	if codehead == nil || k != codetail.kind || codetail.ndata >= NCode {
//		cd := new(Code);
//		cd.kind = k;
//		cd.data = make([]byte, NCode+UTFmax);
//		cd.ndata = 0;
//		cd.next = nil;
//
//		if codehead == nil {
//			codehead = cd;
//		} else
//			codetail.next = cd;
//		codetail = cd;
//	}
//
////!!	codetail.ndata += sys->char2byte(c, codetail.data, codetail.ndata);
//}

//func
//dumpcode(til int)
//{
//	for ; codehead != nil; codehead = codehead.next {
//		if codehead.kind == til {
//			return;
//		}
//		if write(ftable, codehead.data, codehead.ndata) != codehead.ndata {
//			error("can't write output file");
//		}
//	}
//}

//
// write out the module declaration and any token info
//
//func
//dumpmod()
//{
//
//	for ; codehead != nil; codehead = codehead.next {
//		if codehead.kind != CodeMod {
//			break;
//		}
//		if write(ftable, codehead.data, codehead.ndata) != codehead.ndata {
//			error("can't write output file");
//		}
//	}
//
//	for i:=TOKSTART; i<=ntokens; i++ {
//		// non-literals
//		c := tokset[i].name[0];
//		if c != ' ' && c != '$' {
//			fmt.Fprintf(ftable, "vonst	%v	%v\n",
//				tokset[i].name, tokset[i].value);
//		}
//	}
//
//}

//
// skip over comments
// skipcom is called after reading a '/'
//
func skipcom() int {
	var c int

	c = getrune(finput)
	if c == '/' {
		for c != EOF {
			if c == '\n' {
				return 1
			}
			c = getrune(finput)
		}
		error("EOF inside comment")
		return 0
	}
	if c != '*' {
		error("illegal comment")
	}

	nl := 0 // lines skipped
	c = getrune(finput)

l1:
	switch c {
	case '*':
		c = getrune(finput)
		if c == '/' {
			break
		}
		goto l1

	case '\n':
		nl++
		fallthrough

	default:
		c = getrune(finput)
		goto l1
	}
	return nl
}

func dumpprod(curprod []int, max int) {
	fmt.Printf("\n")
	for i := 0; i < max; i++ {
		p := curprod[i]
		if p < 0 {
			fmt.Printf("[%v] %v\n", i, p)
		} else {
			fmt.Printf("[%v] %v\n", i, symnam(p))
		}
	}
}

//
// copy action to the next ; or closing }
//
func cpyact(curprod []int, max int) {

	if !lflag {
		fmt.Fprintf(ftable, "\n//line %v %v\n", lineno, infile)
	}

	lno := lineno
	brac := 0

loop:
	for {
		c := getrune(finput)

	swt:
		switch c {
		case ';':
			if brac == 0 {
				putrune(ftable, c)
				return
			}

		case '{':
			if brac == 0 {
			}
			putrune(ftable, '\t')
			brac++

		case '$':
			s := 1
			tok := -1
			c = getrune(finput)

			// type description
			if c == '<' {
				ungetrune(finput, c)
				if gettok() != TYPENAME {
					error("bad syntax on $<ident> clause")
				}
				tok = numbval
				c = getrune(finput)
			}
			if c == '$' {
				fmt.Fprintf(ftable, "YYVAL")

				// put out the proper tag...
				if ntypes != 0 {
					if tok < 0 {
						tok = fdtype(curprod[0])
					}
					fmt.Fprintf(ftable, ".%v", typeset[tok])
				}
				continue loop
			}
			if c == '-' {
				s = -s
				c = getrune(finput)
			}
			j := 0
			if isdigit(c) {
				for isdigit(c) {
					j = j*10 + c - '0'
					c = getrune(finput)
				}
				ungetrune(finput, c)
				j = j * s
				if j >= max {
					error("Illegal use of $%v", j)
				}
			} else if isword(c) || c == '_' || c == '.' {
				// look for $name
				ungetrune(finput, c)
				if gettok() != IDENTIFIER {
					error("$ must be followed by an identifier")
				}
				tokn := chfind(2, tokname)
				fnd := -1
				c = getrune(finput)
				if c != '@' {
					ungetrune(finput, c)
				} else if gettok() != NUMBER {
					error("@ must be followed by number")
				} else {
					fnd = numbval
				}
				for j = 1; j < max; j++ {
					if tokn == curprod[j] {
						fnd--
						if fnd <= 0 {
							break
						}
					}
				}
				if j >= max {
					error("$name or $name@number not found")
				}
			} else {
				putrune(ftable, '$')
				if s < 0 {
					putrune(ftable, '-')
				}
				ungetrune(finput, c)
				continue loop
			}
			fmt.Fprintf(ftable, "YYS[yypt-%v]", max-j-1)

			// put out the proper tag
			if ntypes != 0 {
				if j <= 0 && tok < 0 {
					error("must specify type of $%v", j)
				}
				if tok < 0 {
					tok = fdtype(curprod[j])
				}
				fmt.Fprintf(ftable, ".%v", typeset[tok])
			}
			continue loop

		case '}':
			brac--
			if brac != 0 {
				break
			}
			putrune(ftable, c)
			return

		case '/':
			nc := getrune(finput)
			if nc != '/' && nc != '*' {
				ungetrune(finput, nc)
				break
			}
			// a comment
			putrune(ftable, c)
			putrune(ftable, nc)
			c = getrune(finput)
			for c != EOF {
				switch {
				case c == '\n':
					lineno++
					if nc == '/' { // end of // comment
						break swt
					}
				case c == '*' && nc == '*': // end of /* comment?
					nnc := getrune(finput)
					if nnc == '/' {
						putrune(ftable, '*')
						putrune(ftable, '/')
						c = getrune(finput)
						break swt
					}
					ungetrune(finput, nnc)
				}
				putrune(ftable, c)
				c = getrune(finput)
			}
			error("EOF inside comment")

		case '\'', '"':
			// character string or constant
			match := c
			putrune(ftable, c)
			c = getrune(finput)
			for c != EOF {
				if c == '\\' {
					putrune(ftable, c)
					c = getrune(finput)
					if c == '\n' {
						lineno++
					}
				} else if c == match {
					break swt
				}
				if c == '\n' {
					error("newline in string or char const")
				}
				putrune(ftable, c)
				c = getrune(finput)
			}
			error("EOF in string or character constant")

		case EOF:
			lineno = lno
			error("action does not terminate")

		case '\n':
			lineno++
		}

		putrune(ftable, c)
	}
}

func openup() {
	infile = flag.Arg(0)
	finput = open(infile)
	if finput == nil {
		error("cannot open %v", infile)
	}

	foutput = nil
	if vflag != "" {
		foutput = create(vflag, 0666)
		if foutput == nil {
			error("can't create file %v", vflag)
		}
	}

	ftable = nil
	if oflag == "" {
		oflag = "y.go"
	}
	ftable = create(oflag, 0666)
	if ftable == nil {
		error("can't create file %v", oflag)
	}

}

//
// return a pointer to the name of symbol i
//
func symnam(i int) string {
	var s string

	if i >= NTBASE {
		s = nontrst[i-NTBASE].name
	} else {
		s = tokset[i].name
	}
	if s[0] == ' ' {
		s = s[1:]
	}
	return s
}

//
// set elements 0 through n-1 to c
//
func aryfil(v []int, n, c int) {
	for i := 0; i < n; i++ {
		v[i] = c
	}
}

//
// compute an array with the beginnings of productions yielding given nonterminals
// The array pres points to these lists
// the array pyield has the lists: the total size is only NPROD+1
//
func cpres() {
	pres = make([][][]int, nnonter+1)
	curres := make([][]int, nprod)

	if false {
		for j := 0; j <= nnonter; j++ {
			fmt.Printf("nnonter[%v] = %v\n", j, nontrst[j].name)
		}
		for j := 0; j < nprod; j++ {
			fmt.Printf("prdptr[%v][0] = %v+NTBASE\n", j, prdptr[j][0]-NTBASE)
		}
	}

	fatfl = 0 // make undefined symbols nonfatal
	for i := 0; i <= nnonter; i++ {
		n := 0
		c := i + NTBASE
		for j := 0; j < nprod; j++ {
			if prdptr[j][0] == c {
				curres[n] = prdptr[j][1:]
				n++
			}
		}
		if n == 0 {
			error("nonterminal %v not defined", nontrst[i].name)
			continue
		}
		pres[i] = make([][]int, n)
		copy(pres[i], curres)
	}
	fatfl = 1
	if nerrors != 0 {
		summary()
		exit(1)
	}
}

func dumppres() {
	for i := 0; i <= nnonter; i++ {
		print("nonterm %d\n", i)
		curres := pres[i]
		for j := 0; j < len(curres); j++ {
			print("\tproduction %d:", j)
			prd := curres[j]
			for k := 0; k < len(prd); k++ {
				print(" %d", prd[k])
			}
			print("\n")
		}
	}
}

//
// mark nonterminals which derive the empty string
// also, look for nonterminals which don't derive any token strings
//
func cempty() {
	var i, p, np int
	var prd []int

	pempty = make([]int, nnonter+1)

	// first, use the array pempty to detect productions that can never be reduced
	// set pempty to WHONOWS
	aryfil(pempty, nnonter+1, WHOKNOWS)

	// now, look at productions, marking nonterminals which derive something
more:
	for {
		for i = 0; i < nprod; i++ {
			prd = prdptr[i]
			if pempty[prd[0]-NTBASE] != 0 {
				continue
			}
			np = len(prd) - 1
			for p = 1; p < np; p++ {
				if prd[p] >= NTBASE && pempty[prd[p]-NTBASE] == WHOKNOWS {
					break
				}
			}
			// production can be derived
			if p == np {
				pempty[prd[0]-NTBASE] = OK
				continue more
			}
		}
		break
	}

	// now, look at the nonterminals, to see if they are all OK
	for i = 0; i <= nnonter; i++ {
		// the added production rises or falls as the start symbol ...
		if i == 0 {
			continue
		}
		if pempty[i] != OK {
			fatfl = 0
			error("nonterminal " + nontrst[i].name + " never derives any token string")
		}
	}

	if nerrors != 0 {
		summary()
		exit(1)
	}

	// now, compute the pempty array, to see which nonterminals derive the empty string
	// set pempty to WHOKNOWS
	aryfil(pempty, nnonter+1, WHOKNOWS)

	// loop as long as we keep finding empty nonterminals

again:
	for {
	next:
		for i = 1; i < nprod; i++ {
			// not known to be empty
			prd = prdptr[i]
			if pempty[prd[0]-NTBASE] != WHOKNOWS {
				continue
			}
			np = len(prd) - 1
			for p = 1; p < np; p++ {
				if prd[p] < NTBASE || pempty[prd[p]-NTBASE] != EMPTY {
					continue next
				}
			}

			// we have a nontrivially empty nonterminal
			pempty[prd[0]-NTBASE] = EMPTY

			// got one ... try for another
			continue again
		}
		return
	}
}

func dumpempty() {
	for i := 0; i <= nnonter; i++ {
		if pempty[i] == EMPTY {
			print("non-term %d %s matches empty\n", i, symnam(i+NTBASE))
		}
	}
}

//
// compute an array with the first of nonterminals
//
func cpfir() {
	var s, n, p, np, ch, i int
	var curres [][]int
	var prd []int

	wsets = make([]Wset, nnonter+WSETINC)
	pfirst = make([]Lkset, nnonter+1)
	for i = 0; i <= nnonter; i++ {
		wsets[i].ws = mkset()
		pfirst[i] = mkset()
		curres = pres[i]
		n = len(curres)

		// initially fill the sets
		for s = 0; s < n; s++ {
			prd = curres[s]
			np = len(prd) - 1
			for p = 0; p < np; p++ {
				ch = prd[p]
				if ch < NTBASE {
					setbit(pfirst[i], ch)
					break
				}
				if pempty[ch-NTBASE] == 0 {
					break
				}
			}
		}
	}

	// now, reflect transitivity
	changes := 1
	for changes != 0 {
		changes = 0
		for i = 0; i <= nnonter; i++ {
			curres = pres[i]
			n = len(curres)
			for s = 0; s < n; s++ {
				prd = curres[s]
				np = len(prd) - 1
				for p = 0; p < np; p++ {
					ch = prd[p] - NTBASE
					if ch < 0 {
						break
					}
					changes |= setunion(pfirst[i], pfirst[ch])
					if pempty[ch] == 0 {
						break
					}
				}
			}
		}
	}

	if indebug == 0 {
		return
	}
	if foutput != nil {
		for i = 0; i <= nnonter; i++ {
			fmt.Fprintf(foutput, "\n%v: %v %v\n",
				nontrst[i].name, pfirst[i], pempty[i])
		}
	}
}

//
// generate the states
//
func stagen() {
	// initialize
	nstate = 0
	tstates = make([]int, ntokens+1)  // states generated by terminal gotos
	ntstates = make([]int, nnonter+1) // states generated by nonterminal gotos
	amem = make([]int, ACTSIZE)
	memp = 0

	clset = mkset()
	pstate[0] = 0
	pstate[1] = 0
	aryfil(clset, tbitset, 0)
	putitem(Pitem{prdptr[0], 0, 0, 0}, clset)
	tystate[0] = MUSTDO
	nstate = 1
	pstate[2] = pstate[1]

	//
	// now, the main state generation loop
	// first pass generates all of the states
	// later passes fix up lookahead
	// could be sped up a lot by remembering
	// results of the first pass rather than recomputing
	//
	first := 1
	for more := 1; more != 0; first = 0 {
		more = 0
		for i := 0; i < nstate; i++ {
			if tystate[i] != MUSTDO {
				continue
			}

			tystate[i] = DONE
			aryfil(temp1, nnonter+1, 0)

			// take state i, close it, and do gotos
			closure(i)

			// generate goto's
			for p := 0; p < cwp; p++ {
				pi := wsets[p]
				if pi.flag != 0 {
					continue
				}
				wsets[p].flag = 1
				c := pi.pitem.first
				if c <= 1 {
					if pstate[i+1]-pstate[i] <= p {
						tystate[i] = MUSTLOOKAHEAD
					}
					continue
				}

				// do a goto on c
				putitem(wsets[p].pitem, wsets[p].ws)
				for q := p + 1; q < cwp; q++ {
					// this item contributes to the goto
					if c == wsets[q].pitem.first {
						putitem(wsets[q].pitem, wsets[q].ws)
						wsets[q].flag = 1
					}
				}

				if c < NTBASE {
					state(c) // register new state
				} else {
					temp1[c-NTBASE] = state(c)
				}
			}

			if gsdebug != 0 && foutput != nil {
				fmt.Fprintf(foutput, "%v: ", i)
				for j := 0; j <= nnonter; j++ {
					if temp1[j] != 0 {
						fmt.Fprintf(foutput, "%v %v,", nontrst[j].name, temp1[j])
					}
				}
				fmt.Fprintf(foutput, "\n")
			}

			if first != 0 {
				indgo[i] = apack(temp1[1:], nnonter-1) - 1
			}

			more++
		}
	}
}

//
// generate the closure of state i
//
func closure(i int) {
	zzclose++

	// first, copy kernel of state i to wsets
	cwp = 0
	q := pstate[i+1]
	for p := pstate[i]; p < q; p++ {
		wsets[cwp].pitem = statemem[p].pitem
		wsets[cwp].flag = 1 // this item must get closed
		copy(wsets[cwp].ws, statemem[p].look)
		cwp++
	}

	// now, go through the loop, closing each item
	work := 1
	for work != 0 {
		work = 0
		for u := 0; u < cwp; u++ {
			if wsets[u].flag == 0 {
				continue
			}

			// dot is before c
			c := wsets[u].pitem.first
			if c < NTBASE {
				wsets[u].flag = 0
				// only interesting case is where . is before nonterminal
				continue
			}

			// compute the lookahead
			aryfil(clset, tbitset, 0)

			// find items involving c
			for v := u; v < cwp; v++ {
				if wsets[v].flag != 1 || wsets[v].pitem.first != c {
					continue
				}
				pi := wsets[v].pitem.prod
				ipi := wsets[v].pitem.off + 1

				wsets[v].flag = 0
				if nolook != 0 {
					continue
				}

				ch := pi[ipi]
				ipi++
				for ch > 0 {
					// terminal symbol
					if ch < NTBASE {
						setbit(clset, ch)
						break
					}

					// nonterminal symbol
					setunion(clset, pfirst[ch-NTBASE])
					if pempty[ch-NTBASE] == 0 {
						break
					}
					ch = pi[ipi]
					ipi++
				}
				if ch <= 0 {
					setunion(clset, wsets[v].ws)
				}
			}

			//
			// now loop over productions derived from c
			//
			curres := pres[c-NTBASE]
			n := len(curres)

		nexts:
			// initially fill the sets
			for s := 0; s < n; s++ {
				prd := curres[s]

				//
				// put these items into the closure
				// is the item there
				//
				for v := 0; v < cwp; v++ {
					// yes, it is there
					if wsets[v].pitem.off == 0 &&
						aryeq(wsets[v].pitem.prod, prd) != 0 {
						if nolook == 0 &&
							setunion(wsets[v].ws, clset) != 0 {
							wsets[v].flag = 1
							work = 1
						}
						continue nexts
					}
				}

				//  not there; make a new entry
				if cwp >= len(wsets) {
					awsets := make([]Wset, cwp+WSETINC)
					copy(awsets, wsets)
					wsets = awsets
				}
				wsets[cwp].pitem = Pitem{prd, 0, prd[0], -prd[len(prd)-1]}
				wsets[cwp].flag = 1
				wsets[cwp].ws = mkset()
				if nolook == 0 {
					work = 1
					copy(wsets[cwp].ws, clset)
				}
				cwp++
			}
		}
	}

	// have computed closure; flags are reset; return
	if cldebug != 0 && foutput != nil {
		fmt.Fprintf(foutput, "\nState %v, nolook = %v\n", i, nolook)
		for u := 0; u < cwp; u++ {
			if wsets[u].flag != 0 {
				fmt.Fprintf(foutput, "flag set\n")
			}
			wsets[u].flag = 0
			fmt.Fprintf(foutput, "\t%v", writem(wsets[u].pitem))
			prlook(wsets[u].ws)
			fmt.Fprintf(foutput, "\n")
		}
	}
}

//
// sorts last state,and sees if it equals earlier ones. returns state number
//
func state(c int) int {
	zzstate++
	p1 := pstate[nstate]
	p2 := pstate[nstate+1]
	if p1 == p2 {
		return 0 // null state
	}

	// sort the items
	var k, l int
	for k = p1 + 1; k < p2; k++ { // make k the biggest
		for l = k; l > p1; l-- {
			if statemem[l].pitem.prodno < statemem[l-1].pitem.prodno ||
				statemem[l].pitem.prodno == statemem[l-1].pitem.prodno &&
					statemem[l].pitem.off < statemem[l-1].pitem.off {
				s := statemem[l]
				statemem[l] = statemem[l-1]
				statemem[l-1] = s
			} else {
				break
			}
		}
	}

	size1 := p2 - p1 // size of state

	var i int
	if c >= NTBASE {
		i = ntstates[c-NTBASE]
	} else {
		i = tstates[c]
	}

look:
	for ; i != 0; i = mstates[i] {
		// get ith state
		q1 := pstate[i]
		q2 := pstate[i+1]
		size2 := q2 - q1
		if size1 != size2 {
			continue
		}
		k = p1
		for l = q1; l < q2; l++ {
			if aryeq(statemem[l].pitem.prod, statemem[k].pitem.prod) == 0 ||
				statemem[l].pitem.off != statemem[k].pitem.off {
				continue look
			}
			k++
		}

		// found it
		pstate[nstate+1] = pstate[nstate] // delete last state

		// fix up lookaheads
		if nolook != 0 {
			return i
		}
		k = p1
		for l = q1; l < q2; l++ {
			if setunion(statemem[l].look, statemem[k].look) != 0 {
				tystate[i] = MUSTDO
			}
			k++
		}
		return i
	}

	// state is new
	zznewstate++
	if nolook != 0 {
		error("yacc state/nolook error")
	}
	pstate[nstate+2] = p2
	if nstate+1 >= NSTATES {
		error("too many states")
	}
	if c >= NTBASE {
		mstates[nstate] = ntstates[c-NTBASE]
		ntstates[c-NTBASE] = nstate
	} else {
		mstates[nstate] = tstates[c]
		tstates[c] = nstate
	}
	tystate[nstate] = MUSTDO
	nstate++
	return nstate - 1
}

func putitem(p Pitem, set Lkset) {
	p.off++
	p.first = p.prod[p.off]

	if pidebug != 0 && foutput != nil {
		fmt.Fprintf(foutput, "putitem(%v), state %v\n", writem(p), nstate)
	}
	j := pstate[nstate+1]
	if j >= len(statemem) {
		asm := make([]Item, j+STATEINC)
		copy(asm, statemem)
		statemem = asm
	}
	statemem[j].pitem = p
	if nolook == 0 {
		s := mkset()
		copy(s, set)
		statemem[j].look = s
	}
	j++
	pstate[nstate+1] = j
}

//
// creates output string for item pointed to by pp
//
func writem(pp Pitem) string {
	var i int

	p := pp.prod
	q := chcopy(nontrst[prdptr[pp.prodno][0]-NTBASE].name) + ": "
	npi := pp.off

	pi := aryeq(p, prdptr[pp.prodno])

	for {
		c := ' '
		if pi == npi {
			c = '.'
		}
		q += string(c)

		i = p[pi]
		pi++
		if i <= 0 {
			break
		}
		q += chcopy(symnam(i))
	}

	// an item calling for a reduction
	i = p[npi]
	if i < 0 {
		q += fmt.Sprintf("    (%v)", -i)
	}

	return q
}

//
// pack state i from temp1 into amem
//
func apack(p []int, n int) int {
	//
	// we don't need to worry about checking because
	// we will only look at entries known to be there...
	// eliminate leading and trailing 0's
	//
	off := 0
	pp := 0
	for ; pp <= n && p[pp] == 0; pp++ {
		off--
	}

	// no actions
	if pp > n {
		return 0
	}
	for ; n > pp && p[n] == 0; n-- {
	}
	p = p[pp : n+1]

	// now, find a place for the elements from p to q, inclusive
	r := len(amem) - len(p)

nextk:
	for rr := 0; rr <= r; rr++ {
		qq := rr
		for pp = 0; pp < len(p); pp++ {
			if p[pp] != 0 {
				if p[pp] != amem[qq] && amem[qq] != 0 {
					continue nextk
				}
			}
			qq++
		}

		// we have found an acceptable k
		if pkdebug != 0 && foutput != nil {
			fmt.Fprintf(foutput, "off = %v, k = %v\n", off+rr, rr)
		}
		qq = rr
		for pp = 0; pp < len(p); pp++ {
			if p[pp] != 0 {
				if qq > memp {
					memp = qq
				}
				amem[qq] = p[pp]
			}
			qq++
		}
		if pkdebug != 0 && foutput != nil {
			for pp = 0; pp <= memp; pp += 10 {
				fmt.Fprintf(foutput, "\n")
				for qq = pp; qq <= pp+9; qq++ {
					fmt.Fprintf(foutput, "%v ", amem[qq])
				}
				fmt.Fprintf(foutput, "\n")
			}
		}
		return off + rr
	}
	error("no space in action table")
	return 0
}

//
// print the output for the states
//
func output() {
	var c, u, v int

	fmt.Fprintf(ftable, "var\tYYEXCA = []int {\n")

	noset := mkset()

	// output the stuff for state i
	for i := 0; i < nstate; i++ {
		nolook = 0
		if tystate[i] != MUSTLOOKAHEAD {
			nolook = 1
		}
		closure(i)

		// output actions
		nolook = 1
		aryfil(temp1, ntokens+nnonter+1, 0)
		for u = 0; u < cwp; u++ {
			c = wsets[u].pitem.first
			if c > 1 && c < NTBASE && temp1[c] == 0 {
				for v = u; v < cwp; v++ {
					if c == wsets[v].pitem.first {
						putitem(wsets[v].pitem, noset)
					}
				}
				temp1[c] = state(c)
			} else if c > NTBASE {
				c -= NTBASE
				if temp1[c+ntokens] == 0 {
					temp1[c+ntokens] = amem[indgo[i]+c]
				}
			}
		}
		if i == 1 {
			temp1[1] = ACCEPTCODE
		}

		// now, we have the shifts; look at the reductions
		lastred = 0
		for u = 0; u < cwp; u++ {
			c = wsets[u].pitem.first

			// reduction
			if c > 0 {
				continue
			}
			lastred = -c
			us := wsets[u].ws
			for k := 0; k <= ntokens; k++ {
				if bitset(us, k) == 0 {
					continue
				}
				if temp1[k] == 0 {
					temp1[k] = c
				} else if temp1[k] < 0 { // reduce/reduce conflict
					if foutput != nil {
						fmt.Fprintf(foutput,
							"\n %v: reduce/reduce conflict  (red'ns "+
								"%v and %v) on %v",
							i, -temp1[k], lastred, symnam(k))
					}
					if -temp1[k] > lastred {
						temp1[k] = -lastred
					}
					zzrrconf++
				} else {
					// potential shift/reduce conflict
					precftn(lastred, k, i)
				}
			}
		}
		wract(i)
	}

	fmt.Fprintf(ftable, "}\n")
	fmt.Fprintf(ftable, "const\tYYNPROD\t= %v\n", nprod)
	fmt.Fprintf(ftable, "const\tYYPRIVATE\t= %v\n", PRIVATE)
	fmt.Fprintf(ftable, "var\tYYTOKENNAMES []string\n")
	fmt.Fprintf(ftable, "var\tYYSTATES []string\n")
}

//
// decide a shift/reduce conflict by precedence.
// r is a rule number, t a token number
// the conflict is in state s
// temp1[t] is changed to reflect the action
//
func precftn(r, t, s int) {
	var action int

	lp := levprd[r]
	lt := toklev[t]
	if PLEVEL(lt) == 0 || PLEVEL(lp) == 0 {
		// conflict
		if foutput != nil {
			fmt.Fprintf(foutput,
				"\n%v: shift/reduce conflict (shift %v(%v), red'n %v(%v)) on %v",
				s, temp1[t], PLEVEL(lt), r, PLEVEL(lp), symnam(t))
		}
		zzsrconf++
		return
	}
	if PLEVEL(lt) == PLEVEL(lp) {
		action = ASSOC(lt)
	} else if PLEVEL(lt) > PLEVEL(lp) {
		action = RASC // shift
	} else {
		action = LASC
	} // reduce
	switch action {
	case BASC: // error action
		temp1[t] = ERRCODE
	case LASC: // reduce
		temp1[t] = -r
	}
}

//
// output state i
// temp1 has the actions, lastred the default
//
func wract(i int) {
	var p, p1 int

	// find the best choice for lastred
	lastred = 0
	ntimes := 0
	for j := 0; j <= ntokens; j++ {
		if temp1[j] >= 0 {
			continue
		}
		if temp1[j]+lastred == 0 {
			continue
		}
		// count the number of appearances of temp1[j]
		count := 0
		tred := -temp1[j]
		levprd[tred] |= REDFLAG
		for p = 0; p <= ntokens; p++ {
			if temp1[p]+tred == 0 {
				count++
			}
		}
		if count > ntimes {
			lastred = tred
			ntimes = count
		}
	}

	//
	// for error recovery, arrange that, if there is a shift on the
	// error recovery token, `error', that the default be the error action
	//
	if temp1[2] > 0 {
		lastred = 0
	}

	// clear out entries in temp1 which equal lastred
	// count entries in optst table
	n := 0
	for p = 0; p <= ntokens; p++ {
		p1 = temp1[p]
		if p1+lastred == 0 {
			temp1[p] = 0
			p1 = 0
		}
		if p1 > 0 && p1 != ACCEPTCODE && p1 != ERRCODE {
			n++
		}
	}

	wrstate(i)
	defact[i] = lastred
	flag := 0
	os := make([]int, n*2)
	n = 0
	for p = 0; p <= ntokens; p++ {
		p1 = temp1[p]
		if p1 != 0 {
			if p1 < 0 {
				p1 = -p1
			} else if p1 == ACCEPTCODE {
				p1 = -1
			} else if p1 == ERRCODE {
				p1 = 0
			} else {
				os[n] = p
				n++
				os[n] = p1
				n++
				zzacent++
				continue
			}
			if flag == 0 {
				fmt.Fprintf(ftable, "-1, %v,\n", i)
			}
			flag++
			fmt.Fprintf(ftable, "\t%v, %v,\n", p, p1)
			zzexcp++
		}
	}
	if flag != 0 {
		defact[i] = -2
		fmt.Fprintf(ftable, "\t-2, %v,\n", lastred)
	}
	optst[i] = os
}

//
// writes state i
//
func wrstate(i int) {
	var j0, j1, u int
	var pp, qq int

	if foutput == nil {
		return
	}
	fmt.Fprintf(foutput, "\nstate %v\n", i)
	qq = pstate[i+1]
	for pp = pstate[i]; pp < qq; pp++ {
		fmt.Fprintf(foutput, "\t%v\n", writem(statemem[pp].pitem))
	}
	if tystate[i] == MUSTLOOKAHEAD {
		// print out empty productions in closure
		for u = pstate[i+1] - pstate[i]; u < cwp; u++ {
			if wsets[u].pitem.first < 0 {
				fmt.Fprintf(foutput, "\t%v\n", writem(wsets[u].pitem))
			}
		}
	}

	// check for state equal to another
	for j0 = 0; j0 <= ntokens; j0++ {
		j1 = temp1[j0]
		if j1 != 0 {
			fmt.Fprintf(foutput, "\n\t%v  ", symnam(j0))

			// shift, error, or accept
			if j1 > 0 {
				if j1 == ACCEPTCODE {
					fmt.Fprintf(foutput, "accept")
				} else if j1 == ERRCODE {
					fmt.Fprintf(foutput, "error")
				} else {
					fmt.Fprintf(foutput, "shift %v", j1)
				}
			} else {
				fmt.Fprintf(foutput, "reduce %v (src line %v)", -j1, rlines[-j1])
			}
		}
	}

	// output the final production
	if lastred != 0 {
		fmt.Fprintf(foutput, "\n\t.  reduce %v (src line %v)\n\n",
			lastred, rlines[lastred])
	} else {
		fmt.Fprintf(foutput, "\n\t.  error\n\n")
	}

	// now, output nonterminal actions
	j1 = ntokens
	for j0 = 1; j0 <= nnonter; j0++ {
		j1++
		if temp1[j1] != 0 {
			fmt.Fprintf(foutput, "\t%v  goto %v\n", symnam(j0+NTBASE), temp1[j1])
		}
	}
}

//
// output the gotos for the nontermninals
//
func go2out() {
	for i := 1; i <= nnonter; i++ {
		go2gen(i)

		// find the best one to make default
		best := -1
		times := 0

		// is j the most frequent
		for j := 0; j < nstate; j++ {
			if tystate[j] == 0 {
				continue
			}
			if tystate[j] == best {
				continue
			}

			// is tystate[j] the most frequent
			count := 0
			cbest := tystate[j]
			for k := j; k < nstate; k++ {
				if tystate[k] == cbest {
					count++
				}
			}
			if count > times {
				best = cbest
				times = count
			}
		}

		// best is now the default entry
		zzgobest += times - 1
		n := 0
		for j := 0; j < nstate; j++ {
			if tystate[j] != 0 && tystate[j] != best {
				n++
			}
		}
		goent := make([]int, 2*n+1)
		n = 0
		for j := 0; j < nstate; j++ {
			if tystate[j] != 0 && tystate[j] != best {
				goent[n] = j
				n++
				goent[n] = tystate[j]
				n++
				zzgoent++
			}
		}

		// now, the default
		if best == -1 {
			best = 0
		}

		zzgoent++
		goent[n] = best
		yypgo[i] = goent
	}
}

//
// output the gotos for nonterminal c
//
func go2gen(c int) {
	var i, cc, p, q int

	// first, find nonterminals with gotos on c
	aryfil(temp1, nnonter+1, 0)
	temp1[c] = 1
	work := 1
	for work != 0 {
		work = 0
		for i = 0; i < nprod; i++ {
			// cc is a nonterminal with a goto on c
			cc = prdptr[i][1] - NTBASE
			if cc >= 0 && temp1[cc] != 0 {
				// thus, the left side of production i does too
				cc = prdptr[i][0] - NTBASE
				if temp1[cc] == 0 {
					work = 1
					temp1[cc] = 1
				}
			}
		}
	}

	// now, we have temp1[c] = 1 if a goto on c in closure of cc
	if g2debug != 0 && foutput != nil {
		fmt.Fprintf(foutput, "%v: gotos on ", nontrst[c].name)
		for i = 0; i <= nnonter; i++ {
			if temp1[i] != 0 {
				fmt.Fprintf(foutput, "%v ", nontrst[i].name)
			}
		}
		fmt.Fprintf(foutput, "\n")
	}

	// now, go through and put gotos into tystate
	aryfil(tystate, nstate, 0)
	for i = 0; i < nstate; i++ {
		q = pstate[i+1]
		for p = pstate[i]; p < q; p++ {
			cc = statemem[p].pitem.first
			if cc >= NTBASE {
				// goto on c is possible
				if temp1[cc-NTBASE] != 0 {
					tystate[i] = amem[indgo[i]+c]
					break
				}
			}
		}
	}
}

//
// in order to free up the mem and amem arrays for the optimizer,
// and still be able to output yyr1, etc., after the sizes of
// the action array is known, we hide the nonterminals
// derived by productions in levprd.
//
func hideprod() {
	nred := 0
	levprd[0] = 0
	for i := 1; i < nprod; i++ {
		if (levprd[i] & REDFLAG) == 0 {
			if foutput != nil {
				fmt.Fprintf(foutput, "Rule not reduced: %v\n",
					writem(Pitem{prdptr[i], 0, 0, i}))
			}
			fmt.Printf("rule %v never reduced\n", writem(Pitem{prdptr[i], 0, 0, i}))
			nred++
		}
		levprd[i] = prdptr[i][0] - NTBASE
	}
	if nred != 0 {
		fmt.Printf("%v rules never reduced\n", nred)
	}
}

func callopt() {
	var j, k, p, q, i int
	var v []int

	pgo = make([]int, nnonter+1)
	pgo[0] = 0
	maxoff = 0
	maxspr = 0
	for i = 0; i < nstate; i++ {
		k = 32000
		j = 0
		v = optst[i]
		q = len(v)
		for p = 0; p < q; p += 2 {
			if v[p] > j {
				j = v[p]
			}
			if v[p] < k {
				k = v[p]
			}
		}

		// nontrivial situation
		if k <= j {
			// j is now the range
			//			j -= k;			// call scj
			if k > maxoff {
				maxoff = k
			}
		}
		tystate[i] = q + 2*j
		if j > maxspr {
			maxspr = j
		}
	}

	// initialize ggreed table
	ggreed = make([]int, nnonter+1)
	for i = 1; i <= nnonter; i++ {
		ggreed[i] = 1
		j = 0

		// minimum entry index is always 0
		v = yypgo[i]
		q = len(v) - 1
		for p = 0; p < q; p += 2 {
			ggreed[i] += 2
			if v[p] > j {
				j = v[p]
			}
		}
		ggreed[i] = ggreed[i] + 2*j
		if j > maxoff {
			maxoff = j
		}
	}

	// now, prepare to put the shift actions into the amem array
	for i = 0; i < ACTSIZE; i++ {
		amem[i] = 0
	}
	maxa = 0
	for i = 0; i < nstate; i++ {
		if tystate[i] == 0 && adb > 1 {
			fmt.Fprintf(ftable, "State %v: null\n", i)
		}
		indgo[i] = YYFLAG
	}

	i = nxti()
	for i != NOMORE {
		if i >= 0 {
			stin(i)
		} else {
			gin(-i)
		}
		i = nxti()
	}

	// print amem array
	if adb > 2 {
		for p = 0; p <= maxa; p += 10 {
			fmt.Fprintf(ftable, "%v  ", p)
			for i = 0; i < 10; i++ {
				fmt.Fprintf(ftable, "%v  ", amem[p+i])
			}
			putrune(ftable, '\n')
		}
	}

	aoutput()
	osummary()
}

//
// finds the next i
//
func nxti() int {
	max := 0
	maxi := 0
	for i := 1; i <= nnonter; i++ {
		if ggreed[i] >= max {
			max = ggreed[i]
			maxi = -i
		}
	}
	for i := 0; i < nstate; i++ {
		if tystate[i] >= max {
			max = tystate[i]
			maxi = i
		}
	}
	if max == 0 {
		return NOMORE
	}
	return maxi
}

func gin(i int) {
	var s int

	// enter gotos on nonterminal i into array amem
	ggreed[i] = 0

	q := yypgo[i]
	nq := len(q) - 1

	// now, find amem place for it
nextgp:
	for p := 0; p < ACTSIZE; p++ {
		if amem[p] != 0 {
			continue
		}
		for r := 0; r < nq; r += 2 {
			s = p + q[r] + 1
			if s > maxa {
				maxa = s
				if maxa >= ACTSIZE {
					error("a array overflow")
				}
			}
			if amem[s] != 0 {
				continue nextgp
			}
		}

		// we have found amem spot
		amem[p] = q[nq]
		if p > maxa {
			maxa = p
		}
		for r := 0; r < nq; r += 2 {
			s = p + q[r] + 1
			amem[s] = q[r+1]
		}
		pgo[i] = p
		if adb > 1 {
			fmt.Fprintf(ftable, "Nonterminal %v, entry at %v\n", i, pgo[i])
		}
		return
	}
	error("cannot place goto %v\n", i)
}

func stin(i int) {
	var s int

	tystate[i] = 0

	// enter state i into the amem array
	q := optst[i]
	nq := len(q)

nextn:
	// find an acceptable place
	for n := -maxoff; n < ACTSIZE; n++ {
		flag := 0
		for r := 0; r < nq; r += 2 {
			s = q[r] + n
			if s < 0 || s > ACTSIZE {
				continue nextn
			}
			if amem[s] == 0 {
				flag++
			} else if amem[s] != q[r+1] {
				continue nextn
			}
		}

		// check the position equals another only if the states are identical
		for j := 0; j < nstate; j++ {
			if indgo[j] == n {

				// we have some disagreement
				if flag != 0 {
					continue nextn
				}
				if nq == len(optst[j]) {

					// states are equal
					indgo[i] = n
					if adb > 1 {
						fmt.Fprintf(ftable, "State %v: entry at"+
							"%v equals state %v\n",
							i, n, j)
					}
					return
				}

				// we have some disagreement
				continue nextn
			}
		}

		for r := 0; r < nq; r += 2 {
			s = q[r] + n
			if s > maxa {
				maxa = s
			}
			if amem[s] != 0 && amem[s] != q[r+1] {
				error("clobber of a array, pos'n %v, by %v", s, q[r+1])
			}
			amem[s] = q[r+1]
		}
		indgo[i] = n
		if adb > 1 {
			fmt.Fprintf(ftable, "State %v: entry at %v\n", i, indgo[i])
		}
		return
	}
	error("Error; failure to place state %v", i)
}

//
// this version is for limbo
// write out the optimized parser
//
func aoutput() {
	fmt.Fprintf(ftable, "const\tYYLAST\t= %v\n", maxa+1)
	arout("YYACT", amem, maxa+1)
	arout("YYPACT", indgo, nstate)
	arout("YYPGO", pgo, nnonter+1)
}

//
// put out other arrays, copy the parsers
//
func others() {
	var i, j int

	arout("YYR1", levprd, nprod)
	aryfil(temp1, nprod, 0)

	//
	//yyr2 is the number of rules for each production
	//
	for i = 1; i < nprod; i++ {
		temp1[i] = len(prdptr[i]) - 2
	}
	arout("YYR2", temp1, nprod)

	aryfil(temp1, nstate, -1000)
	for i = 0; i <= ntokens; i++ {
		for j := tstates[i]; j != 0; j = mstates[j] {
			temp1[j] = i
		}
	}
	for i = 0; i <= nnonter; i++ {
		for j = ntstates[i]; j != 0; j = mstates[j] {
			temp1[j] = -i
		}
	}
	arout("YYCHK", temp1, nstate)
	arout("YYDEF", defact, nstate)

	// put out token translation tables
	// table 1 has 0-256
	aryfil(temp1, 256, 0)
	c := 0
	for i = 1; i <= ntokens; i++ {
		j = tokset[i].value
		if j >= 0 && j < 256 {
			if temp1[j] != 0 {
				print("yacc bug -- cant have 2 different Ts with same value\n")
				print("	%s and %s\n", tokset[i].name, tokset[temp1[j]].name)
				nerrors++
			}
			temp1[j] = i
			if j > c {
				c = j
			}
		}
	}
	for i = 0; i <= c; i++ {
		if temp1[i] == 0 {
			temp1[i] = YYLEXUNK
		}
	}
	arout("YYTOK1", temp1, c+1)

	// table 2 has PRIVATE-PRIVATE+256
	aryfil(temp1, 256, 0)
	c = 0
	for i = 1; i <= ntokens; i++ {
		j = tokset[i].value - PRIVATE
		if j >= 0 && j < 256 {
			if temp1[j] != 0 {
				print("yacc bug -- cant have 2 different Ts with same value\n")
				print("	%s and %s\n", tokset[i].name, tokset[temp1[j]].name)
				nerrors++
			}
			temp1[j] = i
			if j > c {
				c = j
			}
		}
	}
	arout("YYTOK2", temp1, c+1)

	// table 3 has everything else
	fmt.Fprintf(ftable, "var\tYYTOK3\t= []int {\n")
	c = 0
	for i = 1; i <= ntokens; i++ {
		j = tokset[i].value
		if j >= 0 && j < 256 {
			continue
		}
		if j >= PRIVATE && j < 256+PRIVATE {
			continue
		}

		fmt.Fprintf(ftable, "%4d,%4d,", j, i)
		c++
		if c%5 == 0 {
			putrune(ftable, '\n')
		}
	}
	fmt.Fprintf(ftable, "%4d,\n };\n", 0)

	// copy parser text
	c = getrune(finput)
	for c != EOF {
		putrune(ftable, c)
		c = getrune(finput)
	}

	// copy yaccpar
	fmt.Fprintf(ftable, "%v", yaccpar)
}

func arout(s string, v []int, n int) {
	fmt.Fprintf(ftable, "var\t%v\t= []int {\n", s)
	for i := 0; i < n; i++ {
		if i%10 == 0 {
			putrune(ftable, '\n')
		}
		fmt.Fprintf(ftable, "%4d", v[i])
		putrune(ftable, ',')
	}
	fmt.Fprintf(ftable, "\n};\n")
}

//
// output the summary on y.output
//
func summary() {
	if foutput != nil {
		fmt.Fprintf(foutput, "\n%v terminals, %v nonterminals\n", ntokens, nnonter+1)
		fmt.Fprintf(foutput, "%v grammar rules, %v/%v states\n", nprod, nstate, NSTATES)
		fmt.Fprintf(foutput, "%v shift/reduce, %v reduce/reduce conflicts reported\n", zzsrconf, zzrrconf)
		fmt.Fprintf(foutput, "%v working sets used\n", len(wsets))
		fmt.Fprintf(foutput, "memory: parser %v/%v\n", memp, ACTSIZE)
		fmt.Fprintf(foutput, "%v extra closures\n", zzclose-2*nstate)
		fmt.Fprintf(foutput, "%v shift entries, %v exceptions\n", zzacent, zzexcp)
		fmt.Fprintf(foutput, "%v goto entries\n", zzgoent)
		fmt.Fprintf(foutput, "%v entries saved by goto default\n", zzgobest)
	}
	if zzsrconf != 0 || zzrrconf != 0 {
		fmt.Printf("\nconflicts: ")
		if zzsrconf != 0 {
			fmt.Printf("%v shift/reduce", zzsrconf)
		}
		if zzsrconf != 0 && zzrrconf != 0 {
			fmt.Printf(", ")
		}
		if zzrrconf != 0 {
			fmt.Printf("%v reduce/reduce", zzrrconf)
		}
		fmt.Printf("\n")
	}
}

//
// write optimizer summary
//
func osummary() {
	if foutput == nil {
		return
	}
	i := 0
	for p := maxa; p >= 0; p-- {
		if amem[p] == 0 {
			i++
		}
	}

	fmt.Fprintf(foutput, "Optimizer space used: output %v/%v\n", maxa+1, ACTSIZE)
	fmt.Fprintf(foutput, "%v table entries, %v zero\n", maxa+1, i)
	fmt.Fprintf(foutput, "maximum spread: %v, maximum offset: %v\n", maxspr, maxoff)
}

//
// copies and protects "'s in q
//
func chcopy(q string) string {
	s := ""
	i := 0
	j := 0
	for i = 0; i < len(q); i++ {
		if q[i] == '"' {
			s += q[j:i] + "\\"
			j = i
		}
	}
	return s + q[j:i]
}

func usage() {
	fmt.Fprintf(stderr, "usage: gacc [-o output] [-v parsetable] input\n")
	exit(1)
}

func bitset(set Lkset, bit int) int { return set[bit>>5] & (1 << uint(bit&31)) }

func setbit(set Lkset, bit int) { set[bit>>5] |= (1 << uint(bit&31)) }

func mkset() Lkset { return make([]int, tbitset) }

//
// set a to the union of a and b
// return 1 if b is not a subset of a, 0 otherwise
//
func setunion(a, b []int) int {
	sub := 0
	for i := 0; i < tbitset; i++ {
		x := a[i]
		y := x | b[i]
		a[i] = y
		if y != x {
			sub = 1
		}
	}
	return sub
}

func prlook(p Lkset) {
	if p == nil {
		fmt.Fprintf(foutput, "\tNULL")
		return
	}
	fmt.Fprintf(foutput, " { ")
	for j := 0; j <= ntokens; j++ {
		if bitset(p, j) != 0 {
			fmt.Fprintf(foutput, "%v ", symnam(j))
		}
	}
	fmt.Fprintf(foutput, "}")
}

//
// utility routines
//
var peekrune int

func isdigit(c int) bool { return c >= '0' && c <= '9' }

func isword(c int) bool {
	return c >= 0xa0 || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func mktemp(t string) string { return t }

//
// return 1 if 2 arrays are equal
// return 0 if not equal
//
func aryeq(a []int, b []int) int {
	n := len(a)
	if len(b) != n {
		return 0
	}
	for ll := 0; ll < n; ll++ {
		if a[ll] != b[ll] {
			return 0
		}
	}
	return 1
}

func putrune(f *bufio.Writer, c int) {
	s := string(c)
	for i := 0; i < len(s); i++ {
		f.WriteByte(s[i])
	}
}

func getrune(f *bufio.Reader) int {
	var r int

	if peekrune != 0 {
		if peekrune == EOF {
			return EOF
		}
		r = peekrune
		peekrune = 0
		return r
	}

	c, n, err := f.ReadRune()
	if n == 0 {
		return EOF
	}
	if err != nil {
		error("read error: %v", err)
	}
	//fmt.Printf("rune = %v n=%v\n", string(c), n);
	return c
}

func ungetrune(f *bufio.Reader, c int) {
	if f != finput {
		panic("ungetc - not finput")
	}
	if peekrune != 0 {
		panic("ungetc - 2nd unget")
	}
	peekrune = c
}

func write(f *bufio.Writer, b []byte, n int) int {
	println("write")
	return 0
}

func open(s string) *bufio.Reader {
	fi, err := os.Open(s, os.O_RDONLY, 0)
	if err != nil {
		error("error opening %v: %v", s, err)
	}
	//fmt.Printf("open %v\n", s);
	return bufio.NewReader(fi)
}

func create(s string, m int) *bufio.Writer {
	fo, err := os.Open(s, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, m)
	if err != nil {
		error("error opening %v: %v", s, err)
	}
	//fmt.Printf("create %v mode %v\n", s, m);
	return bufio.NewWriter(fo)
}

//
// write out error comment
//
func error(s string, v ...interface{}) {
	nerrors++
	fmt.Fprintf(stderr, s, v)
	fmt.Fprintf(stderr, ": %v:%v\n", infile, lineno)
	if fatfl != 0 {
		summary()
		exit(1)
	}
}

func exit(status int) {
	if ftable != nil {
		ftable.Flush()
		ftable = nil
	}
	if foutput != nil {
		foutput.Flush()
		foutput = nil
	}
	if stderr != nil {
		stderr.Flush()
		stderr = nil
	}
	os.Exit(status)
}

var yaccpar = `
/*	parser for yacc output	*/

var	Nerrs		= 0		/* number of errors */
var	Errflag		= 0		/* error recovery flag */
var	Debug		= 0
const	YYFLAG		= -1000

func
Tokname(yyc int) string {
	if yyc > 0 && yyc <= len(Toknames) {
		if Toknames[yyc-1] != "" {
			return Toknames[yyc-1];
		}
	}
	return fmt.Sprintf("tok-%v", yyc);
}

func
Statname(yys int) string {
	if yys >= 0 && yys < len(Statenames) {
		if Statenames[yys] != "" {
			return Statenames[yys];
		}
	}
	return fmt.Sprintf("state-%v", yys);
}

func
lex1() int {
	var yychar int;
	var c int;

	yychar = Lex();
	if yychar <= 0 {
		c = YYTOK1[0];
		goto out;
	}
	if yychar < len(YYTOK1) {
		c = YYTOK1[yychar];
		goto out;
	}
	if yychar >= YYPRIVATE {
		if yychar < YYPRIVATE+len(YYTOK2) {
			c = YYTOK2[yychar-YYPRIVATE];
			goto out;
		}
	}
	for i:=0; i<len(YYTOK3); i+=2 {
		c = YYTOK3[i+0];
		if c == yychar {
			c = YYTOK3[i+1];
			goto out;
		}
	}
	c = 0;

out:
	if c == 0 {
		c = YYTOK2[1];	/* unknown char */
	}
	if Debug >= 3 {
		fmt.Printf("lex %.4lux %s\n", yychar, Tokname(c));
	}
	return c;
}

func
Parse() int {
	var yyj, yystate, yyn, yyg, yyxi, yyp int;
	var yychar int;
	var yypt, yynt int;

	yystate = 0;
	yychar = -1;
	Nerrs = 0;
	Errflag = 0;
	yyp = -1;
	goto yystack;

ret0:
	return 0;

ret1:
	return 1;

yystack:
	/* put a state and value onto the stack */
	if Debug >= 4 {
		fmt.Printf("char %v in %v", Tokname(yychar), Statname(yystate));
	}

	yyp++;
	if yyp >= len(YYS) {
		Error("yacc stack overflow");
		goto ret1;
	}
	YYS[yyp] = YYVAL;
	YYS[yyp].yys = yystate;

yynewstate:
	yyn = YYPACT[yystate];
	if yyn <= YYFLAG {
		goto yydefault; /* simple state */
	}
	if yychar < 0 {
		yychar = lex1();
	}
	yyn += yychar;
	if yyn < 0 || yyn >= YYLAST {
		goto yydefault;
	}
	yyn = YYACT[yyn];
	if YYCHK[yyn] == yychar { /* valid shift */
		yychar = -1;
		YYVAL = yylval;
		yystate = yyn;
		if Errflag > 0 {
			Errflag--;
		}
		goto yystack;
	}

yydefault:
	/* default state action */
	yyn = YYDEF[yystate];
	if yyn == -2 {
		if yychar < 0 {
			yychar = lex1();
		}

		/* look through exception table */
		for yyxi=0;; yyxi+=2 {
			if YYEXCA[yyxi+0] == -1 && YYEXCA[yyxi+1] == yystate {
				break;
			}
		}
		for yyxi += 2;; yyxi += 2 {
			yyn = YYEXCA[yyxi+0];
			if yyn < 0 || yyn == yychar {
				break;
			}
		}
		yyn = YYEXCA[yyxi+1];
		if yyn < 0 {
			goto ret0;
		}
	}
	if yyn == 0 {
		/* error ... attempt to resume parsing */
		switch Errflag {
		case 0:   /* brand new error */
			Error("syntax error");
			Nerrs++;
			if Debug >= 1 {
				fmt.Printf("%s", Statname(yystate));
				fmt.Printf("saw %s\n", Tokname(yychar));
			}
			fallthrough;

		case 1,2: /* incompletely recovered error ... try again */
			Errflag = 3;

			/* find a state where "error" is a legal shift action */
			for yyp >= len(YYS) {
				yyn = YYPACT[YYS[yyp].yys] + YYERRCODE;
				if yyn >= 0 && yyn < YYLAST {
					yystate = YYACT[yyn];  /* simulate a shift of "error" */
					if YYCHK[yystate] == YYERRCODE {
						goto yystack;
					}
				}

				/* the current yyp has no shift onn "error", pop stack */
				if Debug >= 2 {
					fmt.Printf("error recovery pops state %d, uncovers %d\n",
						YYS[yyp].yys, YYS[yyp-1].yys );
				}
				yyp--;
			}
			/* there is no state on the stack with an error shift ... abort */
			goto ret1;

		case 3:  /* no shift yet; clobber input char */
			if Debug >= 2 {
				fmt.Printf("error recovery discards %s\n", Tokname(yychar));
			}
			if yychar == YYEOFCODE {
				goto ret1;
			}
			yychar = -1;
			goto yynewstate;   /* try again in the same state */
		}
	}

	/* reduction by production yyn */
	if Debug >= 2 {
		fmt.Printf("reduce %v in:\n\t%v", yyn, Statname(yystate));
	}

	yynt = yyn;
	yypt = yyp;

	yyp -= YYR2[yyn];
	YYVAL = YYS[yyp+1];

	/* consult goto table to find next state */
	yyn = YYR1[yyn];
	yyg = YYPGO[yyn];
	yyj = yyg + YYS[yyp].yys + 1;

	if yyj >= YYLAST {
		yystate = YYACT[yyg];
	} else {
		yystate = YYACT[yyj];
		if YYCHK[yystate] != -yyn {
			yystate = YYACT[yyg];
		}
	}

	yyrun(yynt, yypt);
	goto yystack;  /* stack new state and value */
}
`
