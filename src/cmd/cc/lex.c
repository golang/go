// Inferno utils/cc/lex.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/lex.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
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

#include	"cc.h"
#include	"y.tab.h"

#ifndef	CPP
#define	CPP	"/bin/cpp"
#endif

int
systemtype(int sys)
{
#ifdef __MINGW32__
	return sys&Windows;
#else
	return sys&Plan9;
#endif
}

int
pathchar(void)
{
	return '/';
}

/*
 * known debug flags
 *	-a		acid declaration output
 *	-A		!B
 *	-B		non ANSI
 *	-d		print declarations
 *	-D name		define
 *	-F		format specification check
 *	-i		print initialization
 *	-I path		include
 *	-l		generate little-endian code
 *	-L		print every NAME symbol
 *	-M		constant multiplication
 *	-m		print add/sub/mul trees
 *	-n		print acid to file (%.c=%.acid) (with -a or -aa)
 *	-o file		output file
 *	-p		use standard cpp ANSI preprocessor (not on windows)
 *	-r		print registerization
 *	-s		print structure offsets (with -a or -aa)
 *	-S		print assembly
 *	-t		print type trees
 *	-V		enable void* conversion warnings
 *	-v		verbose printing
 *	-w		print warnings
 *	-X		abort on error
 *	-.		Inhibit search for includes in source directory
 */

void
main(int argc, char *argv[])
{
	char *defs[50], *p;
	int nproc, nout, i, c, ndef;

	memset(debug, 0, sizeof(debug));
	tinit();
	cinit();
	ginit();
	arginit();

	tufield = simplet((1L<<tfield->etype) | BUNSIGNED);
	ndef = 0;
	outfile = 0;
	include[ninclude++] = ".";
	ARGBEGIN {
	default:
		c = ARGC();
		if(c >= 0 && c < sizeof(debug))
			debug[c]++;
		break;

	case 'l':			/* for little-endian mips */
		if(thechar != 'v'){
			print("can only use -l with vc");
			errorexit();
		}
		thechar = '0';
		thestring = "spim";
		break;

	case 'o':
		outfile = ARGF();
		break;

	case 'D':
		p = ARGF();
		if(p) {
			defs[ndef++] = p;
			dodefine(p);
		}
		break;

	case 'I':
		p = ARGF();
		setinclude(p);
		break;
	} ARGEND
	if(argc < 1 && outfile == 0) {
		print("usage: %cc [-options] files\n", thechar);
		errorexit();
	}
	if(argc > 1 && systemtype(Windows)){
		print("can't compile multiple files on windows\n");
		errorexit();
	}
	if(argc > 1 && !systemtype(Windows)) {
		nproc = 1;
		/*
		 * if we're writing acid to standard output, don't compile
		 * concurrently, to avoid interleaving output.
		 */
		if(((!debug['a'] && !debug['Z']) || debug['n']) &&
		    (p = getenv("NPROC")) != nil)
			nproc = atol(p);	/* */
		c = 0;
		nout = 0;
		for(;;) {
			Waitmsg *w;

			while(nout < nproc && argc > 0) {
				i = fork();
				if(i < 0) {
					print("cannot create a process\n");
					errorexit();
				}
				if(i == 0) {
					fprint(2, "%s:\n", *argv);
					if (compile(*argv, defs, ndef))
						errorexit();
					exits(0);
				}
				nout++;
				argc--;
				argv++;
			}
			w = wait();
			if(w == nil) {
				if(c)
					errorexit();
				exits(0);
			}
			if(w->msg[0])
				c++;
			nout--;
		}
	}

	if(argc == 0)
		c = compile("stdin", defs, ndef);
	else
		c = compile(argv[0], defs, ndef);

	if(c)
		errorexit();
	exits(0);
}

int
compile(char *file, char **defs, int ndef)
{
	char *ofile, incfile[20];
	char *p, *av[100], opt[256];
	int i, c, fd[2];
	static int first = 1;

	ofile = strdup(file);
	p = utfrrune(ofile, pathchar());
	if(p) {
		*p++ = 0;
		if(!debug['.'])
			include[0] = strdup(ofile);
	} else
		p = ofile;

	if(outfile == 0) {
		outfile = p;
		if(outfile) {
			if(p = utfrrune(outfile, '.'))
				if(p[1] == 'c' && p[2] == 0)
					p[0] = 0;
			p = utfrune(outfile, 0);
			if(debug['a'] && debug['n'])
				strcat(p, ".acid");
			else if(debug['Z'] && debug['n'])
				strcat(p, "_pickle.c");
			else {
				p[0] = '.';
				p[1] = thechar;
				p[2] = 0;
			}
		} else
			outfile = "/dev/null";
	}

	if(p = getenv("INCLUDE")) {
		setinclude(p);
	} else {
		if(systemtype(Plan9)) {
			sprint(incfile, "/%s/include", thestring);
			setinclude(strdup(incfile));
			setinclude("/sys/include");
		}
	}
	if (first)
		Binit(&diagbuf, 1, OWRITE);
	/*
	 * if we're writing acid to standard output, don't keep scratching
	 * outbuf.
	 */
	if((debug['a'] || debug['Z']) && !debug['n']) {
		if (first) {
			outfile = 0;
			Binit(&outbuf, dup(1, -1), OWRITE);
			dup(2, 1);
		}
	} else {
		c = create(outfile, OWRITE, 0664);
		if(c < 0) {
			diag(Z, "cannot open %s - %r", outfile);
			outfile = 0;
			errorexit();
		}
		Binit(&outbuf, c, OWRITE);
		outfile = strdup(outfile);
	}
	newio();
	first = 0;

	/* Use an ANSI preprocessor */
	if(debug['p']) {
		if(systemtype(Windows)) {
			diag(Z, "-p option not supported on windows");
			errorexit();
		}
		if(access(file, AREAD) < 0) {
			diag(Z, "%s does not exist", file);
			errorexit();
		}
		if(pipe(fd) < 0) {
			diag(Z, "pipe failed");
			errorexit();
		}
		switch(fork()) {
		case -1:
			diag(Z, "fork failed");
			errorexit();
		case 0:
			close(fd[0]);
			dup(fd[1], 1);
			close(fd[1]);
			av[0] = CPP;
			i = 1;
			if(debug['.']){
				sprint(opt, "-.");
				av[i++] = strdup(opt);
			}
			if(debug['+']) {
				sprint(opt, "-+");
				av[i++] = strdup(opt);
			}
			for(c = 0; c < ndef; c++)
				av[i++] = smprint("-D%s", defs[c]);
			for(c = 0; c < ninclude; c++)
				av[i++] = smprint("-I%s", include[c]);
			if(strcmp(file, "stdin") != 0)
				av[i++] = file;
			av[i] = 0;
			if(debug['p'] > 1) {
				for(c = 0; c < i; c++)
					fprint(2, "%s ", av[c]);
				fprint(2, "\n");
			}
			exec(av[0], av);
			fprint(2, "can't exec C preprocessor %s: %r\n", CPP);
			errorexit();
		default:
			close(fd[1]);
			newfile(file, fd[0]);
			break;
		}
	} else {
		if(strcmp(file, "stdin") == 0)
			newfile(file, 0);
		else
			newfile(file, -1);
	}
	yyparse();
	if(!debug['a'] && !debug['Z'])
		gclean();
	return nerrors;
}

void
errorexit(void)
{
	if(outfile)
		remove(outfile);
	exits("error");
}

void
pushio(void)
{
	Io *i;

	i = iostack;
	if(i == I) {
		yyerror("botch in pushio");
		errorexit();
	}
	i->p = fi.p;
	i->c = fi.c;
}

void
newio(void)
{
	Io *i;
	static int pushdepth = 0;

	i = iofree;
	if(i == I) {
		pushdepth++;
		if(pushdepth > 1000) {
			yyerror("macro/io expansion too deep");
			errorexit();
		}
		i = alloc(sizeof(*i));
	} else
		iofree = i->link;
	i->c = 0;
	i->f = -1;
	ionext = i;
}

void
newfile(char *s, int f)
{
	Io *i;

	if(debug['e'])
		print("%L: %s\n", lineno, s);

	i = ionext;
	i->link = iostack;
	iostack = i;
	i->f = f;
	if(f < 0)
		i->f = open(s, 0);
	if(i->f < 0) {
		yyerror("%cc: %r: %s", thechar, s);
		errorexit();
	}
	fi.c = 0;
	linehist(s, 0);
}

Sym*
slookup(char *s)
{

	strcpy(symb, s);
	return lookup();
}

Sym*
lookup(void)
{
	Sym *s;
	uint32 h;
	char *p;
	int c, n;

	h = 0;
	for(p=symb; *p;) {
		h = h * 3;
		h += *p++;
	}
	n = (p - symb) + 1;
	h &= 0xffffff;
	h %= NHASH;
	c = symb[0];
	for(s = hash[h]; s != S; s = s->link) {
		if(s->name[0] != c)
			continue;
		if(strcmp(s->name, symb) == 0)
			return s;
	}
	s = alloc(sizeof(*s));
	s->name = alloc(n);
	memmove(s->name, symb, n);
	s->link = hash[h];
	hash[h] = s;
	syminit(s);

	return s;
}

void
syminit(Sym *s)
{
	s->lexical = LNAME;
	s->block = 0;
	s->offset = 0;
	s->type = T;
	s->suetag = T;
	s->class = CXXX;
	s->aused = 0;
	s->sig = SIGNONE;
}

#define	EOF	(-1)
#define	IGN	(-2)
#define	ESC	(1<<20)
#define	GETC()	((--fi.c < 0)? filbuf(): (*fi.p++ & 0xff))

enum
{
	Numdec		= 1<<0,
	Numlong		= 1<<1,
	Numuns		= 1<<2,
	Numvlong	= 1<<3,
	Numflt		= 1<<4,
};

int32
yylex(void)
{
	vlong vv;
	int32 c, c1, t;
	char *cp;
	Rune rune;
	Sym *s;

	if(peekc != IGN) {
		c = peekc;
		peekc = IGN;
		goto l1;
	}
l0:
	c = GETC();

l1:
	if(c >= Runeself) {
		/*
		 * extension --
		 *	all multibyte runes are alpha
		 */
		cp = symb;
		goto talph;
	}
	if(isspace(c)) {
		if(c == '\n')
			lineno++;
		goto l0;
	}
	if(isalpha(c)) {
		cp = symb;
		if(c != 'L')
			goto talph;
		*cp++ = c;
		c = GETC();
		if(c == '\'') {
			/* L'x' */
			c = escchar('\'', 1, 0);
			if(c == EOF)
				c = '\'';
			c1 = escchar('\'', 1, 0);
			if(c1 != EOF) {
				yyerror("missing '");
				peekc = c1;
			}
			yylval.vval = convvtox(c, TUSHORT);
			return LUCONST;
		}
		if(c == '"') {
			goto caselq;
		}
		goto talph;
	}
	if(isdigit(c))
		goto tnum;
	switch(c)
	{

	case EOF:
		peekc = EOF;
		return -1;

	case '_':
		cp = symb;
		goto talph;

	case '#':
		domacro();
		goto l0;

	case '.':
		c1 = GETC();
		if(isdigit(c1)) {
			cp = symb;
			*cp++ = c;
			c = c1;
			c1 = 0;
			goto casedot;
		}
		break;

	case '"':
		strcpy(symb, "\"<string>\"");
		cp = alloc(0);
		c1 = 0;

		/* "..." */
		for(;;) {
			c = escchar('"', 0, 1);
			if(c == EOF)
				break;
			if(c & ESC) {
				cp = allocn(cp, c1, 1);
				cp[c1++] = c;
			} else {
				rune = c;
				c = runelen(rune);
				cp = allocn(cp, c1, c);
				runetochar(cp+c1, &rune);
				c1 += c;
			}
		}
		yylval.sval.l = c1;
		do {
			cp = allocn(cp, c1, 1);
			cp[c1++] = 0;
		} while(c1 & MAXALIGN);
		yylval.sval.s = cp;
		return LSTRING;

	caselq:
		/* L"..." */
		strcpy(symb, "\"L<string>\"");
		cp = alloc(0);
		c1 = 0;
		for(;;) {
			c = escchar('"', 1, 0);
			if(c == EOF)
				break;
			cp = allocn(cp, c1, sizeof(ushort));
			*(ushort*)(cp + c1) = c;
			c1 += sizeof(ushort);
		}
		yylval.sval.l = c1;
		do {
			cp = allocn(cp, c1, sizeof(ushort));
			*(ushort*)(cp + c1) = 0;
			c1 += sizeof(ushort);
		} while(c1 & MAXALIGN);
		yylval.sval.s = cp;
		return LLSTRING;

	case '\'':
		/* '.' */
		c = escchar('\'', 0, 0);
		if(c == EOF)
			c = '\'';
		c1 = escchar('\'', 0, 0);
		if(c1 != EOF) {
			yyerror("missing '");
			peekc = c1;
		}
		vv = c;
		yylval.vval = convvtox(vv, TUCHAR);
		if(yylval.vval != vv)
			yyerror("overflow in character constant: 0x%lx", c);
		else
		if(c & 0x80){
			nearln = lineno;
			warn(Z, "sign-extended character constant");
		}
		yylval.vval = convvtox(vv, TCHAR);
		return LCONST;

	case '/':
		c1 = GETC();
		if(c1 == '*') {
			for(;;) {
				c = getr();
				while(c == '*') {
					c = getr();
					if(c == '/')
						goto l0;
				}
				if(c == EOF) {
					yyerror("eof in comment");
					errorexit();
				}
			}
		}
		if(c1 == '/') {
			for(;;) {
				c = getr();
				if(c == '\n')
					goto l0;
				if(c == EOF) {
					yyerror("eof in comment");
					errorexit();
				}
			}
		}
		if(c1 == '=')
			return LDVE;
		break;

	case '*':
		c1 = GETC();
		if(c1 == '=')
			return LMLE;
		break;

	case '%':
		c1 = GETC();
		if(c1 == '=')
			return LMDE;
		break;

	case '+':
		c1 = GETC();
		if(c1 == '+')
			return LPP;
		if(c1 == '=')
			return LPE;
		break;

	case '-':
		c1 = GETC();
		if(c1 == '-')
			return LMM;
		if(c1 == '=')
			return LME;
		if(c1 == '>')
			return LMG;
		break;

	case '>':
		c1 = GETC();
		if(c1 == '>') {
			c = LRSH;
			c1 = GETC();
			if(c1 == '=')
				return LRSHE;
			break;
		}
		if(c1 == '=')
			return LGE;
		break;

	case '<':
		c1 = GETC();
		if(c1 == '<') {
			c = LLSH;
			c1 = GETC();
			if(c1 == '=')
				return LLSHE;
			break;
		}
		if(c1 == '=')
			return LLE;
		break;

	case '=':
		c1 = GETC();
		if(c1 == '=')
			return LEQ;
		break;

	case '!':
		c1 = GETC();
		if(c1 == '=')
			return LNE;
		break;

	case '&':
		c1 = GETC();
		if(c1 == '&')
			return LANDAND;
		if(c1 == '=')
			return LANDE;
		break;

	case '|':
		c1 = GETC();
		if(c1 == '|')
			return LOROR;
		if(c1 == '=')
			return LORE;
		break;

	case '^':
		c1 = GETC();
		if(c1 == '=')
			return LXORE;
		break;

	default:
		return c;
	}
	peekc = c1;
	return c;

talph:
	/*
	 * cp is set to symb and some
	 * prefix has been stored
	 */
	for(;;) {
		if(c >= Runeself) {
			for(c1=0;;) {
				cp[c1++] = c;
				if(fullrune(cp, c1))
					break;
				c = GETC();
			}
			cp += c1;
			c = GETC();
			continue;
		}
		if(!isalnum(c) && c != '_')
			break;
		*cp++ = c;
		c = GETC();
	}
	*cp = 0;
	if(debug['L'])
		print("%L: %s\n", lineno, symb);
	peekc = c;
	s = lookup();
	if(s->macro) {
		newio();
		cp = ionext->b;
		macexpand(s, cp);
		pushio();
		ionext->link = iostack;
		iostack = ionext;
		fi.p = cp;
		fi.c = strlen(cp);
		if(peekc != IGN) {
			cp[fi.c++] = peekc;
			cp[fi.c] = 0;
			peekc = IGN;
		}
		goto l0;
	}
	yylval.sym = s;
	if(s->class == CTYPEDEF || s->class == CTYPESTR)
		return LTYPE;
	return s->lexical;

tnum:
	c1 = 0;
	cp = symb;
	if(c != '0') {
		c1 |= Numdec;
		for(;;) {
			*cp++ = c;
			c = GETC();
			if(isdigit(c))
				continue;
			goto dc;
		}
	}
	*cp++ = c;
	c = GETC();
	if(c == 'x' || c == 'X')
		for(;;) {
			*cp++ = c;
			c = GETC();
			if(isdigit(c))
				continue;
			if(c >= 'a' && c <= 'f')
				continue;
			if(c >= 'A' && c <= 'F')
				continue;
			if(cp == symb+2)
				yyerror("malformed hex constant");
			goto ncu;
		}
	if(c < '0' || c > '7')
		goto dc;
	for(;;) {
		if(c >= '0' && c <= '7') {
			*cp++ = c;
			c = GETC();
			continue;
		}
		goto ncu;
	}

dc:
	if(c == '.')
		goto casedot;
	if(c == 'e' || c == 'E')
		goto casee;

ncu:
	if((c == 'U' || c == 'u') && !(c1 & Numuns)) {
		c = GETC();
		c1 |= Numuns;
		goto ncu;
	}
	if((c == 'L' || c == 'l') && !(c1 & Numvlong)) {
		c = GETC();
		if(c1 & Numlong)
			c1 |= Numvlong;
		c1 |= Numlong;
		goto ncu;
	}
	*cp = 0;
	peekc = c;
	if(mpatov(symb, &yylval.vval))
		yyerror("overflow in constant");

	vv = yylval.vval;
	if(c1 & Numvlong) {
		if((c1 & Numuns) || convvtox(vv, TVLONG) < 0) {
			c = LUVLCONST;
			t = TUVLONG;
			goto nret;
		}
		c = LVLCONST;
		t = TVLONG;
		goto nret;
	}
	if(c1 & Numlong) {
		if((c1 & Numuns) || convvtox(vv, TLONG) < 0) {
			c = LULCONST;
			t = TULONG;
			goto nret;
		}
		c = LLCONST;
		t = TLONG;
		goto nret;
	}
	if((c1 & Numuns) || convvtox(vv, TINT) < 0) {
		c = LUCONST;
		t = TUINT;
		goto nret;
	}
	c = LCONST;
	t = TINT;
	goto nret;

nret:
	yylval.vval = convvtox(vv, t);
	if(yylval.vval != vv){
		nearln = lineno;
		warn(Z, "truncated constant: %T %s", types[t], symb);
	}
	return c;

casedot:
	for(;;) {
		*cp++ = c;
		c = GETC();
		if(!isdigit(c))
			break;
	}
	if(c != 'e' && c != 'E')
		goto caseout;

casee:
	*cp++ = 'e';
	c = GETC();
	if(c == '+' || c == '-') {
		*cp++ = c;
		c = GETC();
	}
	if(!isdigit(c))
		yyerror("malformed fp constant exponent");
	while(isdigit(c)) {
		*cp++ = c;
		c = GETC();
	}

caseout:
	if(c == 'L' || c == 'l') {
		c = GETC();
		c1 |= Numlong;
	} else
	if(c == 'F' || c == 'f') {
		c = GETC();
		c1 |= Numflt;
	}
	*cp = 0;
	peekc = c;
	yylval.dval = strtod(symb, nil);
	if(isInf(yylval.dval, 1) || isInf(yylval.dval, -1)) {
		yyerror("overflow in float constant");
		yylval.dval = 0;
	}
	if(c1 & Numflt)
		return LFCONST;
	return LDCONST;
}

/*
 * convert a string, s, to vlong in *v
 * return conversion overflow.
 * required syntax is [0[x]]d*
 */
int
mpatov(char *s, vlong *v)
{
	vlong n, nn;
	int c;

	n = 0;
	c = *s;
	if(c == '0')
		goto oct;
	while(c = *s++) {
		if(c >= '0' && c <= '9')
			nn = n*10 + c-'0';
		else
			goto bad;
		if(n < 0 && nn >= 0)
			goto bad;
		n = nn;
	}
	goto out;

oct:
	s++;
	c = *s;
	if(c == 'x' || c == 'X')
		goto hex;
	while(c = *s++) {
		if(c >= '0' || c <= '7')
			nn = n*8 + c-'0';
		else
			goto bad;
		if(n < 0 && nn >= 0)
			goto bad;
		n = nn;
	}
	goto out;

hex:
	s++;
	while(c = *s++) {
		if(c >= '0' && c <= '9')
			c += 0-'0';
		else
		if(c >= 'a' && c <= 'f')
			c += 10-'a';
		else
		if(c >= 'A' && c <= 'F')
			c += 10-'A';
		else
			goto bad;
		nn = n*16 + c;
		if(n < 0 && nn >= 0)
			goto bad;
		n = nn;
	}
out:
	*v = n;
	return 0;

bad:
	*v = ~0;
	return 1;
}

int
getc(void)
{
	int c;

	if(peekc != IGN) {
		c = peekc;
		peekc = IGN;
	} else
		c = GETC();
	if(c == '\n')
		lineno++;
	if(c == EOF) {
		yyerror("End of file");
		errorexit();
	}
	return c;
}

int32
getr(void)
{
	int c, i;
	char str[UTFmax+1];
	Rune rune;


	c = getc();
	if(c < Runeself)
		return c;
	i = 0;
	str[i++] = c;

loop:
	c = getc();
	str[i++] = c;
	if(!fullrune(str, i))
		goto loop;
	c = chartorune(&rune, str);
	if(rune == Runeerror && c == 1) {
		nearln = lineno;
		diag(Z, "illegal rune in string");
		for(c=0; c<i; c++)
			print(" %.2x", *(uchar*)(str+c));
		print("\n");
	}
	return rune;
}

int
getnsc(void)
{
	int c;

	if(peekc != IGN) {
		c = peekc;
		peekc = IGN;
	} else
		c = GETC();
	for(;;) {
		if(!isspace(c))
			return c;
		if(c == '\n') {
			lineno++;
			return c;
		}
		c = GETC();
	}
}

void
unget(int c)
{

	peekc = c;
	if(c == '\n')
		lineno--;
}

int32
escchar(int32 e, int longflg, int escflg)
{
	int32 c, l;
	int i;

loop:
	c = getr();
	if(c == '\n') {
		yyerror("newline in string");
		return EOF;
	}
	if(c != '\\') {
		if(c == e)
			c = EOF;
		return c;
	}
	c = getr();
	if(c == 'x') {
		/*
		 * note this is not ansi,
		 * supposed to only accept 2 hex
		 */
		i = 2;
		if(longflg)
			i = 4;
		l = 0;
		for(; i>0; i--) {
			c = getc();
			if(c >= '0' && c <= '9') {
				l = l*16 + c-'0';
				continue;
			}
			if(c >= 'a' && c <= 'f') {
				l = l*16 + c-'a' + 10;
				continue;
			}
			if(c >= 'A' && c <= 'F') {
				l = l*16 + c-'A' + 10;
				continue;
			}
			unget(c);
			break;
		}
		if(escflg)
			l |= ESC;
		return l;
	}
	if(c >= '0' && c <= '7') {
		/*
		 * note this is not ansi,
		 * supposed to only accept 3 oct
		 */
		i = 2;
		if(longflg)
			i = 5;
		l = c - '0';
		for(; i>0; i--) {
			c = getc();
			if(c >= '0' && c <= '7') {
				l = l*8 + c-'0';
				continue;
			}
			unget(c);
		}
		if(escflg)
			l |= ESC;
		return l;
	}
	switch(c)
	{
	case '\n':	goto loop;
	case 'n':	return '\n';
	case 't':	return '\t';
	case 'b':	return '\b';
	case 'r':	return '\r';
	case 'f':	return '\f';
	case 'a':	return '\a';
	case 'v':	return '\v';
	}
	return c;
}

struct
{
	char	*name;
	ushort	lexical;
	ushort	type;
} itab[] =
{
	"auto",		LAUTO,		0,
	"break",	LBREAK,		0,
	"case",		LCASE,		0,
	"char",		LCHAR,		TCHAR,
	"const",	LCONSTNT,	0,
	"continue",	LCONTINUE,	0,
	"default",	LDEFAULT,	0,
	"do",		LDO,		0,
	"double",	LDOUBLE,	TDOUBLE,
	"else",		LELSE,		0,
	"enum",		LENUM,		0,
	"extern",	LEXTERN,	0,
	"float",	LFLOAT,		TFLOAT,
	"for",		LFOR,		0,
	"goto",		LGOTO,		0,
	"if",		LIF,		0,
	"inline",	LINLINE,	0,
	"int",		LINT,		TINT,
	"long",		LLONG,		TLONG,
	"register",	LREGISTER,	0,
	"restrict",	LRESTRICT,	0,
	"return",	LRETURN,	0,
	"SET",		LSET,		0,
	"short",	LSHORT,		TSHORT,
	"signed",	LSIGNED,	0,
	"signof",	LSIGNOF,	0,
	"sizeof",	LSIZEOF,	0,
	"static",	LSTATIC,	0,
	"struct",	LSTRUCT,	0,
	"switch",	LSWITCH,	0,
	"typedef",	LTYPEDEF,	0,
	"typestr",	LTYPESTR,	0,
	"union",	LUNION,		0,
	"unsigned",	LUNSIGNED,	0,
	"USED",		LUSED,		0,
	"void",		LVOID,		TVOID,
	"volatile",	LVOLATILE,	0,
	"while",	LWHILE,		0,
	0
};

void
cinit(void)
{
	Sym *s;
	int i;
	Type *t;

	nerrors = 0;
	lineno = 1;
	iostack = I;
	iofree = I;
	peekc = IGN;
	nhunk = 0;

	types[TXXX] = T;
	types[TCHAR] = typ(TCHAR, T);
	types[TUCHAR] = typ(TUCHAR, T);
	types[TSHORT] = typ(TSHORT, T);
	types[TUSHORT] = typ(TUSHORT, T);
	types[TINT] = typ(TINT, T);
	types[TUINT] = typ(TUINT, T);
	types[TLONG] = typ(TLONG, T);
	types[TULONG] = typ(TULONG, T);
	types[TVLONG] = typ(TVLONG, T);
	types[TUVLONG] = typ(TUVLONG, T);
	types[TFLOAT] = typ(TFLOAT, T);
	types[TDOUBLE] = typ(TDOUBLE, T);
	types[TVOID] = typ(TVOID, T);
	types[TENUM] = typ(TENUM, T);
	types[TFUNC] = typ(TFUNC, types[TINT]);
	types[TIND] = typ(TIND, types[TVOID]);

	for(i=0; i<NHASH; i++)
		hash[i] = S;
	for(i=0; itab[i].name; i++) {
		s = slookup(itab[i].name);
		s->lexical = itab[i].lexical;
		if(itab[i].type != 0)
			s->type = types[itab[i].type];
	}
	blockno = 0;
	autobn = 0;
	autoffset = 0;

	t = typ(TARRAY, types[TCHAR]);
	t->width = 0;
	symstring = slookup(".string");
	symstring->class = CSTATIC;
	symstring->type = t;

	t = typ(TARRAY, types[TCHAR]);
	t->width = 0;

	nodproto = new(OPROTO, Z, Z);
	dclstack = D;

	pathname = allocn(pathname, 0, 100);
	if(getwd(pathname, 99) == 0) {
		pathname = allocn(pathname, 100, 900);
		if(getwd(pathname, 999) == 0)
			strcpy(pathname, "/???");
	}

	fmtinstall('O', Oconv);
	fmtinstall('T', Tconv);
	fmtinstall('F', FNconv);
	fmtinstall('L', Lconv);
	fmtinstall('Q', Qconv);
	fmtinstall('|', VBconv);
}

int
filbuf(void)
{
	Io *i;

loop:
	i = iostack;
	if(i == I)
		return EOF;
	if(i->f < 0)
		goto pop;
	fi.c = read(i->f, i->b, BUFSIZ) - 1;
	if(fi.c < 0) {
		close(i->f);
		linehist(0, 0);
		goto pop;
	}
	fi.p = i->b + 1;
	return i->b[0] & 0xff;

pop:
	iostack = i->link;
	i->link = iofree;
	iofree = i;
	i = iostack;
	if(i == I)
		return EOF;
	fi.p = i->p;
	fi.c = i->c;
	if(--fi.c < 0)
		goto loop;
	return *fi.p++ & 0xff;
}

int
Oconv(Fmt *fp)
{
	int a;

	a = va_arg(fp->args, int);
	if(a < OXXX || a > OEND)
		return fmtprint(fp, "***badO %d***", a);

	return fmtstrcpy(fp, onames[a]);
}

int
Lconv(Fmt *fp)
{
	char str[STRINGSZ], s[STRINGSZ];
	Hist *h;
	struct
	{
		Hist*	incl;	/* start of this include file */
		int32	idel;	/* delta line number to apply to include */
		Hist*	line;	/* start of this #line directive */
		int32	ldel;	/* delta line number to apply to #line */
	} a[HISTSZ];
	int32 l, d;
	int i, n;

	l = va_arg(fp->args, int32);
	n = 0;
	for(h = hist; h != H; h = h->link) {
		if(l < h->line)
			break;
		if(h->name) {
			if(h->offset != 0) {		/* #line directive, not #pragma */
				if(n > 0 && n < HISTSZ && h->offset >= 0) {
					a[n-1].line = h;
					a[n-1].ldel = h->line - h->offset + 1;
				}
			} else {
				if(n < HISTSZ) {	/* beginning of file */
					a[n].incl = h;
					a[n].idel = h->line;
					a[n].line = 0;
				}
				n++;
			}
			continue;
		}
		n--;
		if(n > 0 && n < HISTSZ) {
			d = h->line - a[n].incl->line;
			a[n-1].ldel += d;
			a[n-1].idel += d;
		}
	}
	if(n > HISTSZ)
		n = HISTSZ;
	str[0] = 0;
	for(i=n-1; i>=0; i--) {
		if(i != n-1) {
			if(fp->flags & ~(FmtWidth|FmtPrec))	/* BUG ROB - was f3 */
				break;
			strcat(str, " ");
		}
		if(a[i].line)
			snprint(s, STRINGSZ, "%s:%ld[%s:%ld]",
				a[i].line->name, l-a[i].ldel+1,
				a[i].incl->name, l-a[i].idel+1);
		else
			snprint(s, STRINGSZ, "%s:%ld",
				a[i].incl->name, l-a[i].idel+1);
		if(strlen(s)+strlen(str) >= STRINGSZ-10)
			break;
		strcat(str, s);
		l = a[i].incl->line - 1;	/* now print out start of this file */
	}
	if(n == 0)
		strcat(str, "<eof>");
	return fmtstrcpy(fp, str);
}

int
Tconv(Fmt *fp)
{
	char str[STRINGSZ+20], s[STRINGSZ+20];
	Type *t, *t1;
	int et;
	int32 n;

	str[0] = 0;
	for(t = va_arg(fp->args, Type*); t != T; t = t->link) {
		et = t->etype;
		if(str[0])
			strcat(str, " ");
		if(t->garb&~GINCOMPLETE) {
			sprint(s, "%s ", gnames[t->garb&~GINCOMPLETE]);
			if(strlen(str) + strlen(s) < STRINGSZ)
				strcat(str, s);
		}
		sprint(s, "%s", tnames[et]);
		if(strlen(str) + strlen(s) < STRINGSZ)
			strcat(str, s);
		if(et == TFUNC && (t1 = t->down)) {
			sprint(s, "(%T", t1);
			if(strlen(str) + strlen(s) < STRINGSZ)
				strcat(str, s);
			while(t1 = t1->down) {
				sprint(s, ", %T", t1);
				if(strlen(str) + strlen(s) < STRINGSZ)
					strcat(str, s);
			}
			if(strlen(str) + strlen(s) < STRINGSZ)
				strcat(str, ")");
		}
		if(et == TARRAY) {
			n = t->width;
			if(t->link && t->link->width)
				n /= t->link->width;
			sprint(s, "[%ld]", n);
			if(strlen(str) + strlen(s) < STRINGSZ)
				strcat(str, s);
		}
		if(t->nbits) {
			sprint(s, " %d:%d", t->shift, t->nbits);
			if(strlen(str) + strlen(s) < STRINGSZ)
				strcat(str, s);
		}
		if(typesu[et]) {
			if(t->tag) {
				strcat(str, " ");
				if(strlen(str) + strlen(t->tag->name) < STRINGSZ)
					strcat(str, t->tag->name);
			} else
				strcat(str, " {}");
			break;
		}
	}
	return fmtstrcpy(fp, str);
}

int
FNconv(Fmt *fp)
{
	char *str;
	Node *n;

	n = va_arg(fp->args, Node*);
	str = "<indirect>";
	if(n != Z && (n->op == ONAME || n->op == ODOT || n->op == OELEM))
		str = n->sym->name;
	return fmtstrcpy(fp, str);
}

int
Qconv(Fmt *fp)
{
	char str[STRINGSZ+20], *s;
	int32 b;
	int i;

	str[0] = 0;
	for(b = va_arg(fp->args, int32); b;) {
		i = bitno(b);
		if(str[0])
			strcat(str, " ");
		s = qnames[i];
		if(strlen(str) + strlen(s) >= STRINGSZ)
			break;
		strcat(str, s);
		b &= ~(1L << i);
	}
	return fmtstrcpy(fp, str);
}

int
VBconv(Fmt *fp)
{
	char str[STRINGSZ];
	int i, n, t, pc;

	n = va_arg(fp->args, int);
	pc = 0;	/* BUG: was printcol */
	i = 0;
	while(pc < n) {
		t = (pc+4) & ~3;
		if(t <= n) {
			str[i++] = '\t';
			pc = t;
			continue;
		}
		str[i++] = ' ';
		pc++;
	}
	str[i] = 0;

	return fmtstrcpy(fp, str);
}

void
setinclude(char *p)
{
	int i;

	if(*p != 0) {
		for(i=1; i < ninclude; i++)
			if(strcmp(p, include[i]) == 0)
				return;

		if(i >= ninclude)
			include[ninclude++] = p;

		if(ninclude > nelem(include)) {
			diag(Z, "ninclude too small %d", nelem(include));
			exits("ninclude");
		}

	}
}

void*
alloc(int32 n)
{
	void *p;

	p = malloc(n);
	if(p == nil) {
		print("alloc out of mem\n");
		exit(1);
	}
	memset(p, 0, n);
	return p;
}

void*
allocn(void *p, int32 n, int32 d)
{
	if(p == nil)
		return alloc(n+d);
	p = realloc(p, n+d);
	if(p == nil) {
		print("allocn out of mem\n");
		exit(1);
	}
	if(d > 0)
		memset((char*)p+n, 0, d);
	return p;
}
