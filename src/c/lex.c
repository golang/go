// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


#define		EXTERN
#include	"go.h"
#include	"y.tab.h"

#define	DBG	if(!debug['x']);else print
enum
{
	EOF		= -1,
};

int
main(int argc, char *argv[])
{
	int c;

	outfile = nil;
	package = "____";
	ARGBEGIN {
	default:
		c = ARGC();
		if(c >= 0 && c < sizeof(debug))
			debug[c]++;
		break;

	case 'o':
		outfile = ARGF();
		break;

	case 'k':
		package = ARGF();
		break;
	} ARGEND

	if(argc != 1)
		goto usage;

	fmtinstall('O', Oconv);		// node opcodes
	fmtinstall('E', Econv);		// etype opcodes
	fmtinstall('J', Jconv);		// all the node flags
	fmtinstall('S', Sconv);		// sym pointer
	fmtinstall('T', Tconv);		// type pointer
	fmtinstall('N', Nconv);		// node pointer
	fmtinstall('Z', Zconv);		// escaped string
	lexinit();

	curio.infile = argv[0];

	curio.bin = Bopen(curio.infile, OREAD);
	if(curio.bin == nil)
		fatal("cant open: %s", curio.infile);

	externdcl = mal(sizeof(*externdcl));
	externdcl->back = externdcl;
	dclcontext = PEXTERN;

	exportlist = mal(sizeof(*exportlist));
	exportlist->back = exportlist;

	// function field skeleton
	fskel = nod(OLIST, N, nod(OLIST, N, N));
	fskel->left = nod(ODCLFIELD, N, N);
	fskel->right->left = nod(ODCLFIELD, N, N);
	fskel->right->right = nod(ODCLFIELD, N, N);

	curio.peekc = 0;
	curio.lineno = 1;
	nerrors = 0;
	yyparse();
	if(nerrors == 0) {
		dumpobj();
	}

	Bterm(curio.bin);
	if(bout != nil)
		Bterm(bout);

	if(nerrors)
		errorexit();

	myexit(0);
	return 0;

usage:
	print("flags:\n");
	print("  -d print declarations\n");
	print("  -f print stack frame structure\n");
	print("  -k name specify package name\n");
	print("  -o file specify output file\n");
	print("  -p print the assembly language\n");
	print("  -w print the parse tree after typing\n");
	print("  -x print lex tokens\n");
	print("  -h panic on an error\n");
	myexit(0);
	return 0;
}

void
importfile(Val *f)
{
	Biobuf *imp;
	long c;

	if(f->ctype != CTSTR) {
		yyerror("import statement not a string");
		return;
	}
	snprint(namebuf, sizeof(namebuf), "%Z.go.c", f->sval);

	imp = Bopen(namebuf, OREAD);
	if(imp == nil) {
		yyerror("cant open import: %s", namebuf);
		return;
	}

	/*
	 * position the input right
	 * after (( and return
	 */
	pushedio = curio;
	curio.bin = imp;
	curio.lineno = 1;
	curio.peekc = 0;
	curio.infile = strdup(namebuf);
	for(;;) {
		c = getc();
		if(c == EOF)
			break;
		if(c != '(')
			continue;
		c = getc();
		if(c == EOF)
			break;
		if(c != '(')
			continue;
		return;
	}
	yyerror("no import in: %Z", f->sval);
	unimportfile();
}

void
unimportfile(void)
{
	if(curio.bin != nil && pushedio.bin != nil) {
		Bterm(curio.bin);
		curio = pushedio;
		pushedio.bin = nil;
	}
}

long
yylex(void)
{
	long c, c1;
	char *cp;
	Rune rune;
	int escflag;
	Sym *s;

l0:
	c = getc();
	if(isspace(c))
		goto l0;

	if(c >= Runeself) {
		/* all multibyte runes are alpha */
		cp = namebuf;
		goto talph;
	}

	if(isalpha(c)) {
		cp = namebuf;
		goto talph;
	}

	if(isdigit(c))
		goto tnum;

	switch(c) {
	case EOF:
		ungetc(EOF);
		return -1;

	case '_':
		cp = namebuf;
		goto talph;

	case '.':
		c1 = getc();
		if(isdigit(c1)) {
			cp = namebuf;
			*cp++ = c;
			c = c1;
			c1 = 0;
			goto casedot;
		}
		break;

	case '"':
		/* "..." */
		strcpy(namebuf, "\"<string>\"");
		cp = mal(sizeof(long));
		c1 = 4;

	caseq:
		for(;;) {
			c = escchar('"', &escflag);
			if(c == EOF)
				break;
			if(escflag) {
				cp = remal(cp, c1, 1);
				cp[c1++] = c;
			} else {
				rune = c;
				c = runelen(rune);
				cp = remal(cp, c1, c);
				runetochar(cp+c1, &rune);
				c1 += c;
			}
		}
		goto catem;

	case '`':
		/* `...` */
		strcpy(namebuf, "`<string>`");
		cp = mal(sizeof(long));
		c1 = 4;

	casebq:
		for(;;) {
			c = getc();
			if(c == EOF || c == '`')
				break;
			cp = remal(cp, c1, 1);
			cp[c1++] = c;
		}

	catem:
		for(;;) {
			/* it takes 2 peekc's to skip comments */
			c = getc();
			if(isspace(c))
				continue;
			if(c == '"')
				goto caseq;
			if(c == '`')
				goto casebq;
			ungetc(c);
			break;
		}

		*(long*)cp = c1-4;	// length
		do {
			cp = remal(cp, c1, 1);
			cp[c1++] = 0;
		} while(c1 & MAXALIGN);
		yylval.val.sval = (String*)cp;
		yylval.val.ctype = CTSTR;
		DBG("lex: string literal\n");
		return LLITERAL;

	case '\'':
		/* '.' */
		c = escchar('\'', &escflag);
		if(c == EOF)
			c = '\'';
		c1 = escchar('\'', &escflag);
		if(c1 != EOF) {
			yyerror("missing '");
			ungetc(c1);
		}
		yylval.val.vval = c;
		yylval.val.ctype = CTINT;
		DBG("lex: codepoint literal\n");
		return LLITERAL;

	case '/':
		c1 = getc();
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
		if(c1 == '=') {
			c = ODIV;
			goto asop;
		}
		break;

	case ':':
		c1 = getc();
		if(c1 == '=') {
			c = LCOLAS;
			goto lx;
		}
		break;

	case '*':
		c1 = getc();
		if(c1 == '=') {
			c = OMUL;
			goto asop;
		}
		break;

	case '%':
		c1 = getc();
		if(c1 == '=') {
			c = OMOD;
			goto asop;
		}
		break;

	case '+':
		c1 = getc();
		if(c1 == '+') {
			c = LINC;
			goto lx;
		}
		if(c1 == '=') {
			c = OADD;
			goto asop;
		}
		break;

	case '-':
		c1 = getc();
		if(c1 == '-') {
			c = LDEC;
			goto lx;
		}
		if(c1 == '=') {
			c = OSUB;
			goto asop;
		}
		break;

	case '>':
		c1 = getc();
		if(c1 == '>') {
			c = LRSH;
			c1 = getc();
			if(c1 == '=') {
				c = ORSH;
				goto asop;
			}
			break;
		}
		if(c1 == '=') {
			c = LGE;
			goto lx;
		}
		c = LGT;
		break;

	case '<':
		c1 = getc();
		if(c1 == '<') {
			c = LLSH;
			c1 = getc();
			if(c1 == '=') {
				c = OLSH;
				goto asop;
			}
			break;
		}
		if(c1 == '=') {
			c = LLE;
			goto lx;
		}
		c = LLT;
		break;

	case '=':
		c1 = getc();
		if(c1 == '=') {
			c = LEQ;
			goto lx;
		}
		break;

	case '!':
		c1 = getc();
		if(c1 == '=') {
			c = LNE;
			goto lx;
		}
		break;

	case '&':
		c1 = getc();
		if(c1 == '&') {
			c = LANDAND;
			goto lx;
		}
		if(c1 == '=') {
			c = OAND;
			goto asop;
		}
		break;

	case '|':
		c1 = getc();
		if(c1 == '|') {
			c = LOROR;
			goto lx;
		}
		if(c1 == '=') {
			c = OOR;
			goto asop;
		}
		break;

	case '^':
		c1 = getc();
		if(c1 == '=') {
			c = OXOR;
			goto asop;
		}
		break;

	default:
		goto lx;
	}
	ungetc(c1);

lx:
	if(c > 0xff)
		DBG("lex: TOKEN %s\n", lexname(c));
	else
		DBG("lex: TOKEN '%c'\n", c);
	return c;

asop:
	yylval.val.vval = c;	// rathole to hold which asop
	DBG("lex: TOKEN ASOP %c\n", c);
	return LASOP;

talph:
	/*
	 * cp is set to namebuf and some
	 * prefix has been stored
	 */
	for(;;) {
		if(c >= Runeself) {
			for(c1=0;;) {
				cp[c1++] = c;
				if(fullrune(cp, c1))
					break;
				c = getc();
			}
			cp += c1;
			c = getc();
			continue;
		}
		if(!isalnum(c) && c != '_')
			break;
		*cp++ = c;
		c = getc();
	}
	*cp = 0;
	ungetc(c);

	s = lookup(namebuf);
	if(s->lexical == LIGNORE)
		goto l0;

	if(context != nil) {
		s = pkglookup(s->name, context);
		if(s->lexical == LIGNORE)
			goto l0;
	}

	DBG("lex: %S %s\n", s, lexname(s->lexical));
	yylval.sym = s;
	if(s->lexical == LBASETYPE)
		return LATYPE;
	return s->lexical;

tnum:
	c1 = 0;
	cp = namebuf;
	if(c != '0') {
		for(;;) {
			*cp++ = c;
			c = getc();
			if(isdigit(c))
				continue;
			goto dc;
		}
	}
	*cp++ = c;
	c = getc();
	if(c == 'x' || c == 'X')
		for(;;) {
			*cp++ = c;
			c = getc();
			if(isdigit(c))
				continue;
			if(c >= 'a' && c <= 'f')
				continue;
			if(c >= 'A' && c <= 'F')
				continue;
			if(cp == namebuf+2)
				yyerror("malformed hex constant");
			goto ncu;
		}
	if(c < '0' || c > '7')
		goto dc;
	for(;;) {
		if(c >= '0' && c <= '7') {
			*cp++ = c;
			c = getc();
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
	*cp = 0;
	ungetc(c);
	if(mpatov(namebuf, &yylval.val.vval)) {
		yyerror("overflow in constant");
		yylval.val.vval = 0;
	}
	yylval.val.ctype = CTINT;
	DBG("lex: integer literal\n");
	return LLITERAL;

casedot:
	for(;;) {
		*cp++ = c;
		c = getc();
		if(!isdigit(c))
			break;
	}
	if(c != 'e' && c != 'E')
		goto caseout;

casee:
	*cp++ = 'e';
	c = getc();
	if(c == '+' || c == '-') {
		*cp++ = c;
		c = getc();
	}
	if(!isdigit(c))
		yyerror("malformed fp constant exponent");
	while(isdigit(c)) {
		*cp++ = c;
		c = getc();
	}

caseout:
	*cp = 0;
	ungetc(c);
	if(mpatof(namebuf, &yylval.val.dval)) {
		yyerror("overflow in float constant");
		yylval.val.dval = 0;
	}
	yylval.val.ctype = CTFLT;
	DBG("lex: floating literal\n");
	return LLITERAL;
}

int
getc(void)
{
	int c;

	c = curio.peekc;
	if(c != 0) {
		curio.peekc = 0;
		if(c == '\n')
			curio.lineno++;
		return c;
	}

	c = Bgetc(curio.bin);
	switch(c) {
	case 0:
	case EOF:
		return EOF;

	case '\n':
		curio.lineno++;
		break;

	}
	return c;
}

void
ungetc(int c)
{
	curio.peekc = c;
	if(c == '\n')
		curio.lineno--;
}

long
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
		yyerror("illegal rune in string");
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

	c = getc();
	for(;;) {
		if(!isspace(c))
			return c;
		if(c == '\n') {
			curio.lineno++;
			return c;
		}
		c = getc();
	}
	return 0;
}


long
escchar(long e, int *escflg)
{
	long c, l;
	int i;

	*escflg = 0;

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
	switch(c) {
	case '\n':
		goto loop;

	case 'x':
		i = 2;
		goto hex;

	case 'u':
		i = 4;
		goto hex;

	case 'U':
		i = 8;
		goto hex;

	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
		goto oct;

	case 'a': return '\a';
	case 'b': return '\b';
	case 'f': return '\f';
	case 'n': return '\n';
	case 'r': return '\r';
	case 't': return '\t';
	case 'v': return '\v';

	default:
		warn("unknown escape sequence: %c", c);
	}
	return c;

hex:
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
		warn("non-hex character in escape sequence: %c", c);
		ungetc(c);
		break;
	}
	*escflg = 1;
	return l;

oct:
	l = c - '0';
	for(i=2; i>0; i--) {
		c = getc();
		if(c >= '0' && c <= '7') {
			l = l*8 + c-'0';
			continue;
		}
		warn("non-oct character in escape sequence: %c", c);
		ungetc(c);
	}
	if(l > 255)
		warn("oct escape value > 255: %d", l);
	*escflg = 1;
	return l;
}

static	struct
{
	char*	name;
	int	lexical;
	int	etype;
} syms[] =
{
/*	name		lexical		etype
 */
/* basic types */
	"int8",		LBASETYPE,	TINT8,
	"int16",	LBASETYPE,	TINT16,
	"int32",	LBASETYPE,	TINT32,
	"int64",	LBASETYPE,	TINT64,

	"uint8",	LBASETYPE,	TUINT8,
	"uint16",	LBASETYPE,	TUINT16,
	"uint32",	LBASETYPE,	TUINT32,
	"uint64",	LBASETYPE,	TUINT64,

	"float32",	LBASETYPE,	TFLOAT32,
	"float64",	LBASETYPE,	TFLOAT64,
	"float80",	LBASETYPE,	TFLOAT80,

	"bool",		LBASETYPE,	TBOOL,
	"byte",		LBASETYPE,	TUINT8,
	"char",		LBASETYPE,	TUINT8,		// temp??
	"string",	LBASETYPE,	TSTRING,

/* keywords */
	"any",		LANY,		Txxx,
	"break",	LBREAK,		Txxx,
	"case",		LCASE,		Txxx,
	"chan",		LCHAN,		Txxx,
	"const",	LCONST,		Txxx,
	"continue",	LCONTINUE,	Txxx,
	"convert",	LCONVERT,	Txxx,
	"default",	LDEFAULT,	Txxx,
	"else",		LELSE,		Txxx,
	"export",	LEXPORT,	Txxx,
	"fallthrough",	LFALL,		Txxx,
	"false",	LFALSE,		Txxx,
	"for",		LFOR,		Txxx,
	"func",		LFUNC,		Txxx,
	"go",		LGO,		Txxx,
	"goto",		LGOTO,		Txxx,
	"if",		LIF,		Txxx,
	"import",	LIMPORT,	Txxx,
	"interface",	LINTERFACE,	Txxx,
	"iota",		LIOTA,		Txxx,
	"map",		LMAP,		Txxx,
	"new",		LNEW,		Txxx,
	"len",		LLEN,		Txxx,
	"nil",		LNIL,		Txxx,
	"package",	LPACKAGE,	Txxx,
	"panic",	LPANIC,		Txxx,
	"print",	LPRINT,		Txxx,
	"range",	LRANGE,		Txxx,
	"return",	LRETURN,	Txxx,
	"struct",	LSTRUCT,	Txxx,
	"switch",	LSWITCH,	Txxx,
	"true",		LTRUE,		Txxx,
	"type",		LTYPE,		Txxx,
	"var",		LVAR,		Txxx,

	"notwithstanding",		LIGNORE,	Txxx,
	"thetruthofthematter",		LIGNORE,	Txxx,
	"despiteallobjections",		LIGNORE,	Txxx,
	"whereas",			LIGNORE,	Txxx,
	"insofaras",			LIGNORE,	Txxx,
};

void
lexinit(void)
{
	int i, etype, lex;
	Sym *s;
	Node *t;


	for(i=TINT8; i<=TUINT64; i++)
		isint[i] = 1;
	for(i=TFLOAT32; i<=TFLOAT80; i++)
		isfloat[i] = 1;

	/*
	 * initialize okfor
	 */
	for(i=0; i<NTYPE; i++) {
		if(isint[i]) {
			okforeq[i] = 1;
			okforadd[i] = 1;
			okforand[i] = 1;
		}
		if(isfloat[i]) {
			okforeq[i] = 1;
			okforadd[i] = 1;
		}
		switch(i) {
		case TBOOL:
			okforeq[i] = 1;
			break;
		case TPTR:
			okforeq[i] = 1;
			break;
		}
		minfloatval[i] = 0.0;
		maxfloatval[i] = 0.0;
		minintval[i] = 0;
		maxintval[i] = 0;
	}
// this stuff smells - really need to do constants
// in multi precision arithmetic

	maxintval[TINT8] = 0x7f;
	minintval[TINT8] = -maxintval[TINT8]-1;
	maxintval[TINT16] = 0x7fff;
	minintval[TINT16] = -maxintval[TINT16]-1;
	maxintval[TINT32] = 0x7fffffffL;
	minintval[TINT32] = -maxintval[TINT32]-1;
	maxintval[TINT64] = 0x7fffffffffffffffLL;
	minintval[TINT64] = -maxintval[TINT64]-1;

	maxintval[TUINT8] = 0xff;
	maxintval[TUINT16] = 0xffff;
	maxintval[TUINT32] = 0xffffffffL;
	maxintval[TUINT64] = 0xffffffffffffffffLL;

	maxfloatval[TFLOAT32] = 3.40282347e+38;
	minfloatval[TFLOAT32] = -maxfloatval[TFLOAT32];
	maxfloatval[TFLOAT64] = 1.7976931348623157e+308;
	minfloatval[TFLOAT64] = -maxfloatval[TFLOAT64]-1;

	/*
	 * initialize basic types array
	 * initialize known symbols
	 */
	for(i=0; i<nelem(syms); i++) {
		lex = syms[i].lexical;
		s = lookup(syms[i].name);
		s->lexical = lex;

		if(lex != LBASETYPE)
			continue;

		etype = syms[i].etype;
		if(etype < 0 || etype >= nelem(types))
			fatal("lexinit: %s bad etype", s->name);

		t = types[etype];
		if(t != N) {
			s->otype = t;
			continue;
		}
		t = nod(OTYPE, N, N);
		t->etype = etype;
		switch(etype) {
		case TSTRING:
		case TCHAN:
		case TMAP:
			t = ptrto(t);
		}
		t->sym = s;
		t->recur = 1;	// supresses printing beyond name

		types[etype] = t;
		s->otype = t;
	}

	/* pick up the backend typedefs */
	belexinit(LBASETYPE);

	booltrue = nod(OLITERAL, N, N);
	booltrue->val.ctype = CTBOOL;
	booltrue->val.vval = 1;
	booltrue->type = types[TBOOL];

	boolfalse = nod(OLITERAL, N, N);
	boolfalse->val.ctype = CTBOOL;
	boolfalse->val.vval = 0;
	booltrue->type = types[TBOOL];
}

struct
{
	int	lex;
	char*	name;
} lexn[] =
{
	LANDAND,	"ANDAND",
	LASOP,		"ASOP",
	LACONST,	"ACONST",
	LATYPE,		"ATYPE",
	LBASETYPE,	"BASETYPE",
	LBREAK,		"BREAK",
	LCASE,		"CASE",
	LCHAN,		"CHAN",
	LCOLAS,		"COLAS",
	LCONST,		"CONST",
	LCONTINUE,	"CONTINUE",
	LDEC,		"DEC",
	LELSE,		"ELSE",
	LEQ,		"EQ",
	LFUNC,		"FUNC",
	LGE,		"GE",
	LGO,		"GO",
	LGOTO,		"GOTO",
	LGT,		"GT",
	LIF,		"IF",
	LINC,		"INC",
	LINTERFACE,	"INTERFACE",
	LLE,		"LE",
	LLITERAL,	"LITERAL",
	LLSH,		"LSH",
	LLT,		"LT",
	LMAP,		"MAP",
	LNAME,		"NAME",
	LNE,		"NE",
	LOROR,		"OROR",
	LPACK,		"PACK",
	LRANGE,		"RANGE",
	LRETURN,	"RETURN",
	LRSH,		"RSH",
	LSTRUCT,	"STRUCT",
	LSWITCH,	"SWITCH",
	LTYPE,		"TYPE",
	LVAR,		"VAR",
	LFOR,		"FOR",
	LNEW,		"NEW",
	LLEN,		"LEN",
	LFALL,		"FALL",
	LCONVERT,	"CONVERT",
	LIOTA,		"IOTA",
	LPRINT,		"PRINT",
	LPACKAGE,	"PACKAGE",
	LIMPORT,	"IMPORT",
	LEXPORT,	"EXPORT",
	LPANIC,		"PANIC",
};

char*
lexname(int lex)
{
	int i;
	static char buf[100];

	for(i=0; i<nelem(lexn); i++)
		if(lexn[i].lex == lex)
			return lexn[i].name;
	snprint(buf, sizeof(buf), "LEX-%d", lex);
	return buf;
}

void
mkpackage(char* pkg)
{
	Sym *s;
	long h;

	if(bout != nil) {
		yyerror("mkpackage: called again %s %s", pkg, package);
		return;
	}

	// defefine all names to be this package
	package = pkg;
	for(h=0; h<NHASH; h++)
		for(s = hash[h]; s != S; s = s->link) {
			s->package = package;
			s->opackage = package;
		}

	if(outfile == nil) {
		snprint(namebuf, sizeof(namebuf), "%s.go.c", package);
		outfile = strdup(namebuf);
	}

	bout = Bopen(outfile, OWRITE);
	if(bout == nil)
		fatal("cant open %s", outfile);
}
