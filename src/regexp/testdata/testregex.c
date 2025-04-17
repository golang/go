#pragma prototyped noticed

/*
 * regex(3) test harness
 *
 * build:	cc -o testregex testregex.c
 * help:	testregex --man
 * note:	REG_* features are detected by #ifdef; if REG_* are enums
 *		then supply #define REG_foo REG_foo for each enum REG_foo
 *
 *	Glenn Fowler <gsf@research.att.com>
 *	AT&T Research
 *
 * PLEASE: publish your tests so everyone can benefit
 *
 * The following license covers testregex.c and all associated test data.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of THIS SOFTWARE FILE (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, and/or sell copies of the
 * Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following disclaimer:
 *
 * THIS SOFTWARE IS PROVIDED BY AT&T ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL AT&T BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

static const char id[] = "\n@(#)$Id: testregex (AT&T Research) 2010-06-10 $\0\n";

#if _PACKAGE_ast
#include <ast.h>
#else
#include <sys/types.h>
#endif

#include <stdio.h>
#include <regex.h>
#include <ctype.h>
#include <setjmp.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#ifdef	__STDC__
#include <stdlib.h>
#include <locale.h>
#endif

#ifndef RE_DUP_MAX
#define RE_DUP_MAX	32767
#endif

#if !_PACKAGE_ast
#undef	REG_DISCIPLINE
#endif

#ifndef REG_DELIMITED
#undef	_REG_subcomp
#endif

#define TEST_ARE		0x00000001
#define TEST_BRE		0x00000002
#define TEST_ERE		0x00000004
#define TEST_KRE		0x00000008
#define TEST_LRE		0x00000010
#define TEST_SRE		0x00000020

#define TEST_EXPAND		0x00000100
#define TEST_LENIENT		0x00000200

#define TEST_QUERY		0x00000400
#define TEST_SUB		0x00000800
#define TEST_UNSPECIFIED	0x00001000
#define TEST_VERIFY		0x00002000
#define TEST_AND		0x00004000
#define TEST_OR			0x00008000

#define TEST_DELIMIT		0x00010000
#define TEST_OK			0x00020000
#define TEST_SAME		0x00040000

#define TEST_ACTUAL		0x00100000
#define TEST_BASELINE		0x00200000
#define TEST_FAIL		0x00400000
#define TEST_PASS		0x00800000
#define TEST_SUMMARY		0x01000000

#define TEST_IGNORE_ERROR	0x02000000
#define TEST_IGNORE_OVER	0x04000000
#define TEST_IGNORE_POSITION	0x08000000

#define TEST_CATCH		0x10000000
#define TEST_VERBOSE		0x20000000

#define TEST_DECOMP		0x40000000

#define TEST_GLOBAL		(TEST_ACTUAL|TEST_AND|TEST_BASELINE|TEST_CATCH|TEST_FAIL|TEST_IGNORE_ERROR|TEST_IGNORE_OVER|TEST_IGNORE_POSITION|TEST_OR|TEST_PASS|TEST_SUMMARY|TEST_VERBOSE)

#ifdef REG_DISCIPLINE


#include <stk.h>

typedef struct Disc_s
{
	regdisc_t	disc;
	int		ordinal;
	Sfio_t*		sp;
} Disc_t;

static void*
compf(const regex_t* re, const char* xstr, size_t xlen, regdisc_t* disc)
{
	Disc_t*		dp = (Disc_t*)disc;

	return (void*)((char*)0 + ++dp->ordinal);
}

static int
execf(const regex_t* re, void* data, const char* xstr, size_t xlen, const char* sstr, size_t slen, char** snxt, regdisc_t* disc)
{
	Disc_t*		dp = (Disc_t*)disc;

	sfprintf(dp->sp, "{%-.*s}(%lu:%d)", xlen, xstr, (char*)data - (char*)0, slen);
	return atoi(xstr);
}

static void*
resizef(void* handle, void* data, size_t size)
{
	if (!size)
		return 0;
	return stkalloc((Sfio_t*)handle, size);
}

#endif

#ifndef NiL
#ifdef	__STDC__
#define NiL		0
#else
#define NiL		(char*)0
#endif
#endif

#define H(x)		do{if(html)fprintf(stderr,x);}while(0)
#define T(x)		fprintf(stderr,x)

static void
help(int html)
{
H("<!DOCTYPE HTML PUBLIC \"-//IETF//DTD HTML//EN\">\n");
H("<HTML>\n");
H("<HEAD>\n");
H("<TITLE>testregex man document</TITLE>\n");
H("</HEAD>\n");
H("<BODY bgcolor=white>\n");
H("<PRE>\n");
T("NAME\n");
T("  testregex - regex(3) test harness\n");
T("\n");
T("SYNOPSIS\n");
T("  testregex [ options ]\n");
T("\n");
T("DESCRIPTION\n");
T("  testregex reads regex(3) test specifications, one per line, from the\n");
T("  standard input and writes one output line for each failed test. A\n");
T("  summary line is written after all tests are done. Each successful\n");
T("  test is run again with REG_NOSUB. Unsupported features are noted\n");
T("  before the first test, and tests requiring these features are\n");
T("  silently ignored.\n");
T("\n");
T("OPTIONS\n");
T("  -c	catch signals and non-terminating calls\n");
T("  -e	ignore error return mismatches\n");
T("  -h	list help on standard error\n");
T("  -n	do not repeat successful tests with regnexec()\n");
T("  -o	ignore match[] overrun errors\n");
T("  -p	ignore negative position mismatches\n");
T("  -s	use stack instead of malloc\n");
T("  -x	do not repeat successful tests with REG_NOSUB\n");
T("  -v	list each test line\n");
T("  -A	list failed test lines with actual answers\n");
T("  -B	list all test lines with actual answers\n");
T("  -F	list failed test lines\n");
T("  -P	list passed test lines\n");
T("  -S	output one summary line\n");
T("\n");
T("INPUT FORMAT\n");
T("  Input lines may be blank, a comment beginning with #, or a test\n");
T("  specification. A specification is five fields separated by one\n");
T("  or more tabs. NULL denotes the empty string and NIL denotes the\n");
T("  0 pointer.\n");
T("\n");
T("  Field 1: the regex(3) flags to apply, one character per REG_feature\n");
T("  flag. The test is skipped if REG_feature is not supported by the\n");
T("  implementation. If the first character is not [BEASKLP] then the\n");
T("  specification is a global control line. One or more of [BEASKLP] may be\n");
T("  specified; the test will be repeated for each mode.\n");
T("\n");
T("    B 	basic			BRE	(grep, ed, sed)\n");
T("    E 	REG_EXTENDED		ERE	(egrep)\n");
T("    A	REG_AUGMENTED		ARE	(egrep with negation)\n");
T("    S	REG_SHELL		SRE	(sh glob)\n");
T("    K	REG_SHELL|REG_AUGMENTED	KRE	(ksh glob)\n");
T("    L	REG_LITERAL		LRE	(fgrep)\n");
T("\n");
T("    a	REG_LEFT|REG_RIGHT	implicit ^...$\n");
T("    b	REG_NOTBOL		lhs does not match ^\n");
T("    c	REG_COMMENT		ignore space and #...\\n\n");
T("    d	REG_SHELL_DOT		explicit leading . match\n");
T("    e	REG_NOTEOL		rhs does not match $\n");
T("    f	REG_MULTIPLE		multiple \\n separated patterns\n");
T("    g	FNM_LEADING_DIR		testfnmatch only -- match until /\n");
T("    h	REG_MULTIREF		multiple digit backref\n");
T("    i	REG_ICASE		ignore case\n");
T("    j	REG_SPAN		. matches \\n\n");
T("    k	REG_ESCAPE		\\ to ecape [...] delimiter\n");
T("    l	REG_LEFT		implicit ^...\n");
T("    m	REG_MINIMAL		minimal match\n");
T("    n	REG_NEWLINE		explicit \\n match\n");
T("    o	REG_ENCLOSED		(|&) magic inside [@|&](...)\n");
T("    p	REG_SHELL_PATH		explicit / match\n");
T("    q	REG_DELIMITED		delimited pattern\n");
T("    r	REG_RIGHT		implicit ...$\n");
T("    s	REG_SHELL_ESCAPED	\\ not special\n");
T("    t	REG_MUSTDELIM		all delimiters must be specified\n");
T("    u	standard unspecified behavior -- errors not counted\n");
T("    v	REG_CLASS_ESCAPE	\\ special inside [...]\n");
T("    w	REG_NOSUB		no subexpression match array\n");
T("    x	REG_LENIENT		let some errors slide\n");
T("    y	REG_LEFT		regexec() implicit ^...\n");
T("    z	REG_NULL		NULL subexpressions ok\n");
T("    $	                        expand C \\c escapes in fields 2 and 3\n");
T("    /	                        field 2 is a regsubcomp() expression\n");
T("    =	                        field 3 is a regdecomp() expression\n");
T("\n");
T("  Field 1 control lines:\n");
T("\n");
T("    C		set LC_COLLATE and LC_CTYPE to locale in field 2\n");
T("\n");
T("    ?test ...	output field 5 if passed and != EXPECTED, silent otherwise\n");
T("    &test ...	output field 5 if current and previous passed\n");
T("    |test ...	output field 5 if current passed and previous failed\n");
T("    ; ...	output field 2 if previous failed\n");
T("    {test ...	skip if failed until }\n");
T("    }		end of skip\n");
T("\n");
T("    : comment		comment copied as output NOTE\n");
T("    :comment:test	:comment: ignored\n");
T("    N[OTE] comment	comment copied as output NOTE\n");
T("    T[EST] comment	comment\n");
T("\n");
T("    number		use number for nmatch (20 by default)\n");
T("\n");
T("  Field 2: the regular expression pattern; SAME uses the pattern from\n");
T("    the previous specification. RE_DUP_MAX inside {...} expands to the\n");
T("    value from <limits.h>.\n");
T("\n");
T("  Field 3: the string to match. X...{RE_DUP_MAX} expands to RE_DUP_MAX\n");
T("    copies of X.\n");
T("\n");
T("  Field 4: the test outcome. This is either one of the posix error\n");
T("    codes (with REG_ omitted) or the match array, a list of (m,n)\n");
T("    entries with m and n being first and last+1 positions in the\n");
T("    field 3 string, or NULL if REG_NOSUB is in effect and success\n");
T("    is expected. BADPAT is acceptable in place of any regcomp(3)\n");
T("    error code. The match[] array is initialized to (-2,-2) before\n");
T("    each test. All array elements from 0 to nmatch-1 must be specified\n");
T("    in the outcome. Unspecified endpoints (offset -1) are denoted by ?.\n");
T("    Unset endpoints (offset -2) are denoted by X. {x}(o:n) denotes a\n");
T("    matched (?{...}) expression, where x is the text enclosed by {...},\n");
T("    o is the expression ordinal counting from 1, and n is the length of\n");
T("    the unmatched portion of the subject string. If x starts with a\n");
T("    number then that is the return value of re_execf(), otherwise 0 is\n");
T("    returned. RE_DUP_MAX[-+]N expands to the <limits.h> value -+N.\n");
T("\n");
T("  Field 5: optional comment appended to the report.\n");
T("\n");
T("CAVEAT\n");
T("    If a regex implementation misbehaves with memory then all bets are off.\n");
T("\n");
T("CONTRIBUTORS\n");
T("  Glenn Fowler    gsf@research.att.com        (ksh strmatch, regex extensions)\n");
T("  David Korn      dgk@research.att.com        (ksh glob matcher)\n");
T("  Doug McIlroy    mcilroy@dartmouth.edu       (ast regex/testre in C++)\n");
T("  Tom Lord        lord@regexps.com            (rx tests)\n");
T("  Henry Spencer   henry@zoo.toronto.edu       (original public regex)\n");
T("  Andrew Hume     andrew@research.att.com     (gre tests)\n");
T("  John Maddock    John_Maddock@compuserve.com (regex++ tests)\n");
T("  Philip Hazel    ph10@cam.ac.uk              (pcre tests)\n");
T("  Ville Laurikari vl@iki.fi                   (libtre tests)\n");
H("</PRE>\n");
H("</BODY>\n");
H("</HTML>\n");
}

#ifndef elementsof
#define elementsof(x)	(sizeof(x)/sizeof(x[0]))
#endif

#ifndef streq
#define streq(a,b)	(*(a)==*(b)&&!strcmp(a,b))
#endif

#define HUNG		2
#define NOTEST		(~0)

#ifndef REG_TEST_DEFAULT
#define REG_TEST_DEFAULT	0
#endif

#ifndef REG_EXEC_DEFAULT
#define REG_EXEC_DEFAULT	0
#endif

static const char* unsupported[] =
{
	"BASIC",
#ifndef REG_EXTENDED
	"EXTENDED",
#endif
#ifndef REG_AUGMENTED
	"AUGMENTED",
#endif
#ifndef REG_SHELL
	"SHELL",
#endif

#ifndef REG_CLASS_ESCAPE
	"CLASS_ESCAPE",
#endif
#ifndef REG_COMMENT
	"COMMENT",
#endif
#ifndef REG_DELIMITED
	"DELIMITED",
#endif
#ifndef REG_DISCIPLINE
	"DISCIPLINE",
#endif
#ifndef REG_ESCAPE
	"ESCAPE",
#endif
#ifndef REG_ICASE
	"ICASE",
#endif
#ifndef REG_LEFT
	"LEFT",
#endif
#ifndef REG_LENIENT
	"LENIENT",
#endif
#ifndef REG_LITERAL
	"LITERAL",
#endif
#ifndef REG_MINIMAL
	"MINIMAL",
#endif
#ifndef REG_MULTIPLE
	"MULTIPLE",
#endif
#ifndef REG_MULTIREF
	"MULTIREF",
#endif
#ifndef REG_MUSTDELIM
	"MUSTDELIM",
#endif
#ifndef REG_NEWLINE
	"NEWLINE",
#endif
#ifndef REG_NOTBOL
	"NOTBOL",
#endif
#ifndef REG_NOTEOL
	"NOTEOL",
#endif
#ifndef REG_NULL
	"NULL",
#endif
#ifndef REG_RIGHT
	"RIGHT",
#endif
#ifndef REG_SHELL_DOT
	"SHELL_DOT",
#endif
#ifndef REG_SHELL_ESCAPED
	"SHELL_ESCAPED",
#endif
#ifndef REG_SHELL_GROUP
	"SHELL_GROUP",
#endif
#ifndef REG_SHELL_PATH
	"SHELL_PATH",
#endif
#ifndef REG_SPAN
	"SPAN",
#endif
#if REG_NOSUB & REG_TEST_DEFAULT
	"SUBMATCH",
#endif
#if !_REG_nexec
	"regnexec",
#endif
#if !_REG_subcomp
	"regsubcomp",
#endif
#if !_REG_decomp
	"redecomp",
#endif
	0
};

#ifndef REG_CLASS_ESCAPE
#define REG_CLASS_ESCAPE	NOTEST
#endif
#ifndef REG_COMMENT
#define REG_COMMENT	NOTEST
#endif
#ifndef REG_DELIMITED
#define REG_DELIMITED	NOTEST
#endif
#ifndef REG_ESCAPE
#define REG_ESCAPE	NOTEST
#endif
#ifndef REG_ICASE
#define REG_ICASE	NOTEST
#endif
#ifndef REG_LEFT
#define REG_LEFT	NOTEST
#endif
#ifndef REG_LENIENT
#define REG_LENIENT	0
#endif
#ifndef REG_MINIMAL
#define REG_MINIMAL	NOTEST
#endif
#ifndef REG_MULTIPLE
#define REG_MULTIPLE	NOTEST
#endif
#ifndef REG_MULTIREF
#define REG_MULTIREF	NOTEST
#endif
#ifndef REG_MUSTDELIM
#define REG_MUSTDELIM	NOTEST
#endif
#ifndef REG_NEWLINE
#define REG_NEWLINE	NOTEST
#endif
#ifndef REG_NOTBOL
#define REG_NOTBOL	NOTEST
#endif
#ifndef REG_NOTEOL
#define REG_NOTEOL	NOTEST
#endif
#ifndef REG_NULL
#define REG_NULL	NOTEST
#endif
#ifndef REG_RIGHT
#define REG_RIGHT	NOTEST
#endif
#ifndef REG_SHELL_DOT
#define REG_SHELL_DOT	NOTEST
#endif
#ifndef REG_SHELL_ESCAPED
#define REG_SHELL_ESCAPED	NOTEST
#endif
#ifndef REG_SHELL_GROUP
#define REG_SHELL_GROUP	NOTEST
#endif
#ifndef REG_SHELL_PATH
#define REG_SHELL_PATH	NOTEST
#endif
#ifndef REG_SPAN
#define REG_SPAN	NOTEST
#endif

#define REG_UNKNOWN	(-1)

#ifndef REG_ENEWLINE
#define REG_ENEWLINE	(REG_UNKNOWN-1)
#endif
#ifndef REG_ENULL
#ifndef REG_EMPTY
#define REG_ENULL	(REG_UNKNOWN-2)
#else
#define REG_ENULL	REG_EMPTY
#endif
#endif
#ifndef REG_ECOUNT
#define REG_ECOUNT	(REG_UNKNOWN-3)
#endif
#ifndef REG_BADESC
#define REG_BADESC	(REG_UNKNOWN-4)
#endif
#ifndef REG_EMEM
#define REG_EMEM	(REG_UNKNOWN-5)
#endif
#ifndef REG_EHUNG
#define REG_EHUNG	(REG_UNKNOWN-6)
#endif
#ifndef REG_EBUS
#define REG_EBUS	(REG_UNKNOWN-7)
#endif
#ifndef REG_EFAULT
#define REG_EFAULT	(REG_UNKNOWN-8)
#endif
#ifndef REG_EFLAGS
#define REG_EFLAGS	(REG_UNKNOWN-9)
#endif
#ifndef REG_EDELIM
#define REG_EDELIM	(REG_UNKNOWN-9)
#endif

static const struct { int code; char* name; } codes[] =
{
	REG_UNKNOWN,	"UNKNOWN",
	REG_NOMATCH,	"NOMATCH",
	REG_BADPAT,	"BADPAT",
	REG_ECOLLATE,	"ECOLLATE",
	REG_ECTYPE,	"ECTYPE",
	REG_EESCAPE,	"EESCAPE",
	REG_ESUBREG,	"ESUBREG",
	REG_EBRACK,	"EBRACK",
	REG_EPAREN,	"EPAREN",
	REG_EBRACE,	"EBRACE",
	REG_BADBR,	"BADBR",
	REG_ERANGE,	"ERANGE",
	REG_ESPACE,	"ESPACE",
	REG_BADRPT,	"BADRPT",
	REG_ENEWLINE,	"ENEWLINE",
	REG_ENULL,	"ENULL",
	REG_ECOUNT,	"ECOUNT",
	REG_BADESC,	"BADESC",
	REG_EMEM,	"EMEM",
	REG_EHUNG,	"EHUNG",
	REG_EBUS,	"EBUS",
	REG_EFAULT,	"EFAULT",
	REG_EFLAGS,	"EFLAGS",
	REG_EDELIM,	"EDELIM",
};

static struct
{
	regmatch_t	NOMATCH;
	int		errors;
	int		extracted;
	int		ignored;
	int		lineno;
	int		passed;
	int		signals;
	int		unspecified;
	int		verify;
	int		warnings;
	char*		file;
	char*		stack;
	char*		which;
	jmp_buf		gotcha;
#ifdef REG_DISCIPLINE
	Disc_t		disc;
#endif
} state;

static void
quote(char* s, int len, unsigned long test)
{
	unsigned char*	u = (unsigned char*)s;
	unsigned char*	e;
	int		c;
#ifdef MB_CUR_MAX
	int		w;
#endif

	if (!u)
		printf("NIL");
	else if (!*u && len <= 1)
		printf("NULL");
	else if (test & TEST_EXPAND)
	{
		if (len < 0)
			len = strlen((char*)u);
		e = u + len;
		if (test & TEST_DELIMIT)
			printf("\"");
		while (u < e)
			switch (c = *u++)
			{
			case '\\':
				printf("\\\\");
				break;
			case '"':
				if (test & TEST_DELIMIT)
					printf("\\\"");
				else
					printf("\"");
				break;
			case '\a':
				printf("\\a");
				break;
			case '\b':
				printf("\\b");
				break;
			case 033:
				printf("\\e");
				break;
			case '\f':
				printf("\\f");
				break;
			case '\n':
				printf("\\n");
				break;
			case '\r':
				printf("\\r");
				break;
			case '\t':
				printf("\\t");
				break;
			case '\v':
				printf("\\v");
				break;
			default:
#ifdef MB_CUR_MAX
				s = (char*)u - 1;
				if ((w = mblen(s, (char*)e - s)) > 1)
				{
					u += w - 1;
					fwrite(s, 1, w, stdout);
				}
				else
#endif
				if (!iscntrl(c) && isprint(c))
					putchar(c);
				else
					printf("\\x%02x", c);
				break;
			}
		if (test & TEST_DELIMIT)
			printf("\"");
	}
	else
		printf("%s", s);
}

static void
report(char* comment, char* fun, char* re, char* s, int len, char* msg, int flags, unsigned long test)
{
	if (state.file)
		printf("%s:", state.file);
	printf("%d:", state.lineno);
	if (re)
	{
		printf(" ");
		quote(re, -1, test|TEST_DELIMIT);
		if (s)
		{
			printf(" versus ");
			quote(s, len, test|TEST_DELIMIT);
		}
	}
	if (test & TEST_UNSPECIFIED)
	{
		state.unspecified++;
		printf(" unspecified behavior");
	}
	else
		state.errors++;
	if (state.which)
		printf(" %s", state.which);
	if (flags & REG_NOSUB)
		printf(" NOSUB");
	if (fun)
		printf(" %s", fun);
	if (comment[strlen(comment)-1] == '\n')
		printf(" %s", comment);
	else
	{
		printf(" %s: ", comment);
		if (msg)
			printf("%s: ", msg);
	}
}

static void
error(regex_t* preg, int code)
{
	char*	msg;
	char	buf[256];

	switch (code)
	{
	case REG_EBUS:
		msg = "bus error";
		break;
	case REG_EFAULT:
		msg = "memory fault";
		break;
	case REG_EHUNG:
		msg = "did not terminate";
		break;
	default:
		regerror(code, preg, msg = buf, sizeof buf);
		break;
	}
	printf("%s\n", msg);
}

static void
bad(char* comment, char* re, char* s, int len, unsigned long test)
{
	printf("bad test case ");
	report(comment, NiL, re, s, len, NiL, 0, test);
	exit(1);
}

static int
escape(char* s)
{
	char*	b;
	char*	t;
	char*	q;
	char*	e;
	int	c;

	for (b = t = s; *t = *s; s++, t++)
		if (*s == '\\')
			switch (*++s)
			{
			case '\\':
				break;
			case 'a':
				*t = '\a';
				break;
			case 'b':
				*t = '\b';
				break;
			case 'c':
				if (*t = *++s)
					*t &= 037;
				else
					s--;
				break;
			case 'e':
			case 'E':
				*t = 033;
				break;
			case 'f':
				*t = '\f';
				break;
			case 'n':
				*t = '\n';
				break;
			case 'r':
				*t = '\r';
				break;
			case 's':
				*t = ' ';
				break;
			case 't':
				*t = '\t';
				break;
			case 'v':
				*t = '\v';
				break;
			case 'u':
			case 'x':
				c = 0;
				q = c == 'u' ? (s + 5) : (char*)0;
				e = s + 1;
				while (!e || !q || s < q)
				{
					switch (*++s)
					{
					case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
						c = (c << 4) + *s - 'a' + 10;
						continue;
					case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
						c = (c << 4) + *s - 'A' + 10;
						continue;
					case '0': case '1': case '2': case '3': case '4':
					case '5': case '6': case '7': case '8': case '9':
						c = (c << 4) + *s - '0';
						continue;
					case '{':
					case '[':
						if (s != e)
						{
							s--;
							break;
						}
						e = 0;
						continue;
					case '}':
					case ']':
						if (e)
							s--;
						break;
					default:
						s--;
						break;
					}
					break;
				}
				*t = c;
				break;
			case '0': case '1': case '2': case '3':
			case '4': case '5': case '6': case '7':
				c = *s - '0';
				q = s + 2;
				while (s < q)
				{
					switch (*++s)
					{
					case '0': case '1': case '2': case '3':
					case '4': case '5': case '6': case '7':
						c = (c << 3) + *s - '0';
						break;
					default:
						q = --s;
						break;
					}
				}
				*t = c;
				break;
			default:
				*(s + 1) = 0;
				bad("invalid C \\ escape\n", s - 1, NiL, 0, 0);
			}
	return t - b;
}

static void
matchoffprint(int off)
{
	switch (off)
	{
	case -2:
		printf("X");
		break;
	case -1:
		printf("?");
		break;
	default:
		printf("%d", off);
		break;
	}
}

static void
matchprint(regmatch_t* match, int nmatch, int nsub, char* ans, unsigned long test)
{
	int	i;

	for (; nmatch > nsub + 1; nmatch--)
		if ((match[nmatch-1].rm_so != -1 || match[nmatch-1].rm_eo != -1) && (!(test & TEST_IGNORE_POSITION) || match[nmatch-1].rm_so >= 0 && match[nmatch-1].rm_eo >= 0))
			break;
	for (i = 0; i < nmatch; i++)
	{
		printf("(");
		matchoffprint(match[i].rm_so);
		printf(",");
		matchoffprint(match[i].rm_eo);
		printf(")");
	}
	if (!(test & (TEST_ACTUAL|TEST_BASELINE)))
	{
		if (ans)
			printf(" expected: %s", ans);
		printf("\n");
	}
}

static int
matchcheck(regmatch_t* match, int nmatch, int nsub, char* ans, char* re, char* s, int len, int flags, unsigned long test)
{
	char*	p;
	int	i;
	int	m;
	int	n;

	if (streq(ans, "OK"))
		return test & (TEST_BASELINE|TEST_PASS|TEST_VERIFY);
	for (i = 0, p = ans; i < nmatch && *p; i++)
	{
		if (*p == '{')
		{
#ifdef REG_DISCIPLINE
			char*	x;

			if (!(x = sfstruse(state.disc.sp)))
				bad("out of space [discipline string]\n", NiL, NiL, 0, 0);
			if (strcmp(p, x))
			{
				if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
					return 0;
				report("callout failed", NiL, re, s, len, NiL, flags, test);
				quote(p, -1, test);
				printf(" expected, ");
				quote(x, -1, test);
				printf(" returned\n");
			}
#endif
			break;
		}
		if (*p++ != '(')
			bad("improper answer\n", re, s, -1, test);
		if (*p == '?')
		{
			m = -1;
			p++;
		}
		else if (*p == 'R' && !memcmp(p, "RE_DUP_MAX", 10))
		{
			m = RE_DUP_MAX;
			p += 10;
			if (*p == '+' || *p == '-')
				m += strtol(p, &p, 10);
		}
		else
			m = strtol(p, &p, 10);
		if (*p++ != ',')
			bad("improper answer\n", re, s, -1, test);
		if (*p == '?')
		{
			n = -1;
			p++;
		}
		else if (*p == 'R' && !memcmp(p, "RE_DUP_MAX", 10))
		{
			n = RE_DUP_MAX;
			p += 10;
			if (*p == '+' || *p == '-')
				n += strtol(p, &p, 10);
		}
		else
			n = strtol(p, &p, 10);
		if (*p++ != ')')
			bad("improper answer\n", re, s, -1, test);
		if (m!=match[i].rm_so || n!=match[i].rm_eo)
		{
			if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY)))
			{
				report("failed: match was", NiL, re, s, len, NiL, flags, test);
				matchprint(match, nmatch, nsub, ans, test);
			}
			return 0;
		}
	}
	for (; i < nmatch; i++)
	{
		if (match[i].rm_so!=-1 || match[i].rm_eo!=-1)
		{
			if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_VERIFY)))
			{
				if ((test & TEST_IGNORE_POSITION) && (match[i].rm_so<0 || match[i].rm_eo<0))
				{
					state.ignored++;
					return 0;
				}
				if (!(test & TEST_SUMMARY))
				{
					report("failed: match was", NiL, re, s, len, NiL, flags, test);
					matchprint(match, nmatch, nsub, ans, test);
				}
			}
			return 0;
		}
	}
	if (!(test & TEST_IGNORE_OVER) && match[nmatch].rm_so != state.NOMATCH.rm_so)
	{
		if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY)))
		{
			report("failed: overran match array", NiL, re, s, len, NiL, flags, test);
			matchprint(match, nmatch + 1, nsub, NiL, test);
		}
		return 0;
	}
	return 1;
}

static void
sigunblock(int s)
{
#ifdef SIG_SETMASK
	int		op;
	sigset_t	mask;

	sigemptyset(&mask);
	if (s)
	{
		sigaddset(&mask, s);
		op = SIG_UNBLOCK;
	}
	else op = SIG_SETMASK;
	sigprocmask(op, &mask, NiL);
#else
#ifdef sigmask
	sigsetmask(s ? (sigsetmask(0L) & ~sigmask(s)) : 0L);
#endif
#endif
}

static void
gotcha(int sig)
{
	int	ret;

	signal(sig, gotcha);
	alarm(0);
	state.signals++;
	switch (sig)
	{
	case SIGALRM:
		ret = REG_EHUNG;
		break;
	case SIGBUS:
		ret = REG_EBUS;
		break;
	default:
		ret = REG_EFAULT;
		break;
	}
	sigunblock(sig);
	longjmp(state.gotcha, ret);
}

static char*
getline(FILE* fp)
{
	static char	buf[32 * 1024];

	register char*	s = buf;
	register char*	e = &buf[sizeof(buf)];
	register char*	b;

	for (;;)
	{
		if (!(b = fgets(s, e - s, fp)))
			return 0;
		state.lineno++;
		s += strlen(s);
		if (s == b || *--s != '\n' || s == b || *(s - 1) != '\\')
		{
			*s = 0;
			break;
		}
		s--;
	}
	return buf;
}

static unsigned long
note(unsigned long level, char* msg, unsigned long skip, unsigned long test)
{
	if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_SUMMARY)) && !skip)
	{
		printf("NOTE\t");
		if (msg)
			printf("%s: ", msg);
		printf("skipping lines %d", state.lineno);
	}
	return skip | level;
}

#define TABS(n)		&ts[7-((n)&7)]

static char		ts[] = "\t\t\t\t\t\t\t";

static unsigned long
extract(int* tabs, char* spec, char* re, char* s, char* ans, char* msg, char* accept, regmatch_t* match, int nmatch, int nsub, unsigned long skip, unsigned long level, unsigned long test)
{
	if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_OK|TEST_PASS|TEST_SUMMARY))
	{
		state.extracted = 1;
		if (test & TEST_OK)
		{
			state.passed++;
			if ((test & TEST_VERIFY) && !(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_SUMMARY)))
			{
				if (msg && strcmp(msg, "EXPECTED"))
					printf("NOTE\t%s\n", msg);
				return skip;
			}
			test &= ~(TEST_PASS|TEST_QUERY);
		}
		if (test & (TEST_QUERY|TEST_VERIFY))
		{
			if (test & TEST_BASELINE)
				test &= ~(TEST_BASELINE|TEST_PASS);
			else
				test |= TEST_PASS;
			skip |= level;
		}
		if (!(test & TEST_OK))
		{
			if (test & TEST_UNSPECIFIED)
				state.unspecified++;
			else
				state.errors++;
		}
		if (test & (TEST_PASS|TEST_SUMMARY))
			return skip;
		test &= ~TEST_DELIMIT;
		printf("%s%s", spec, TABS(*tabs++));
		if ((test & (TEST_BASELINE|TEST_SAME)) == (TEST_BASELINE|TEST_SAME))
			printf("SAME");
		else
			quote(re, -1, test);
		printf("%s", TABS(*tabs++));
		quote(s, -1, test);
		printf("%s", TABS(*tabs++));
		if (!(test & (TEST_ACTUAL|TEST_BASELINE)) || !accept && !match)
			printf("%s", ans);
		else if (accept)
			printf("%s", accept);
		else
			matchprint(match, nmatch, nsub, NiL, test);
		if (msg)
			printf("%s%s", TABS(*tabs++), msg);
		putchar('\n');
	}
	else if (test & TEST_QUERY)
		skip = note(level, msg, skip, test);
	else if (test & TEST_VERIFY)
		state.extracted = 1;
	return skip;
}

static int
catchfree(regex_t* preg, int flags, int* tabs, char* spec, char* re, char* s, char* ans, char* msg, char* accept, regmatch_t* match, int nmatch, int nsub, unsigned long skip, unsigned long level, unsigned long test)
{
	int	eret;

	if (!(test & TEST_CATCH))
	{
		regfree(preg);
		eret = 0;
	}
	else if (!(eret = setjmp(state.gotcha)))
	{
		alarm(HUNG);
		regfree(preg);
		alarm(0);
	}
	else if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
		extract(tabs, spec, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test);
	else
	{
		report("failed", "regfree", re, NiL, -1, msg, flags, test);
		error(preg, eret);
	}
	return eret;
}

static char*
expand(char* os, char* ot)
{
	char*	s = os;
	char*	t;
	int	n = 0;
	int	r;
	long	m;

	for (;;)
	{
		switch (*s++)
		{
		case 0:
			break;
		case '{':
			n++;
			continue;
		case '}':
			n--;
			continue;
		case 'R':
			if (n == 1 && !memcmp(s, "E_DUP_MAX", 9))
			{
				s--;
				for (t = ot; os < s; *t++ = *os++);
				r = ((t - ot) >= 5 && t[-1] == '{' && t[-2] == '.' && t[-3] == '.' && t[-4] == '.') ? t[-5] : 0;
				os = ot;
				m = RE_DUP_MAX;
				if (*(s += 10) == '+' || *s == '-')
					m += strtol(s, &s, 10);
				if (r)
				{
					t -= 5;
					while (m-- > 0)
						*t++ = r;
					while (*s && *s++ != '}');
				}
				else
					t += snprintf(t, 32, "%ld", m);
				while (*t = *s++)
					t++;
				break;
			}
			continue;
		default:
			continue;
		}
		break;
	}
	return os;
}

int
main(int argc, char** argv)
{
	int		flags;
	int		cflags;
	int		eflags;
	int		nmatch;
	int		nexec;
	int		nstr;
	int		cret;
	int		eret;
	int		nsub;
	int		i;
	int		j;
	int		expected;
	int		got;
	int		locale;
	int		subunitlen;
	int		testno;
	unsigned long	level;
	unsigned long	skip;
	char*		p;
	char*		line;
	char*		spec;
	char*		re;
	char*		s;
	char*		ans;
	char*		msg;
	char*		fun;
	char*		ppat;
	char*		subunit;
	char*		version;
	char*		field[6];
	char*		delim[6];
	FILE*		fp;
	int		tabs[6];
	char		unit[64];
	regmatch_t	match[100];
	regex_t		preg;

	static char	pat[32 * 1024];
	static char	patbuf[32 * 1024];
	static char	strbuf[32 * 1024];

	int		nonosub = REG_NOSUB == 0;
	int		nonexec = 0;

	unsigned long	test = 0;

	static char*	filter[] = { "-", 0 };

	state.NOMATCH.rm_so = state.NOMATCH.rm_eo = -2;
	p = unit;
	version = (char*)id + 10;
	while (p < &unit[sizeof(unit)-1] && (*p = *version++) && !isspace(*p))
		p++;
	*p = 0;
	while ((p = *++argv) && *p == '-')
		for (;;)
		{
			switch (*++p)
			{
			case 0:
				break;
			case 'c':
				test |= TEST_CATCH;
				continue;
			case 'e':
				test |= TEST_IGNORE_ERROR;
				continue;
			case 'h':
			case '?':
				help(0);
				return 2;
			case '-':
				help(p[1] == 'h');
				return 2;
			case 'n':
				nonexec = 1;
				continue;
			case 'o':
				test |= TEST_IGNORE_OVER;
				continue;
			case 'p':
				test |= TEST_IGNORE_POSITION;
				continue;
			case 's':
#ifdef REG_DISCIPLINE
				if (!(state.stack = stkalloc(stkstd, 0)))
					fprintf(stderr, "%s: out of space [stack]", unit);
				state.disc.disc.re_resizef = resizef;
				state.disc.disc.re_resizehandle = (void*)stkstd;
#endif
				continue;
			case 'x':
				nonosub = 1;
				continue;
			case 'v':
				test |= TEST_VERBOSE;
				continue;
			case 'A':
				test |= TEST_ACTUAL;
				continue;
			case 'B':
				test |= TEST_BASELINE;
				continue;
			case 'F':
				test |= TEST_FAIL;
				continue;
			case 'P':
				test |= TEST_PASS;
				continue;
			case 'S':
				test |= TEST_SUMMARY;
				continue;
			default:
				fprintf(stderr, "%s: %c: invalid option\n", unit, *p);
				return 2;
			}
			break;
		}
	if (!*argv)
		argv = filter;
	locale = 0;
	while (state.file = *argv++)
	{
		if (streq(state.file, "-") || streq(state.file, "/dev/stdin") || streq(state.file, "/dev/fd/0"))
		{
			state.file = 0;
			fp = stdin;
		}
		else if (!(fp = fopen(state.file, "r")))
		{
			fprintf(stderr, "%s: %s: cannot read\n", unit, state.file);
			return 2;
		}
		testno = state.errors = state.ignored = state.lineno = state.passed =
		state.signals = state.unspecified = state.warnings = 0;
		skip = 0;
		level = 1;
		if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_SUMMARY)))
		{
			printf("TEST\t%s ", unit);
			if (s = state.file)
			{
				subunit = p = 0;
				for (;;)
				{
					switch (*s++)
					{
					case 0:
						break;
					case '/':
						subunit = s;
						continue;
					case '.':
						p = s - 1;
						continue;
					default:
						continue;
					}
					break;
				}
				if (!subunit)
					subunit = state.file;
				if (p < subunit)
					p = s - 1;
				subunitlen = p - subunit;
				printf("%-.*s ", subunitlen, subunit);
			}
			else
				subunit = 0;
			for (s = version; *s && (*s != ' ' || *(s + 1) != '$'); s++)
				putchar(*s);
			if (test & TEST_CATCH)
				printf(", catch");
			if (test & TEST_IGNORE_ERROR)
				printf(", ignore error code mismatches");
			if (test & TEST_IGNORE_POSITION)
				printf(", ignore negative position mismatches");
#ifdef REG_DISCIPLINE
			if (state.stack)
				printf(", stack");
#endif
			if (test & TEST_VERBOSE)
				printf(", verbose");
			printf("\n");
#ifdef REG_VERSIONID
			if (regerror(REG_VERSIONID, NiL, pat, sizeof(pat)) > 0)
				s = pat;
			else
#endif
#ifdef REG_TEST_VERSION
			s = REG_TEST_VERSION;
#else
			s = "regex";
#endif
			printf("NOTE\t%s\n", s);
			if (elementsof(unsupported) > 1)
			{
#if (REG_TEST_DEFAULT & (REG_AUGMENTED|REG_EXTENDED|REG_SHELL)) || !defined(REG_EXTENDED)
				i = 0;
#else
				i = REG_EXTENDED != 0;
#endif
				for (got = 0; i < elementsof(unsupported) - 1; i++)
				{
					if (!got)
					{
						got = 1;
						printf("NOTE\tunsupported: %s", unsupported[i]);
					}
					else
						printf(",%s", unsupported[i]);
				}
				if (got)
					printf("\n");
			}
		}
#ifdef REG_DISCIPLINE
		state.disc.disc.re_version = REG_VERSION;
		state.disc.disc.re_compf = compf;
		state.disc.disc.re_execf = execf;
		if (!(state.disc.sp = sfstropen()))
			bad("out of space [discipline string stream]\n", NiL, NiL, 0, 0);
		preg.re_disc = &state.disc.disc;
#endif
		if (test & TEST_CATCH)
		{
			signal(SIGALRM, gotcha);
			signal(SIGBUS, gotcha);
			signal(SIGSEGV, gotcha);
		}
		while (p = getline(fp))
		{

		/* parse: */

			line = p;
			if (*p == ':' && !isspace(*(p + 1)))
			{
				while (*++p && *p != ':');
				if (!*p++)
				{
					if (test & TEST_BASELINE)
						printf("%s\n", line);
					continue;
				}
			}
			while (isspace(*p))
				p++;
			if (*p == 0 || *p == '#' || *p == 'T')
			{
				if (test & TEST_BASELINE)
					printf("%s\n", line);
				continue;
			}
			if (*p == ':' || *p == 'N')
			{
				if (test & TEST_BASELINE)
					printf("%s\n", line);
				else if (!(test & (TEST_ACTUAL|TEST_FAIL|TEST_PASS|TEST_SUMMARY)))
				{
					while (*++p && !isspace(*p));
					while (isspace(*p))
						p++;
					printf("NOTE	%s\n", p);
				}
				continue;
			}
			j = 0;
			i = 0;
			field[i++] = p;
			for (;;)
			{
				switch (*p++)
				{
				case 0:
					p--;
					j = 0;
					goto checkfield;
				case '\t':
					*(delim[i] = p - 1) = 0;
					j = 1;
				checkfield:
					s = field[i - 1];
					if (streq(s, "NIL"))
						field[i - 1] = 0;
					else if (streq(s, "NULL"))
						*s = 0;
					while (*p == '\t')
					{
						p++;
						j++;
					}
					tabs[i - 1] = j;
					if (!*p)
						break;
					if (i >= elementsof(field))
						bad("too many fields\n", NiL, NiL, 0, 0);
					field[i++] = p;
					/*FALLTHROUGH*/
				default:
					continue;
				}
				break;
			}
			if (!(spec = field[0]))
				bad("NIL spec\n", NiL, NiL, 0, 0);

		/* interpret: */

			cflags = REG_TEST_DEFAULT;
			eflags = REG_EXEC_DEFAULT;
			test &= TEST_GLOBAL;
			state.extracted = 0;
			nmatch = 20;
			nsub = -1;
			for (p = spec; *p; p++)
			{
				if (isdigit(*p))
				{
					nmatch = strtol(p, &p, 10);
					if (nmatch >= elementsof(match))
						bad("nmatch must be < 100\n", NiL, NiL, 0, 0);
					p--;
					continue;
				}
				switch (*p)
				{
				case 'A':
					test |= TEST_ARE;
					continue;
				case 'B':
					test |= TEST_BRE;
					continue;
				case 'C':
					if (!(test & TEST_QUERY) && !(skip & level))
						bad("locale must be nested\n", NiL, NiL, 0, 0);
					test &= ~TEST_QUERY;
					if (locale)
						bad("locale nesting not supported\n", NiL, NiL, 0, 0);
					if (i != 2)
						bad("locale field expected\n", NiL, NiL, 0, 0);
					if (!(skip & level))
					{
#if defined(LC_COLLATE) && defined(LC_CTYPE)
						s = field[1];
						if (!s || streq(s, "POSIX"))
							s = "C";
						if ((ans = setlocale(LC_COLLATE, s)) && streq(ans, "POSIX"))
							ans = "C";
						if (!ans || !streq(ans, s) && streq(s, "C"))
							ans = 0;
						else if ((ans = setlocale(LC_CTYPE, s)) && streq(ans, "POSIX"))
							ans = "C";
						if (!ans || !streq(ans, s) && streq(s, "C"))
							skip = note(level, s, skip, test);
						else
						{
							if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_SUMMARY)))
								printf("NOTE	\"%s\" locale\n", s);
							locale = level;
						}
#else
						skip = note(level, skip, test, "locales not supported");
#endif
					}
					cflags = NOTEST;
					continue;
				case 'E':
					test |= TEST_ERE;
					continue;
				case 'K':
					test |= TEST_KRE;
					continue;
				case 'L':
					test |= TEST_LRE;
					continue;
				case 'S':
					test |= TEST_SRE;
					continue;

				case 'a':
					cflags |= REG_LEFT|REG_RIGHT;
					continue;
				case 'b':
					eflags |= REG_NOTBOL;
					continue;
				case 'c':
					cflags |= REG_COMMENT;
					continue;
				case 'd':
					cflags |= REG_SHELL_DOT;
					continue;
				case 'e':
					eflags |= REG_NOTEOL;
					continue;
				case 'f':
					cflags |= REG_MULTIPLE;
					continue;
				case 'g':
					cflags |= NOTEST;
					continue;
				case 'h':
					cflags |= REG_MULTIREF;
					continue;
				case 'i':
					cflags |= REG_ICASE;
					continue;
				case 'j':
					cflags |= REG_SPAN;
					continue;
				case 'k':
					cflags |= REG_ESCAPE;
					continue;
				case 'l':
					cflags |= REG_LEFT;
					continue;
				case 'm':
					cflags |= REG_MINIMAL;
					continue;
				case 'n':
					cflags |= REG_NEWLINE;
					continue;
				case 'o':
					cflags |= REG_SHELL_GROUP;
					continue;
				case 'p':
					cflags |= REG_SHELL_PATH;
					continue;
				case 'q':
					cflags |= REG_DELIMITED;
					continue;
				case 'r':
					cflags |= REG_RIGHT;
					continue;
				case 's':
					cflags |= REG_SHELL_ESCAPED;
					continue;
				case 't':
					cflags |= REG_MUSTDELIM;
					continue;
				case 'u':
					test |= TEST_UNSPECIFIED;
					continue;
				case 'v':
					cflags |= REG_CLASS_ESCAPE;
					continue;
				case 'w':
					cflags |= REG_NOSUB;
					continue;
				case 'x':
					if (REG_LENIENT)
						cflags |= REG_LENIENT;
					else
						test |= TEST_LENIENT;
					continue;
				case 'y':
					eflags |= REG_LEFT;
					continue;
				case 'z':
					cflags |= REG_NULL;
					continue;

				case '$':
					test |= TEST_EXPAND;
					continue;

				case '/':
					test |= TEST_SUB;
					continue;

				case '=':
					test |= TEST_DECOMP;
					continue;

				case '?':
					test |= TEST_VERIFY;
					test &= ~(TEST_AND|TEST_OR);
					state.verify = state.passed;
					continue;
				case '&':
					test |= TEST_VERIFY|TEST_AND;
					test &= ~TEST_OR;
					continue;
				case '|':
					test |= TEST_VERIFY|TEST_OR;
					test &= ~TEST_AND;
					continue;
				case ';':
					test |= TEST_OR;
					test &= ~TEST_AND;
					continue;

				case '{':
					level <<= 1;
					if (skip & (level >> 1))
					{
						skip |= level;
						cflags = NOTEST;
					}
					else
					{
						skip &= ~level;
						test |= TEST_QUERY;
					}
					continue;
				case '}':
					if (level == 1)
						bad("invalid {...} nesting\n", NiL, NiL, 0, 0);
					if ((skip & level) && !(skip & (level>>1)))
					{
						if (!(test & (TEST_BASELINE|TEST_SUMMARY)))
						{
							if (test & (TEST_ACTUAL|TEST_FAIL))
								printf("}\n");
							else if (!(test & TEST_PASS))
								printf("-%d\n", state.lineno);
						}
					}
#if defined(LC_COLLATE) && defined(LC_CTYPE)
					else if (locale & level)
					{
						locale = 0;
						if (!(skip & level))
						{
							s = "C";
							setlocale(LC_COLLATE, s);
							setlocale(LC_CTYPE, s);
							if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_SUMMARY)))
								printf("NOTE	\"%s\" locale\n", s);
							else if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_PASS))
								printf("}\n");
						}
						else if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL))
							printf("}\n");
					}
#endif
					level >>= 1;
					cflags = NOTEST;
					continue;

				default:
					bad("bad spec\n", spec, NiL, 0, test);
					break;

				}
				break;
			}
			if ((cflags|eflags) == NOTEST || (skip & level) && (test & TEST_BASELINE))
			{
				if (test & TEST_BASELINE)
				{
					while (i > 1)
						*delim[--i] = '\t';
					printf("%s\n", line);
				}
				continue;
			}
			if (test & TEST_OR)
			{
				if (!(test & TEST_VERIFY))
				{
					test &= ~TEST_OR;
					if (state.passed == state.verify && i > 1)
						printf("NOTE\t%s\n", field[1]);
					continue;
				}
				else if (state.passed > state.verify)
					continue;
			}
			else if (test & TEST_AND)
			{
				if (state.passed == state.verify)
					continue;
				state.passed = state.verify;
			}
			if (i < ((test & TEST_DECOMP) ? 3 : 4))
				bad("too few fields\n", NiL, NiL, 0, test);
			while (i < elementsof(field))
				field[i++] = 0;
			if (re = field[1])
			{
				if (streq(re, "SAME"))
				{
					re = ppat;
					test |= TEST_SAME;
				}
				else
				{
					if (test & TEST_EXPAND)
						escape(re);
					re = expand(re, patbuf);
					strcpy(ppat = pat, re);
				}
			}
			else
				ppat = 0;
			nstr = -1;
			if (s = field[2])
			{
				s = expand(s, strbuf);
				if (test & TEST_EXPAND)
				{
					nstr = escape(s);
#if _REG_nexec
					if (nstr != strlen(s))
						nexec = nstr;
#endif
				}
			}
			if (!(ans = field[(test & TEST_DECOMP) ? 2 : 3]))
				bad("NIL answer\n", NiL, NiL, 0, test);
			msg = field[4];
			fflush(stdout);
			if (test & TEST_SUB)
#if _REG_subcomp
				cflags |= REG_DELIMITED;
#else
				continue;
#endif
#if !_REG_decomp
			if (test & TEST_DECOMP)
				continue;
#endif

		compile:

			if (state.extracted || (skip & level))
				continue;
#if !(REG_TEST_DEFAULT & (REG_AUGMENTED|REG_EXTENDED|REG_SHELL))
#ifdef REG_EXTENDED
			if (REG_EXTENDED != 0 && (test & TEST_BRE))
#else
			if (test & TEST_BRE)
#endif
			{
				test &= ~TEST_BRE;
				flags = cflags;
				state.which = "BRE";
			}
			else
#endif
#ifdef REG_EXTENDED
			if (test & TEST_ERE)
			{
				test &= ~TEST_ERE;
				flags = cflags | REG_EXTENDED;
				state.which = "ERE";
			}
			else
#endif
#ifdef REG_AUGMENTED
			if (test & TEST_ARE)
			{
				test &= ~TEST_ARE;
				flags = cflags | REG_AUGMENTED;
				state.which = "ARE";
			}
			else
#endif
#ifdef REG_LITERAL
			if (test & TEST_LRE)
			{
				test &= ~TEST_LRE;
				flags = cflags | REG_LITERAL;
				state.which = "LRE";
			}
			else
#endif
#ifdef REG_SHELL
			if (test & TEST_SRE)
			{
				test &= ~TEST_SRE;
				flags = cflags | REG_SHELL;
				state.which = "SRE";
			}
			else
#ifdef REG_AUGMENTED
			if (test & TEST_KRE)
			{
				test &= ~TEST_KRE;
				flags = cflags | REG_SHELL | REG_AUGMENTED;
				state.which = "KRE";
			}
			else
#endif
#endif
			{
				if (test & (TEST_BASELINE|TEST_PASS|TEST_VERIFY))
					extract(tabs, line, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test|TEST_OK);
				continue;
			}
			if ((test & (TEST_QUERY|TEST_VERBOSE|TEST_VERIFY)) == TEST_VERBOSE)
			{
				printf("test %-3d %s ", state.lineno, state.which);
				quote(re, -1, test|TEST_DELIMIT);
				printf(" ");
				quote(s, nstr, test|TEST_DELIMIT);
				printf("\n");
			}

		nosub:
			fun = "regcomp";
#if _REG_nexec
			if (nstr >= 0 && nstr != strlen(s))
				nexec = nstr;

			else
#endif
				nexec = -1;
			if (state.extracted || (skip & level))
				continue;
			if (!(test & TEST_QUERY))
				testno++;
#ifdef REG_DISCIPLINE
			if (state.stack)
				stkset(stkstd, state.stack, 0);
			flags |= REG_DISCIPLINE;
			state.disc.ordinal = 0;
			sfstrseek(state.disc.sp, 0, SEEK_SET);
#endif
			if (!(test & TEST_CATCH))
				cret = regcomp(&preg, re, flags);
			else if (!(cret = setjmp(state.gotcha)))
			{
				alarm(HUNG);
				cret = regcomp(&preg, re, flags);
				alarm(0);
			}
#if _REG_subcomp
			if (!cret && (test & TEST_SUB))
			{
				fun = "regsubcomp";
				p = re + preg.re_npat;
				if (!(test & TEST_CATCH))
					cret = regsubcomp(&preg, p, NiL, 0, 0);
				else if (!(cret = setjmp(state.gotcha)))
				{
					alarm(HUNG);
					cret = regsubcomp(&preg, p, NiL, 0, 0);
					alarm(0);
				}
				if (!cret && *(p += preg.re_npat) && !(preg.re_sub->re_flags & REG_SUB_LAST))
				{
					if (catchfree(&preg, flags, tabs, line, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test))
						continue;
					cret = REG_EFLAGS;
				}
			}
#endif
#if _REG_decomp
			if (!cret && (test & TEST_DECOMP))
			{
				char	buf[128];

				if ((j = nmatch) > sizeof(buf))
					j = sizeof(buf);
				fun = "regdecomp";
				p = re + preg.re_npat;
				if (!(test & TEST_CATCH))
					i = regdecomp(&preg, -1, buf, j);
				else if (!(cret = setjmp(state.gotcha)))
				{
					alarm(HUNG);
					i = regdecomp(&preg, -1, buf, j);
					alarm(0);
				}
				if (!cret)
				{
					catchfree(&preg, flags, tabs, line, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test);
					if (i > j)
					{
						if (i != (strlen(ans) + 1))
						{
							report("failed", fun, re, s, nstr, msg, flags, test);
							printf(" %d byte buffer supplied, %d byte buffer required\n", j, i);
						}
					}
					else if (strcmp(buf, ans))
					{
						report("failed", fun, re, s, nstr, msg, flags, test);
						quote(ans, -1, test|TEST_DELIMIT);
						printf(" expected, ");
						quote(buf, -1, test|TEST_DELIMIT);
						printf(" returned\n");
					}
					continue;
				}
			}
#endif
			if (!cret)
			{
				if (!(flags & REG_NOSUB) && nsub < 0 && *ans == '(')
				{
					for (p = ans; *p; p++)
						if (*p == '(')
							nsub++;
						else if (*p == '{')
							nsub--;
					if (nsub >= 0)
					{
						if (test & TEST_IGNORE_OVER)
						{
							if (nmatch > nsub)
								nmatch = nsub + 1;
						}
						else if (nsub != preg.re_nsub)
						{
							if (nsub > preg.re_nsub)
							{
								if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
									skip = extract(tabs, line, re, s, ans, msg, "OK", NiL, 0, 0, skip, level, test|TEST_DELIMIT);
								else
								{
									report("re_nsub incorrect", fun, re, NiL, -1, msg, flags, test);
									printf("at least %d expected, %d returned\n", nsub, preg.re_nsub);
									state.errors++;
								}
							}
							else
								nsub = preg.re_nsub;
						}
					}
				}
				if (!(test & (TEST_DECOMP|TEST_SUB)) && *ans && *ans != '(' && !streq(ans, "OK") && !streq(ans, "NOMATCH"))
				{
					if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
						skip = extract(tabs, line, re, s, ans, msg, "OK", NiL, 0, 0, skip, level, test|TEST_DELIMIT);
					else if (!(test & TEST_LENIENT))
					{
						report("failed", fun, re, NiL, -1, msg, flags, test);
						printf("%s expected, OK returned\n", ans);
					}
					catchfree(&preg, flags, tabs, line, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test);
					continue;
				}
			}
			else
			{
				if (test & TEST_LENIENT)
					/* we'll let it go this time */;
				else if (!*ans || ans[0]=='(' || cret == REG_BADPAT && streq(ans, "NOMATCH"))
				{
					got = 0;
					for (i = 1; i < elementsof(codes); i++)
						if (cret==codes[i].code)
							got = i;
					if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
						skip = extract(tabs, line, re, s, ans, msg, codes[got].name, NiL, 0, 0, skip, level, test|TEST_DELIMIT);
					else
					{
						report("failed", fun, re, NiL, -1, msg, flags, test);
						printf("%s returned: ", codes[got].name);
						error(&preg, cret);
					}
				}
				else
				{
					expected = got = 0;
					for (i = 1; i < elementsof(codes); i++)
					{
						if (streq(ans, codes[i].name))
							expected = i;
						if (cret==codes[i].code)
							got = i;
					}
					if (!expected)
					{
						if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
							skip = extract(tabs, line, re, s, ans, msg, codes[got].name, NiL, 0, 0, skip, level, test|TEST_DELIMIT);
						else
						{
							report("failed: invalid error code", NiL, re, NiL, -1, msg, flags, test);
							printf("%s expected, %s returned\n", ans, codes[got].name);
						}
					}
					else if (cret != codes[expected].code && cret != REG_BADPAT)
					{
						if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
							skip = extract(tabs, line, re, s, ans, msg, codes[got].name, NiL, 0, 0, skip, level, test|TEST_DELIMIT);
						else if (test & TEST_IGNORE_ERROR)
							state.ignored++;
						else
						{
							report("should fail and did", fun, re, NiL, -1, msg, flags, test);
							printf("%s expected, %s returned: ", ans, codes[got].name);
							state.errors--;
							state.warnings++;
							error(&preg, cret);
						}
					}
				}
				goto compile;
			}

#if _REG_nexec
		execute:
			if (nexec >= 0)
				fun = "regnexec";
			else
#endif
				fun = "regexec";
			
			for (i = 0; i < elementsof(match); i++)
				match[i] = state.NOMATCH;

#if _REG_nexec
			if (nexec >= 0)
			{
				eret = regnexec(&preg, s, nexec, nmatch, match, eflags);
				s[nexec] = 0;
			}
			else
#endif
			{
				if (!(test & TEST_CATCH))
					eret = regexec(&preg, s, nmatch, match, eflags);
				else if (!(eret = setjmp(state.gotcha)))
				{
					alarm(HUNG);
					eret = regexec(&preg, s, nmatch, match, eflags);
					alarm(0);
				}
			}
#if _REG_subcomp
			if ((test & TEST_SUB) && !eret)
			{
				fun = "regsubexec";
				if (!(test & TEST_CATCH))
					eret = regsubexec(&preg, s, nmatch, match);
				else if (!(eret = setjmp(state.gotcha)))
				{
					alarm(HUNG);
					eret = regsubexec(&preg, s, nmatch, match);
					alarm(0);
				}
			}
#endif
			if (flags & REG_NOSUB)
			{
				if (eret)
				{
					if (eret != REG_NOMATCH || !streq(ans, "NOMATCH"))
					{
						if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
							skip = extract(tabs, line, re, s, ans, msg, "NOMATCH", NiL, 0, 0, skip, level, test|TEST_DELIMIT);
						else
						{
							report("REG_NOSUB failed", fun, re, s, nstr, msg, flags, test);
							error(&preg, eret);
						}
					}
				}
				else if (streq(ans, "NOMATCH"))
				{
					if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
						skip = extract(tabs, line, re, s, ans, msg, NiL, match, nmatch, nsub, skip, level, test|TEST_DELIMIT);
					else
					{
						report("should fail and didn't", fun, re, s, nstr, msg, flags, test);
						error(&preg, eret);
					}
				}
			}
			else if (eret)
			{
				if (eret != REG_NOMATCH || !streq(ans, "NOMATCH"))
				{
					if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
						skip = extract(tabs, line, re, s, ans, msg, "NOMATCH", NiL, 0, nsub, skip, level, test|TEST_DELIMIT);
					else
					{
						report("failed", fun, re, s, nstr, msg, flags, test);
						if (eret != REG_NOMATCH)
							error(&preg, eret);
						else if (*ans)
							printf("expected: %s\n", ans);
						else
							printf("\n");
					}
				}
			}
			else if (streq(ans, "NOMATCH"))
			{
				if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
					skip = extract(tabs, line, re, s, ans, msg, NiL, match, nmatch, nsub, skip, level, test|TEST_DELIMIT);
				else
				{
					report("should fail and didn't", fun, re, s, nstr, msg, flags, test);
					matchprint(match, nmatch, nsub, NiL, test);
				}
			}
#if _REG_subcomp
			else if (test & TEST_SUB)
			{
				p = preg.re_sub->re_buf;
				if (strcmp(p, ans))
				{
					report("failed", fun, re, s, nstr, msg, flags, test);
					quote(ans, -1, test|TEST_DELIMIT);
					printf(" expected, ");
					quote(p, -1, test|TEST_DELIMIT);
					printf(" returned\n");
				}
			}
#endif
			else if (!*ans)
			{
				if (match[0].rm_so != state.NOMATCH.rm_so)
				{
					if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
						skip = extract(tabs, line, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test);
					else
					{
						report("failed: no match but match array assigned", NiL, re, s, nstr, msg, flags, test);
						matchprint(match, nmatch, nsub, NiL, test);
					}
				}
			}
			else if (matchcheck(match, nmatch, nsub, ans, re, s, nstr, flags, test))
			{
#if _REG_nexec
				if (nexec < 0 && !nonexec)
				{
					nexec = nstr >= 0 ? nstr : strlen(s);
					s[nexec] = '\n';
					testno++;
					goto execute;
				}
#endif
				if (!(test & (TEST_DECOMP|TEST_SUB|TEST_VERIFY)) && !nonosub)
				{
					if (catchfree(&preg, flags, tabs, line, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test))
						continue;
					flags |= REG_NOSUB;
					goto nosub;
				}
				if (test & (TEST_BASELINE|TEST_PASS|TEST_VERIFY))
					skip = extract(tabs, line, re, s, ans, msg, NiL, match, nmatch, nsub, skip, level, test|TEST_OK);
			}
			else if (test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS|TEST_QUERY|TEST_SUMMARY|TEST_VERIFY))
				skip = extract(tabs, line, re, s, ans, msg, NiL, match, nmatch, nsub, skip, level, test|TEST_DELIMIT);
			if (catchfree(&preg, flags, tabs, line, re, s, ans, msg, NiL, NiL, 0, 0, skip, level, test))
				continue;
			goto compile;
		}
		if (test & TEST_SUMMARY)
			printf("tests=%-4d errors=%-4d warnings=%-2d ignored=%-2d unspecified=%-2d signals=%d\n", testno, state.errors, state.warnings, state.ignored, state.unspecified, state.signals);
		else if (!(test & (TEST_ACTUAL|TEST_BASELINE|TEST_FAIL|TEST_PASS)))
		{
			printf("TEST\t%s", unit);
			if (subunit)
				printf(" %-.*s", subunitlen, subunit);
			printf(", %d test%s", testno, testno == 1 ? "" : "s");
			if (state.ignored)
				printf(", %d ignored mismatche%s", state.ignored, state.ignored == 1 ? "" : "s");
			if (state.warnings)
				printf(", %d warning%s", state.warnings, state.warnings == 1 ? "" : "s");
			if (state.unspecified)
				printf(", %d unspecified difference%s", state.unspecified, state.unspecified == 1 ? "" : "s");
			if (state.signals)
				printf(", %d signal%s", state.signals, state.signals == 1 ? "" : "s");
			printf(", %d error%s\n", state.errors, state.errors == 1 ? "" : "s");
		}
		if (fp != stdin)
			fclose(fp);
	}
	return 0;
}
