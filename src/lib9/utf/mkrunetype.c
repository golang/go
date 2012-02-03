// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
 * make is(upper|lower|title|space|alpha)rune and
 * to(upper|lower|title)rune from a UnicodeData.txt file.
 * these can be found at unicode.org
 *
 * with -c, runs a check of the existing runetype functions vs.
 * those extracted from UnicodeData.
 *
 * with -p, generates tables for pairs of chars, as well as for ranges
 * and singletons.
 *
 * UnicodeData defines 4 fields of interest:
 * 1) a category
 * 2) an upper case mapping
 * 3) a lower case mapping
 * 4) a title case mapping
 *
 * toupper, tolower, and totitle are defined directly from the mapping.
 *
 * isalpharune(c) is true iff c is a "letter" category
 * isupperrune(c) is true iff c is the target of toupperrune,
 *	or is in the uppercase letter category
 * similarly for islowerrune and istitlerune.
 * isspacerune is true for space category chars, "C" locale white space chars,
 *	and two additions:
 *	0085	"next line" control char
 *	feff]	"zero-width non-break space"
 * isdigitrune is true iff c is a numeric-digit category.
 */

#include <u.h>
#include <libc.h>
#include <stdio.h>
#include "utf.h"
#include "utfdef.h"

enum {
	/*
	 * fields in the unicode data file
	 */
	FIELD_CODE,
	FIELD_NAME,
	FIELD_CATEGORY,
	FIELD_COMBINING,
	FIELD_BIDIR,
	FIELD_DECOMP,
	FIELD_DECIMAL_DIG,
	FIELD_DIG,
	FIELD_NUMERIC_VAL,
	FIELD_MIRRORED,
	FIELD_UNICODE_1_NAME,
	FIELD_COMMENT,
	FIELD_UPPER,
	FIELD_LOWER,
	FIELD_TITLE,
	NFIELDS,

	MAX_LINE	= 1024,

	TO_OFFSET	= 1 << 20,

	NRUNES		= 1 << 21,
};

#define TO_DELTA(xmapped,x)	(TO_OFFSET + (xmapped) - (x))

static char	myisspace[NRUNES];
static char	myisalpha[NRUNES];
static char	myisdigit[NRUNES];
static char	myisupper[NRUNES];
static char	myislower[NRUNES];
static char	myistitle[NRUNES];

static int	mytoupper[NRUNES];
static int	mytolower[NRUNES];
static int	mytotitle[NRUNES];

static void	check(void);
static void	mktables(char *src, int usepairs);
static void	fatal(const char *fmt, ...);
static int	mygetfields(char **fields, int nfields, char *str, const char *delim);
static int	getunicodeline(FILE *in, char **fields, char *buf);
static int	getcode(char *s);

static void
usage(void)
{
	fprintf(stderr, "usage: mktables [-cp] <UnicodeData.txt>\n");
	exit(1);
}

void
main(int argc, char *argv[])
{
	FILE *in;
	char buf[MAX_LINE], buf2[MAX_LINE];
	char *fields[NFIELDS + 1], *fields2[NFIELDS + 1];
	char *p;
	int i, code, last, docheck, usepairs;

	docheck = 0;
	usepairs = 0;
	ARGBEGIN{
	case 'c':
		docheck = 1;
		break;
	case 'p':
		usepairs = 1;
		break;
	default:
		usage();
	}ARGEND

	if(argc != 1){
		usage();
	}

	in = fopen(argv[0], "r");
	if(in == NULL){
		fatal("can't open %s", argv[0]);
	}

	for(i = 0; i < NRUNES; i++){
		mytoupper[i] = i;
		mytolower[i] = i;
		mytotitle[i] = i;
	}

	/*
	 * make sure isspace has all of the "C" locale whitespace chars
	 */
	myisspace['\t'] = 1;
	myisspace['\n'] = 1;
	myisspace['\r'] = 1;
	myisspace['\f'] = 1;
	myisspace['\v'] = 1;

	/*
	 * a couple of other exceptions
	 */
	myisspace[0x85] = 1;	/* control char, "next line" */
	myisspace[0xfeff] = 1;	/* zero-width non-break space */

	last = -1;
	while(getunicodeline(in, fields, buf)){
		code = getcode(fields[FIELD_CODE]);
		if (code >= NRUNES)
			fatal("code-point value too big: %x", code);
		if(code <= last)
			fatal("bad code sequence: %x then %x", last, code);
		last = code;

		/*
		 * check for ranges
		 */
		p = fields[FIELD_CATEGORY];
		if(strstr(fields[FIELD_NAME], ", First>") != NULL){
			if(!getunicodeline(in, fields2, buf2))
				fatal("range start at eof");
			if (strstr(fields2[FIELD_NAME], ", Last>") == NULL)
				fatal("range start not followed by range end");
			last = getcode(fields2[FIELD_CODE]);
			if(last <= code)
				fatal("range out of sequence: %x then %x", code, last);
			if(strcmp(p, fields2[FIELD_CATEGORY]) != 0)
				fatal("range with mismatched category");
		}

		/*
		 * set properties and conversions
		 */
		for (; code <= last; code++){
			if(p[0] == 'L')
				myisalpha[code] = 1;
			if(p[0] == 'Z')
				myisspace[code] = 1;

			if(strcmp(p, "Lu") == 0)
				myisupper[code] = 1;
			if(strcmp(p, "Ll") == 0)
				myislower[code] = 1;

			if(strcmp(p, "Lt") == 0)
				myistitle[code] = 1;

			if(strcmp(p, "Nd") == 0)
				myisdigit[code] = 1;

			/*
			 * when finding conversions, also need to mark
			 * upper/lower case, since some chars, like
			 * "III" (0x2162), aren't defined as letters but have a
			 * lower case mapping ("iii" (0x2172)).
			 */
			if(fields[FIELD_UPPER][0] != '\0'){
				mytoupper[code] = getcode(fields[FIELD_UPPER]);
			}
			if(fields[FIELD_LOWER][0] != '\0'){
				mytolower[code] = getcode(fields[FIELD_LOWER]);
			}
			if(fields[FIELD_TITLE][0] != '\0'){
				mytotitle[code] = getcode(fields[FIELD_TITLE]);
			}
		}
	}

	fclose(in);

	/*
	 * check for codes with no totitle mapping but a toupper mapping.
	 * these appear in UnicodeData-2.0.14.txt, but are almost certainly
	 * erroneous.
	 */
	for(i = 0; i < NRUNES; i++){
		if(mytotitle[i] == i
		&& mytoupper[i] != i
		&& !myistitle[i])
			fprintf(stderr, "warning: code=%.4x not istitle, totitle is same, toupper=%.4x\n", i, mytoupper[i]);
	}

	/*
	 * make sure isupper[c] is true if for some x toupper[x]  == c
	 * ditto for islower and istitle
	 */
	for(i = 0; i < NRUNES; i++) {
		if(mytoupper[i] != i)
			myisupper[mytoupper[i]] = 1;
		if(mytolower[i] != i)
			myislower[mytolower[i]] = 1;
		if(mytotitle[i] != i)
			myistitle[mytotitle[i]] = 1;
	}

	if(docheck){
		check();
	}else{
		mktables(argv[0], usepairs);
	}
	exit(0);
}

/*
 * generate a properties array for ranges, clearing those cases covered.
 * if force, generate one-entry ranges for singletons.
 */
static int
mkisrange(const char* label, char* prop, int force)
{
	int start, stop, some;

	/*
	 * first, the ranges
	 */
	some = 0;
	for(start = 0; start < NRUNES; ) {
		if(!prop[start]){
			start++;
			continue;
		}

		for(stop = start + 1; stop < NRUNES; stop++){
			if(!prop[stop]){
				break;
			}
			prop[stop] = 0;
		}
		if(force || stop != start + 1){
			if(!some){
				printf("static Rune __is%sr[] = {\n", label);
				some = 1;
			}
			prop[start] = 0;
			printf("\t0x%.4x, 0x%.4x,\n", start, stop - 1);
		}

		start = stop;
	}
	if(some)
		printf("};\n\n");
	return some;
}

/*
 * generate a mapping array for pairs with a skip between,
 * clearing those entries covered.
 */
static int
mkispair(const char *label, char *prop)
{
	int start, stop, some;

	some = 0;
	for(start = 0; start + 2 < NRUNES; ) {
		if(!prop[start]){
			start++;
			continue;
		}

		for(stop = start + 2; stop < NRUNES; stop += 2){
			if(!prop[stop]){
				break;
			}
			prop[stop] = 0;
		}
		if(stop != start + 2){
			if(!some){
				printf("static Rune __is%sp[] = {\n", label);
				some = 1;
			}
			prop[start] = 0;
			printf("\t0x%.4x, 0x%.4x,\n", start, stop - 2);
		}

		start = stop;
	}
	if(some)
		printf("};\n\n");
	return some;
}

/*
 * generate a properties array for singletons, clearing those cases covered.
 */
static int
mkissingle(const char *label, char *prop)
{
	int start, some;

	some = 0;
	for(start = 0; start < NRUNES; start++) {
		if(!prop[start]){
			continue;
		}

		if(!some){
			printf("static Rune __is%ss[] = {\n", label);
			some = 1;
		}
		prop[start] = 0;
		printf("\t0x%.4x,\n", start);
	}
	if(some)
		printf("};\n\n");
	return some;
}

/*
 * generate tables and a function for is<label>rune
 */
static void
mkis(const char* label, char* prop, int usepairs)
{
	int isr, isp, iss;

	isr = mkisrange(label, prop, 0);
	isp = 0;
	if(usepairs)
		isp = mkispair(label, prop);
	iss = mkissingle(label, prop);

	printf(
		"int\n"
		"is%srune(Rune c)\n"
		"{\n"
		"	Rune *p;\n"
		"\n",
		label);

	if(isr)
		printf(
			"	p = rbsearch(c, __is%sr, nelem(__is%sr)/2, 2);\n"
			"	if(p && c >= p[0] && c <= p[1])\n"
			"		return 1;\n",
			label, label);

	if(isp)
		printf(
			"	p = rbsearch(c, __is%sp, nelem(__is%sp)/2, 2);\n"
			"	if(p && c >= p[0] && c <= p[1] && !((c - p[0]) & 1))\n"
			"		return 1;\n",
			label, label);

	if(iss)
		printf(
			"	p = rbsearch(c, __is%ss, nelem(__is%ss), 1);\n"
			"	if(p && c == p[0])\n"
			"		return 1;\n",
			label, label);


	printf(
		"	return 0;\n"
		"}\n"
		"\n"
	);
}

/*
 * generate a mapping array for ranges, clearing those entries covered.
 * if force, generate one-entry ranges for singletons.
 */
static int
mktorange(const char* label, int* map, int force)
{
	int start, stop, delta, some;

	some = 0;
	for(start = 0; start < NRUNES; ) {
		if(map[start] == start){
			start++;
			continue;
		}

		delta = TO_DELTA(map[start], start);
		if(delta != (Rune)delta)
			fatal("bad map delta %d", delta);
		for(stop = start + 1; stop < NRUNES; stop++){
			if(TO_DELTA(map[stop], stop) != delta){
				break;
			}
			map[stop] = stop;
		}
		if(stop != start + 1){
			if(!some){
				printf("static Rune __to%sr[] = {\n", label);
				some = 1;
			}
			map[start] = start;
			printf("\t0x%.4x, 0x%.4x, %d,\n", start, stop - 1, delta);
		}

		start = stop;
	}
	if(some)
		printf("};\n\n");
	return some;
}

/*
 * generate a mapping array for pairs with a skip between,
 * clearing those entries covered.
 */
static int
mktopair(const char* label, int* map)
{
	int start, stop, delta, some;

	some = 0;
	for(start = 0; start + 2 < NRUNES; ) {
		if(map[start] == start){
			start++;
			continue;
		}

		delta = TO_DELTA(map[start], start);
		if(delta != (Rune)delta)
			fatal("bad map delta %d", delta);
		for(stop = start + 2; stop < NRUNES; stop += 2){
			if(TO_DELTA(map[stop], stop) != delta){
				break;
			}
			map[stop] = stop;
		}
		if(stop != start + 2){
			if(!some){
				printf("static Rune __to%sp[] = {\n", label);
				some = 1;
			}
			map[start] = start;
			printf("\t0x%.4x, 0x%.4x, %d,\n", start, stop - 2, delta);
		}

		start = stop;
	}
	if(some)
		printf("};\n\n");
	return some;
}

/*
 * generate a mapping array for singletons, clearing those entries covered.
 */
static int
mktosingle(const char* label, int* map)
{
	int start, delta, some;

	some = 0;
	for(start = 0; start < NRUNES; start++) {
		if(map[start] == start){
			continue;
		}

		delta = TO_DELTA(map[start], start);
		if(delta != (Rune)delta)
			fatal("bad map delta %d", delta);
		if(!some){
			printf("static Rune __to%ss[] = {\n", label);
			some = 1;
		}
		map[start] = start;
		printf("\t0x%.4x, %d,\n", start, delta);
	}
	if(some)
		printf("};\n\n");
	return some;
}

/*
 * generate tables and a function for to<label>rune
 */
static void
mkto(const char* label, int* map, int usepairs)
{
	int tor, top, tos;

	tor = mktorange(label, map, 0);
	top = 0;
	if(usepairs)
		top = mktopair(label, map);
	tos = mktosingle(label, map);

	printf(
		"Rune\n"
		"to%srune(Rune c)\n"
		"{\n"
		"	Rune *p;\n"
		"\n",
		label);

	if(tor)
		printf(
			"	p = rbsearch(c, __to%sr, nelem(__to%sr)/3, 3);\n"
			"	if(p && c >= p[0] && c <= p[1])\n"
			"		return c + p[2] - %d;\n",
			label, label, TO_OFFSET);

	if(top)
		printf(
			"	p = rbsearch(c, __to%sp, nelem(__to%sp)/3, 3);\n"
			"	if(p && c >= p[0] && c <= p[1] && !((c - p[0]) & 1))\n"
			"		return c + p[2] - %d;\n",
			label, label, TO_OFFSET);

	if(tos)
		printf(
			"	p = rbsearch(c, __to%ss, nelem(__to%ss)/2, 2);\n"
			"	if(p && c == p[0])\n"
			"		return c + p[1] - %d;\n",
			label, label, TO_OFFSET);


	printf(
		"	return c;\n"
		"}\n"
		"\n"
	);
}

// Make only range tables and a function for is<label>rune.
static void
mkisronly(const char* label, char* prop)
{
	mkisrange(label, prop, 1);
	printf(
		"int\n"
		"is%srune(Rune c)\n"
		"{\n"
		"	Rune *p;\n"
		"\n"
		"	p = rbsearch(c, __is%sr, nelem(__is%sr)/2, 2);\n"
		"	if(p && c >= p[0] && c <= p[1])\n"
		"		return 1;\n"
		"	return 0;\n"
		"}\n"
		"\n",
	        label, label, label);
}

/*
 * generate the body of runetype.
 * assumes there is a function Rune* rbsearch(Rune c, Rune *t, int n, int ne);
 */
static void
mktables(char *src, int usepairs)
{
	printf("/* generated automatically by mkrunetype.c from %s */\n\n", src);

	/*
	 * we special case the space and digit tables, since they are assumed
	 * to be small with several ranges.
	 */
	mkisronly("space", myisspace);
	mkisronly("digit", myisdigit);

	mkis("alpha", myisalpha, 0);
	mkis("upper", myisupper, usepairs);
	mkis("lower", myislower, usepairs);
	mkis("title", myistitle, usepairs);

	mkto("upper", mytoupper, usepairs);
	mkto("lower", mytolower, usepairs);
	mkto("title", mytotitle, usepairs);
}

/*
 * find differences between the newly generated tables and current runetypes.
 */
static void
check(void)
{
	int i;

	for(i = 0; i < NRUNES; i++){
		if(isdigitrune(i) != myisdigit[i])
			fprintf(stderr, "isdigit diff at %x: runetype=%x, unicode=%x\n",
				i, isdigitrune(i), myisdigit[i]);

		if(isspacerune(i) != myisspace[i])
			fprintf(stderr, "isspace diff at %x: runetype=%x, unicode=%x\n",
				i, isspacerune(i), myisspace[i]);

		if(isupperrune(i) != myisupper[i])
			fprintf(stderr, "isupper diff at %x: runetype=%x, unicode=%x\n",
				i, isupperrune(i), myisupper[i]);

		if(islowerrune(i) != myislower[i])
			fprintf(stderr, "islower diff at %x: runetype=%x, unicode=%x\n",
				i, islowerrune(i), myislower[i]);

		if(isalpharune(i) != myisalpha[i])
			fprintf(stderr, "isalpha diff at %x: runetype=%x, unicode=%x\n",
				i, isalpharune(i), myisalpha[i]);

		if(toupperrune(i) != mytoupper[i])
			fprintf(stderr, "toupper diff at %x: runetype=%x, unicode=%x\n",
				i, toupperrune(i), mytoupper[i]);

		if(tolowerrune(i) != mytolower[i])
			fprintf(stderr, "tolower diff at %x: runetype=%x, unicode=%x\n",
				i, tolowerrune(i), mytolower[i]);

		if(istitlerune(i) != myistitle[i])
			fprintf(stderr, "istitle diff at %x: runetype=%x, unicode=%x\n",
				i, istitlerune(i), myistitle[i]);

		if(totitlerune(i) != mytotitle[i])
			fprintf(stderr, "totitle diff at %x: runetype=%x, unicode=%x\n",
				i, totitlerune(i), mytotitle[i]);


	}
}

static int
mygetfields(char **fields, int nfields, char *str, const char *delim)
{
	int nf;

	fields[0] = str;
	nf = 1;
	if(nf >= nfields)
		return nf;

	for(; *str; str++){
		if(strchr(delim, *str) != NULL){
			*str = '\0';
			fields[nf++] = str + 1;
			if(nf >= nfields)
				break;
		}
	}
	return nf;
}

static int
getunicodeline(FILE *in, char **fields, char *buf)
{
	char *p;

	if(fgets(buf, MAX_LINE, in) == NULL)
		return 0;

	p = strchr(buf, '\n');
	if (p == NULL)
		fatal("line too long");
	*p = '\0';

	if (mygetfields(fields, NFIELDS + 1, buf, ";") != NFIELDS)
		fatal("bad number of fields");

	return 1;
}

static int
getcode(char *s)
{
	int i, code;

	code = 0;
	i = 0;
	/* Parse a hex number */
	while(s[i]) {
		code <<= 4;
		if(s[i] >= '0' && s[i] <= '9')
			code += s[i] - '0';
		else if(s[i] >= 'A' && s[i] <= 'F')
			code += s[i] - 'A' + 10;
		else
			fatal("bad code char '%c'", s[i]);
		i++;
	}
	return code;
}

static void
fatal(const char *fmt, ...)
{
	va_list arg;

	fprintf(stderr, "%s: fatal error: ", argv0);
	va_start(arg, fmt);
	vfprintf(stderr, fmt, arg);
	va_end(arg);
	fprintf(stderr, "\n");

	exit(1);
}
