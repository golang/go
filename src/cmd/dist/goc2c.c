// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "a.h"

/*
 * Translate a .goc file into a .c file.  A .goc file is a combination
 * of a limited form of Go with C.
 */

/*
	package PACKAGENAME
	{# line}
	func NAME([NAME TYPE { , NAME TYPE }]) [(NAME TYPE { , NAME TYPE })] \{
	  C code with proper brace nesting
	\}
*/

/*
 * We generate C code which implements the function such that it can
 * be called from Go and executes the C code.
 */

static char *input;
static Buf *output;
#define EOF -1

enum
{
	use64bitint = 1,
};

static int
xgetchar(void)
{
	int c;
	
	c = *input;
	if(c == 0)
		return EOF;
	input++;
	return c;
}

static void
xungetc(void)
{
	input--;
}

static void
xputchar(char c)
{
	bwrite(output, &c, 1);
}

static int
xisspace(int c)
{
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/* Whether we're emitting for gcc */
static int gcc;

/* File and line number */
static const char *file;
static const char *final_file;
static unsigned int lineno;

/* List of names and types.  */
struct params {
	struct params *next;
	char *name;
	char *type;
};

/* index into type_table */
enum {
	Bool,
	Float,
	Int,
	Uint,
	Uintptr,
	String,
	Slice,
	Eface,
	Complex128,
	Float32,
	Float64,
};

static struct {
	char *name;
	int size;
	int rnd; // alignment
} type_table[] = {
	/* 
	 * variable sized first, for easy replacement.
	 * order matches enum above.
	 * default is 32-bit architecture sizes.
	 * spelling as in package runtime, so intgo/uintgo not int/uint.
	 */
	{"bool",	1},
	{"float",	4},
	{"intgo",		4},
	{"uintgo",	4},
	{"uintptr",	4},
	{"String",	8},
	{"Slice",	12},
	{"Eface",	8},
	{"Complex128", 16},

	/* fixed size */
	{"float32",	4},
	{"float64",	8},
	{"byte",	1},
	{"int8",	1},
	{"uint8",	1},
	{"int16",	2},
	{"uint16",	2},
	{"int32",	4},
	{"rune",	4},
	{"uint32",	4},
	{"int64",	8},
	{"uint64",	8},

	{nil, 0},
};

/* Fixed structure alignment (non-gcc only) */
int structround = 4;

/* Unexpected EOF.  */
static void
bad_eof(void)
{
	fatal("%s:%d: unexpected EOF\n", file, lineno);
}

/* Free a list of parameters.  */
static void
free_params(struct params *p)
{
	while (p != nil) {
		struct params *next;

		next = p->next;
		xfree(p->name);
		xfree(p->type);
		xfree(p);
		p = next;
	}
}

/* Read a character, tracking lineno.  */
static int
getchar_update_lineno(void)
{
	int c;

	c = xgetchar();
	if (c == '\n')
		++lineno;
	return c;
}

/* Read a character, giving an error on EOF, tracking lineno.  */
static int
getchar_no_eof(void)
{
	int c;

	c = getchar_update_lineno();
	if (c == EOF)
		bad_eof();
	return c;
}

/* Read a character, skipping comments.  */
static int
getchar_skipping_comments(void)
{
	int c;

	while (1) {
		c = getchar_update_lineno();
		if (c != '/')
			return c;

		c = xgetchar();
		if (c == '/') {
			do {
				c = getchar_update_lineno();
			} while (c != EOF && c != '\n');
			return c;
		} else if (c == '*') {
			while (1) {
				c = getchar_update_lineno();
				if (c == EOF)
					return EOF;
				if (c == '*') {
					do {
						c = getchar_update_lineno();
					} while (c == '*');
					if (c == '/')
						break;
				}
			}
		} else {
			xungetc();
			return '/';
		}
	}
}

/*
 * Read and return a token.  Tokens are string or character literals
 * or else delimited by whitespace or by [(),{}].
 * The latter are all returned as single characters.
 */
static char *
read_token(void)
{
	int c, q;
	char *buf;
	unsigned int alc, off;
	char* delims = "(),{}";

	while (1) {
		c = getchar_skipping_comments();
		if (c == EOF)
			return nil;
		if (!xisspace(c))
			break;
	}
	alc = 16;
	buf = xmalloc(alc + 1);
	off = 0;
	if(c == '"' || c == '\'') {
		q = c;
		buf[off] = c;
		++off;
		while (1) {
			if (off+2 >= alc) { // room for c and maybe next char
				alc *= 2;
				buf = xrealloc(buf, alc + 1);
			}
			c = getchar_no_eof();
			buf[off] = c;
			++off;
			if(c == q)
				break;
			if(c == '\\') {
				buf[off] = getchar_no_eof();
				++off;
			}
		}
	} else if (xstrrchr(delims, c) != nil) {
		buf[off] = c;
		++off;
	} else {
		while (1) {
			if (off >= alc) {
				alc *= 2;
				buf = xrealloc(buf, alc + 1);
			}
			buf[off] = c;
			++off;
			c = getchar_skipping_comments();
			if (c == EOF)
				break;
			if (xisspace(c) || xstrrchr(delims, c) != nil) {
				if (c == '\n')
					lineno--;
				xungetc();
				break;
			}
		}
	}
	buf[off] = '\0';
	return buf;
}

/* Read a token, giving an error on EOF.  */
static char *
read_token_no_eof(void)
{
	char *token = read_token();
	if (token == nil)
		bad_eof();
	return token;
}

/* Read the package clause, and return the package name.  */
static char *
read_package(void)
{
	char *token;

	token = read_token_no_eof();
	if (token == nil)
		fatal("%s:%d: no token\n", file, lineno);
	if (!streq(token, "package")) {
		fatal("%s:%d: expected \"package\", got \"%s\"\n",
			file, lineno, token);
	}
	return read_token_no_eof();
}

/* Read and copy preprocessor lines.  */
static void
read_preprocessor_lines(void)
{
	int first;
	
	first = 1;
	while (1) {
		int c;

		do {
			c = getchar_skipping_comments();
		} while (xisspace(c));
		if (c != '#') {
			xungetc();
			break;
		}
		if(first) {
			first = 0;
			xputchar('\n');
		}
		xputchar(c);
		do {
			c = getchar_update_lineno();
			xputchar(c);
		} while (c != '\n');
	}
}

/*
 * Read a type in Go syntax and return a type in C syntax.  We only
 * permit basic types and pointers.
 */
static char *
read_type(void)
{
	char *p, *op, *q;
	int pointer_count;
	unsigned int len;

	p = read_token_no_eof();
	if (*p != '*' && !streq(p, "int") && !streq(p, "uint"))
		return p;
	op = p;
	pointer_count = 0;
	while (*p == '*') {
		++pointer_count;
		++p;
	}
	len = xstrlen(p);
	q = xmalloc(len + 2 + pointer_count + 1);
	xmemmove(q, p, len);

	// Turn int/uint into intgo/uintgo.
	if((len == 3 && xmemcmp(q, "int", 3) == 0) || (len == 4 && xmemcmp(q, "uint", 4) == 0)) {
		q[len++] = 'g';
		q[len++] = 'o';
	}

	while (pointer_count-- > 0)
		q[len++] = '*';
	
	q[len] = '\0';
	xfree(op);
	return q;
}

/* Return the size of the given type. */
static int
type_size(char *p, int *rnd)
{
	int i;

	if(p[xstrlen(p)-1] == '*') {
		*rnd = type_table[Uintptr].rnd;
		return type_table[Uintptr].size;
	}

	if(streq(p, "Iface"))
		p = "Eface";

	for(i=0; type_table[i].name; i++)
		if(streq(type_table[i].name, p)) {
			*rnd = type_table[i].rnd;
			return type_table[i].size;
		}
	fatal("%s:%d: unknown type %s\n", file, lineno, p);
	return 0;
}

/*
 * Read a list of parameters.  Each parameter is a name and a type.
 * The list ends with a ')'.  We have already read the '('.
 */
static struct params *
read_params(int *poffset)
{
	char *token;
	struct params *ret, **pp, *p;
	int offset, size, rnd;

	ret = nil;
	pp = &ret;
	token = read_token_no_eof();
	offset = 0;
	if (!streq(token, ")")) {
		while (1) {
			p = xmalloc(sizeof(struct params));
			p->name = token;
			p->next = nil;
			*pp = p;
			pp = &p->next;

			if(streq(token, "...")) {
				p->type = xstrdup("");
			} else {
				p->type = read_type();
				rnd = 0;
				size = type_size(p->type, &rnd);
				if(rnd > structround)
					rnd = structround;
				if(offset%rnd)
					offset += rnd - offset%rnd;
				offset += size;
			}

			token = read_token_no_eof();
			if (!streq(token, ","))
				break;
			token = read_token_no_eof();
		}
	}
	if (!streq(token, ")")) {
		fatal("%s:%d: expected '('\n",
			file, lineno);
	}
	if (poffset != nil)
		*poffset = offset;
	return ret;
}

/*
 * Read a function header.  This reads up to and including the initial
 * '{' character.  Returns 1 if it read a header, 0 at EOF.
 */
static int
read_func_header(char **name, struct params **params, int *paramwid, struct params **rets)
{
	int lastline;
	char *token;

	lastline = -1;
	while (1) {
		read_preprocessor_lines();
		token = read_token();
		if (token == nil)
			return 0;
		if (streq(token, "func")) {
			if(lastline != -1)
				bwritef(output, "\n");
			break;
		}
		if (lastline != lineno) {
			if (lastline == lineno-1)
				bwritef(output, "\n");
			else
				bwritef(output, "\n#line %d \"%s\"\n", lineno, final_file);
			lastline = lineno;
		}
		bwritef(output, "%s ", token);
	}

	*name = read_token_no_eof();

	token = read_token();
	if (token == nil || !streq(token, "(")) {
		fatal("%s:%d: expected \"(\"\n",
			file, lineno);
	}
	*params = read_params(paramwid);

	token = read_token();
	if (token == nil || !streq(token, "("))
		*rets = nil;
	else {
		*rets = read_params(nil);
		token = read_token();
	}
	if (token == nil || !streq(token, "{")) {
		fatal("%s:%d: expected \"{\"\n",
			file, lineno);
	}
	return 1;
}

/* Write out parameters.  */
static void
write_params(struct params *params, int *first)
{
	struct params *p;

	for (p = params; p != nil; p = p->next) {
		if (*first)
			*first = 0;
		else
			bwritef(output, ", ");
		bwritef(output, "%s %s", p->type, p->name);
	}
}

/* Write a 6g function header.  */
static void
write_6g_func_header(char *package, char *name, struct params *params,
		     int paramwid, struct params *rets)
{
	int first, n;
	struct params *p;

	bwritef(output, "void\n");
	if(!contains(name, "·"))
		bwritef(output, "%s·", package);
	bwritef(output, "%s(", name);

	first = 1;
	write_params(params, &first);

	/* insert padding to align output struct */
	if(rets != nil && paramwid%structround != 0) {
		n = structround - paramwid%structround;
		if(n & 1)
			bwritef(output, ", uint8");
		if(n & 2)
			bwritef(output, ", uint16");
		if(n & 4)
			bwritef(output, ", uint32");
	}

	write_params(rets, &first);
	bwritef(output, ")\n{\n");
	
	for (p = rets; p != nil; p = p->next) {
		if(streq(p->name, "..."))
			continue;
		if(streq(p->type, "Slice"))
			bwritef(output, "\t%s.array = 0;\n\t%s.len = 0;\n\t%s.cap = 0;\n", p->name, p->name, p->name);
		else if(streq(p->type, "String"))
			bwritef(output, "\t%s.str = 0;\n\t%s.len = 0;\n", p->name, p->name);
		else if(streq(p->type, "Eface"))
			bwritef(output, "\t%s.type = 0;\n\t%s.data = 0;\n", p->name, p->name);
		else if(streq(p->type, "Iface"))
			bwritef(output, "\t%s.tab = 0;\n\t%s.data = 0;\n", p->name, p->name);
		else if(streq(p->type, "Complex128"))
			bwritef(output, "\t%s.real = 0;\n\t%s.imag = 0;\n", p->name, p->name);
		else
			bwritef(output, "\t%s = 0;\n", p->name);
		bwritef(output, "\tFLUSH(&%s);\n", p->name);
	}
}

/* Write a 6g function trailer.  */
static void
write_6g_func_trailer(struct params *rets)
{
	struct params *p;

	for (p = rets; p != nil; p = p->next)
		if(!streq(p->name, "..."))
			bwritef(output, "\tFLUSH(&%s);\n", p->name);
	bwritef(output, "}\n");
}

/* Define the gcc function return type if necessary.  */
static void
define_gcc_return_type(char *package, char *name, struct params *rets)
{
	struct params *p;

	if (rets == nil || rets->next == nil)
		return;
	bwritef(output, "struct %s_%s_ret {\n", package, name);
	for (p = rets; p != nil; p = p->next)
		bwritef(output, "  %s %s;\n", p->type, p->name);
	bwritef(output, "};\n");
}

/* Write out the gcc function return type.  */
static void
write_gcc_return_type(char *package, char *name, struct params *rets)
{
	if (rets == nil)
		bwritef(output, "void");
	else if (rets->next == nil)
		bwritef(output, "%s", rets->type);
	else
		bwritef(output, "struct %s_%s_ret", package, name);
}

/* Write out a gcc function header.  */
static void
write_gcc_func_header(char *package, char *name, struct params *params,
		      struct params *rets)
{
	int first;
	struct params *p;

	define_gcc_return_type(package, name, rets);
	write_gcc_return_type(package, name, rets);
	bwritef(output, " %s_%s(", package, name);
	first = 1;
	write_params(params, &first);
	bwritef(output, ") asm (\"%s.%s\");\n", package, name);
	write_gcc_return_type(package, name, rets);
	bwritef(output, " %s_%s(", package, name);
	first = 1;
	write_params(params, &first);
	bwritef(output, ")\n{\n");
	for (p = rets; p != nil; p = p->next)
		bwritef(output, "  %s %s;\n", p->type, p->name);
}

/* Write out a gcc function trailer.  */
static void
write_gcc_func_trailer(char *package, char *name, struct params *rets)
{
	if (rets == nil) {
		// nothing to do
	}
	else if (rets->next == nil)
		bwritef(output, "return %s;\n", rets->name);
	else {
		struct params *p;

		bwritef(output, "  {\n    struct %s_%s_ret __ret;\n", package, name);
		for (p = rets; p != nil; p = p->next)
			bwritef(output, "    __ret.%s = %s;\n", p->name, p->name);
		bwritef(output, "    return __ret;\n  }\n");
	}
	bwritef(output, "}\n");
}

/* Write out a function header.  */
static void
write_func_header(char *package, char *name,
		  struct params *params, int paramwid,
		  struct params *rets)
{
	if (gcc)
		write_gcc_func_header(package, name, params, rets);
	else
		write_6g_func_header(package, name, params, paramwid, rets);
	bwritef(output, "#line %d \"%s\"\n", lineno, final_file);
}

/* Write out a function trailer.  */
static void
write_func_trailer(char *package, char *name,
		   struct params *rets)
{
	if (gcc)
		write_gcc_func_trailer(package, name, rets);
	else
		write_6g_func_trailer(rets);
}

/*
 * Read and write the body of the function, ending in an unnested }
 * (which is read but not written).
 */
static void
copy_body(void)
{
	int nesting = 0;
	while (1) {
		int c;

		c = getchar_no_eof();
		if (c == '}' && nesting == 0)
			return;
		xputchar(c);
		switch (c) {
		default:
			break;
		case '{':
			++nesting;
			break;
		case '}':
			--nesting;
			break;
		case '/':
			c = getchar_update_lineno();
			xputchar(c);
			if (c == '/') {
				do {
					c = getchar_no_eof();
					xputchar(c);
				} while (c != '\n');
			} else if (c == '*') {
				while (1) {
					c = getchar_no_eof();
					xputchar(c);
					if (c == '*') {
						do {
							c = getchar_no_eof();
							xputchar(c);
						} while (c == '*');
						if (c == '/')
							break;
					}
				}
			}
			break;
		case '"':
		case '\'':
			{
				int delim = c;
				do {
					c = getchar_no_eof();
					xputchar(c);
					if (c == '\\') {
						c = getchar_no_eof();
						xputchar(c);
						c = '\0';
					}
				} while (c != delim);
			}
			break;
		}
	}
}

/* Process the entire file.  */
static void
process_file(void)
{
	char *package, *name, *p, *n;
	struct params *params, *rets;
	int paramwid;

	package = read_package();
	read_preprocessor_lines();
	while (read_func_header(&name, &params, &paramwid, &rets)) {
		// name may have a package override already
		n = xstrstr(name, "·");
		if(n != nil) {
			p = xmalloc(n - name + 1);
			xmemmove(p, name, n - name);
			p[n - name] = 0;
			n += xstrlen("·");
		} else {
			p = package;
			n = name;
		}
		write_func_header(p, n, params, paramwid, rets);
		copy_body();
		write_func_trailer(p, n, rets);
		xfree(name);
		if(p != package) xfree(p);
		free_params(params);
		free_params(rets);
	}
	xfree(package);
}

void
goc2c(char *goc, char *goc_final, char *c)
{
	int i;
	Buf in, out;
	
	binit(&in);
	binit(&out);
	
	file = goc;
	final_file = goc_final;
	readfile(&in, goc);

	// TODO: set gcc=1 when using gcc

	if(!gcc) {
		if(streq(goarch, "amd64")) {
			type_table[Uintptr].size = 8;
			if(use64bitint) {
				type_table[Int].size = 8;
			} else {
				type_table[Int].size = 4;
			}
			structround = 8;
		} else if(streq(goarch, "amd64p32")) {
			type_table[Uintptr].size = 4;
			type_table[Int].size = 4;
			structround = 8;
		} else {
			// NOTE: These are set in the initializer,
			// but they might have been changed by a
			// previous invocation of goc2c, so we have
			// to restore them.
			type_table[Uintptr].size = 4;
			type_table[Int].size = 4;
			structround = 4;
		}

		type_table[Uint].size = type_table[Int].size;
		type_table[Slice].size = type_table[Uintptr].size+2*type_table[Int].size;
		type_table[Eface].size = 2*type_table[Uintptr].size;
		type_table[String].size = 2*type_table[Uintptr].size;

		for(i=0; i<nelem(type_table); i++)
			type_table[i].rnd = type_table[i].size;

		type_table[String].rnd = type_table[Uintptr].rnd;
		type_table[Slice].rnd = type_table[Uintptr].rnd;
		type_table[Eface].rnd = type_table[Uintptr].rnd;
		type_table[Complex128].rnd = type_table[Float64].rnd;
	}

	bprintf(&out, "// auto generated by go tool dist\n// goos=%s goarch=%s\n\n", goos, goarch);
	if(streq(goos, "linux")) {
		bwritestr(&out, "// +build !android\n\n");
	}
	
	input = bstr(&in);
	output = &out;

	lineno = 1;
	process_file();
	
	writefile(&out, c, 0);
}
