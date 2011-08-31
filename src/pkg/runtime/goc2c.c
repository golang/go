// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

#include <u.h>
#include <stdio.h>
#include <libc.h>

/* Whether we're emitting for gcc */
static int gcc;

/* File and line number */
static const char *file;
static unsigned int lineno = 1;

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
};

static struct {
	char *name;
	int size;
} type_table[] = {
	/* variable sized first, for easy replacement */
	/* order matches enum above */
	/* default is 32-bit architecture sizes */
	"bool",		1,
	"float",	4,
	"int",		4,
	"uint",		4,
	"uintptr",	4,
	"String",	8,
	"Slice",	12,
	"Eface",	8,

	/* fixed size */
	"float32",	4,
	"float64",	8,
	"byte",		1,
	"int8",		1,
	"uint8",	1,
	"int16",	2,
	"uint16",	2,
	"int32",	4,
	"uint32",	4,
	"int64",	8,
	"uint64",	8,

	NULL,
};

/* Fixed structure alignment (non-gcc only) */
int structround = 4;

/* Unexpected EOF.  */
static void
bad_eof(void)
{
	sysfatal("%s:%ud: unexpected EOF\n", file, lineno);
}

/* Out of memory.  */
static void
bad_mem(void)
{
	sysfatal("%s:%ud: out of memory\n", file, lineno);
}

/* Allocate memory without fail.  */
static void *
xmalloc(unsigned int size)
{
	void *ret = malloc(size);
	if (ret == NULL)
		bad_mem();
	return ret;
}

/* Reallocate memory without fail.  */
static void*
xrealloc(void *buf, unsigned int size)
{
	void *ret = realloc(buf, size);
	if (ret == NULL)
		bad_mem();
	return ret;
}

/* Free a list of parameters.  */
static void
free_params(struct params *p)
{
	while (p != NULL) {
		struct params *next;

		next = p->next;
		free(p->name);
		free(p->type);
		free(p);
		p = next;
	}
}

/* Read a character, tracking lineno.  */
static int
getchar_update_lineno(void)
{
	int c;

	c = getchar();
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

		c = getchar();
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
			ungetc(c, stdin);
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
	const char* delims = "(),{}";

	while (1) {
		c = getchar_skipping_comments();
		if (c == EOF)
			return NULL;
		if (!isspace(c))
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
	} else if (strchr(delims, c) != NULL) {
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
			if (isspace(c) || strchr(delims, c) != NULL) {
				if (c == '\n')
					lineno--;
				ungetc(c, stdin);
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
	if (token == NULL)
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
		sysfatal("%s:%ud: no token\n", file, lineno);
	if (strcmp(token, "package") != 0) {
		sysfatal("%s:%ud: expected \"package\", got \"%s\"\n",
			file, lineno, token);
	}
	return read_token_no_eof();
}

/* Read and copy preprocessor lines.  */
static void
read_preprocessor_lines(void)
{
	while (1) {
		int c;

		do {
			c = getchar_skipping_comments();
		} while (isspace(c));
		if (c != '#') {
			ungetc(c, stdin);
			break;
		}
		putchar(c);
		do {
			c = getchar_update_lineno();
			putchar(c);
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
	if (*p != '*')
		return p;
	op = p;
	pointer_count = 0;
	while (*p == '*') {
		++pointer_count;
		++p;
	}
	len = strlen(p);
	q = xmalloc(len + pointer_count + 1);
	memcpy(q, p, len);
	while (pointer_count > 0) {
		q[len] = '*';
		++len;
		--pointer_count;
	}
	q[len] = '\0';
	free(op);
	return q;
}

/* Return the size of the given type. */
static int
type_size(char *p)
{
	int i;

	if(p[strlen(p)-1] == '*')
		return type_table[Uintptr].size;

	for(i=0; type_table[i].name; i++)
		if(strcmp(type_table[i].name, p) == 0)
			return type_table[i].size;
	sysfatal("%s:%ud: unknown type %s\n", file, lineno, p);
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

	ret = NULL;
	pp = &ret;
	token = read_token_no_eof();
	offset = 0;
	if (strcmp(token, ")") != 0) {
		while (1) {
			p = xmalloc(sizeof(struct params));
			p->name = token;
			p->type = read_type();
			p->next = NULL;
			*pp = p;
			pp = &p->next;

			size = type_size(p->type);
			rnd = size;
			if(rnd > structround)
				rnd = structround;
			if(offset%rnd)
				offset += rnd - offset%rnd;
			offset += size;

			token = read_token_no_eof();
			if (strcmp(token, ",") != 0)
				break;
			token = read_token_no_eof();
		}
	}
	if (strcmp(token, ")") != 0) {
		sysfatal("%s:%ud: expected '('\n",
			file, lineno);
	}
	if (poffset != NULL)
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
		token = read_token();
		if (token == NULL)
			return 0;
		if (strcmp(token, "func") == 0) {
			if(lastline != -1)
				printf("\n");
			break;
		}
		if (lastline != lineno) {
			if (lastline == lineno-1)
				printf("\n");
			else
				printf("\n#line %d \"%s\"\n", lineno, file);
			lastline = lineno;
		}
		printf("%s ", token);
	}

	*name = read_token_no_eof();

	token = read_token();
	if (token == NULL || strcmp(token, "(") != 0) {
		sysfatal("%s:%ud: expected \"(\"\n",
			file, lineno);
	}
	*params = read_params(paramwid);

	token = read_token();
	if (token == NULL || strcmp(token, "(") != 0)
		*rets = NULL;
	else {
		*rets = read_params(NULL);
		token = read_token();
	}
	if (token == NULL || strcmp(token, "{") != 0) {
		sysfatal("%s:%ud: expected \"{\"\n",
			file, lineno);
	}
	return 1;
}

/* Write out parameters.  */
static void
write_params(struct params *params, int *first)
{
	struct params *p;

	for (p = params; p != NULL; p = p->next) {
		if (*first)
			*first = 0;
		else
			printf(", ");
		printf("%s %s", p->type, p->name);
	}
}

/* Write a 6g function header.  */
static void
write_6g_func_header(char *package, char *name, struct params *params,
		     int paramwid, struct params *rets)
{
	int first, n;

	printf("void\n%s·%s(", package, name);
	first = 1;
	write_params(params, &first);

	/* insert padding to align output struct */
	if(rets != NULL && paramwid%structround != 0) {
		n = structround - paramwid%structround;
		if(n & 1)
			printf(", uint8");
		if(n & 2)
			printf(", uint16");
		if(n & 4)
			printf(", uint32");
	}

	write_params(rets, &first);
	printf(")\n{\n");
}

/* Write a 6g function trailer.  */
static void
write_6g_func_trailer(struct params *rets)
{
	struct params *p;

	for (p = rets; p != NULL; p = p->next)
		printf("\tFLUSH(&%s);\n", p->name);
	printf("}\n");
}

/* Define the gcc function return type if necessary.  */
static void
define_gcc_return_type(char *package, char *name, struct params *rets)
{
	struct params *p;

	if (rets == NULL || rets->next == NULL)
		return;
	printf("struct %s_%s_ret {\n", package, name);
	for (p = rets; p != NULL; p = p->next)
		printf("  %s %s;\n", p->type, p->name);
	printf("};\n");
}

/* Write out the gcc function return type.  */
static void
write_gcc_return_type(char *package, char *name, struct params *rets)
{
	if (rets == NULL)
		printf("void");
	else if (rets->next == NULL)
		printf("%s", rets->type);
	else
		printf("struct %s_%s_ret", package, name);
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
	printf(" %s_%s(", package, name);
	first = 1;
	write_params(params, &first);
	printf(") asm (\"%s.%s\");\n", package, name);
	write_gcc_return_type(package, name, rets);
	printf(" %s_%s(", package, name);
	first = 1;
	write_params(params, &first);
	printf(")\n{\n");
	for (p = rets; p != NULL; p = p->next)
		printf("  %s %s;\n", p->type, p->name);
}

/* Write out a gcc function trailer.  */
static void
write_gcc_func_trailer(char *package, char *name, struct params *rets)
{
	if (rets == NULL)
		;
	else if (rets->next == NULL)
		printf("return %s;\n", rets->name);
	else {
		struct params *p;

		printf("  {\n    struct %s_%s_ret __ret;\n", package, name);
		for (p = rets; p != NULL; p = p->next)
			printf("    __ret.%s = %s;\n", p->name, p->name);
		printf("    return __ret;\n  }\n");
	}
	printf("}\n");
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
	printf("#line %d \"%s\"\n", lineno, file);
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
		putchar(c);
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
			putchar(c);
			if (c == '/') {
				do {
					c = getchar_no_eof();
					putchar(c);
				} while (c != '\n');
			} else if (c == '*') {
				while (1) {
					c = getchar_no_eof();
					putchar(c);
					if (c == '*') {
						do {
							c = getchar_no_eof();
							putchar(c);
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
					putchar(c);
					if (c == '\\') {
						c = getchar_no_eof();
						putchar(c);
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
	char *package, *name;
	struct params *params, *rets;
	int paramwid;

	package = read_package();
	read_preprocessor_lines();
	while (read_func_header(&name, &params, &paramwid, &rets)) {
		write_func_header(package, name, params, paramwid, rets);
		copy_body();
		write_func_trailer(package, name, rets);
		free(name);
		free_params(params);
		free_params(rets);
	}
	free(package);
}

static void
usage(void)
{
	sysfatal("Usage: goc2c [--6g | --gc] [file]\n");
}

void
main(int argc, char **argv)
{
	char *goarch;

	argv0 = argv[0];
	while(argc > 1 && argv[1][0] == '-') {
		if(strcmp(argv[1], "-") == 0)
			break;
		if(strcmp(argv[1], "--6g") == 0)
			gcc = 0;
		else if(strcmp(argv[1], "--gcc") == 0)
			gcc = 1;
		else
			usage();
		argc--;
		argv++;
	}

	if(argc <= 1 || strcmp(argv[1], "-") == 0) {
		file = "<stdin>";
		process_file();
		exits(0);
	}

	if(argc > 2)
		usage();

	file = argv[1];
	if(freopen(file, "r", stdin) == 0) {
		sysfatal("open %s: %r\n", file);
	}

	if(!gcc) {
		// 6g etc; update size table
		goarch = getenv("GOARCH");
		if(goarch != NULL && strcmp(goarch, "amd64") == 0) {
			type_table[Uintptr].size = 8;
			type_table[String].size = 16;
			type_table[Slice].size = 8+4+4;
			type_table[Eface].size = 8+8;
			structround = 8;
		}
	}

	process_file();
	exits(0);
}
