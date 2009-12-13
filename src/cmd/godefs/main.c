// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Godefs takes as input a host-compilable C file that includes
// standard system headers.  From that input file, it generates
// a standalone (no #includes) C or Go file containing equivalent
// definitions.
//
// The input C file is expected to define new types and enumerated
// constants whose names begin with $ (a legal identifier character
// in gcc).  The output is the standalone definitions of those names,
// with the $ removed.
//
// For example, if this is x.c:
//
//	#include <sys/stat.h>
//
//	typedef struct timespec $Timespec;
//	typedef struct stat $Stat;
//	enum {
//		$S_IFMT = S_IFMT,
//		$S_IFIFO = S_IFIFO,
//		$S_IFCHR = S_IFCHR,
//	};
//
// then "godefs x.c" generates:
//
//	// godefs x.c
//
//	// MACHINE GENERATED - DO NOT EDIT.
//
//	// Constants
//	enum {
//		S_IFMT = 0xf000,
//		S_IFIFO = 0x1000,
//		S_IFCHR = 0x2000,
//	};
//
//	// Types
//	#pragma pack on
//
//	typedef struct Timespec Timespec;
//	struct Timespec {
//		int32 tv_sec;
//		int32 tv_nsec;
//	};
//
//	typedef struct Stat Stat;
//	struct Stat {
//		int32 st_dev;
//		uint32 st_ino;
//		uint16 st_mode;
//		uint16 st_nlink;
//		uint32 st_uid;
//		uint32 st_gid;
//		int32 st_rdev;
//		Timespec st_atimespec;
//		Timespec st_mtimespec;
//		Timespec st_ctimespec;
//		int64 st_size;
//		int64 st_blocks;
//		int32 st_blksize;
//		uint32 st_flags;
//		uint32 st_gen;
//		int32 st_lspare;
//		int64 st_qspare[2];
//	};
//	#pragma pack off
//
// The -g flag to godefs causes it to generate Go output, not C.
// In the Go output, struct fields have leading xx_ prefixes removed
// and the first character capitalized (exported).
//
// Godefs works by invoking gcc to compile the given input file
// and then parses the debug info embedded in the assembly output.
// This is far easier than reading system headers on most machines.
//
// The -c flag sets the compiler (default "gcc").
//
// The -f flag adds a flag to pass to the compiler (e.g., -f -m64).

#include "a.h"

#ifdef __MINGW32__
int
spawn(char *prog, char **argv)
{
	return _spawnvp(P_NOWAIT, prog, (const char**)argv);
}
#undef waitfor
void
waitfor(int pid)
{
	_cwait(0, pid, 0);
}
#else
int
spawn(char *prog, char **argv)
{
	int pid = fork();
	if(pid < 0)
		sysfatal("fork: %r");
	if(pid == 0) {
		exec(argv[0], argv);
		fprint(2, "exec gcc: %r\n");
		exit(1);
	}
	return pid;
}
#endif

void
usage(void)
{
	fprint(2, "usage: godefs [-g package] [-c cc] [-f cc-arg] [defs.c ...]\n");
	exit(1);
}

int gotypefmt(Fmt*);
int ctypefmt(Fmt*);
int prefixlen(Type*);
int cutprefix(char*);

Lang go =
{
	"const (\n",
	"\t%s = %#llx;\n",
	")\n",

	"type",
	"\n",

	"type %s struct {\n",
	"type %s struct {\n",
	"\tPad%d [%d]byte;\n",
	"}\n",

	gotypefmt,
};

Lang c =
{
	"enum {\n",
	"\t%s = %#llx,\n",
	"};\n",

	"typedef",
	";\n",

	"typedef struct %s %s;\nstruct %s {\n",
	"typedef union %s %s;\nunion %s {\n",
	"\tbyte pad%d[%d];\n",
	"};\n",

	ctypefmt,
};

char *pkg;

int oargc;
char **oargv;
Lang *lang = &c;

Const *con;
int ncon;

Type **typ;
int ntyp;

void
waitforgcc(void)
{
	waitpid();
}

void
main(int argc, char **argv)
{
	int p[2], pid, i, j, n, off, npad, prefix;
	char **av, *q, *r, *tofree, *name;
	char nambuf[100];
	Biobuf *bin, *bout;
	Type *t;
	Field *f;
	int orig_output_fd;

	quotefmtinstall();

	oargc = argc;
	oargv = argv;
	av = emalloc((30+argc)*sizeof av[0]);
	atexit(waitforgcc);

	n = 0;
	av[n++] = "gcc";
	av[n++] = "-fdollars-in-identifiers";
	av[n++] = "-S";	// write assembly
	av[n++] = "-gstabs";	// include stabs info
	av[n++] = "-o";	// to ...
	av[n++] = "-";	// ... stdout
	av[n++] = "-xc";	// read C

	ARGBEGIN{
	case 'g':
		lang = &go;
		pkg = EARGF(usage());
		break;
	case 'c':
		av[0] = EARGF(usage());
		break;
	case 'f':
		av[n++] = EARGF(usage());
		break;
	default:
		usage();
	}ARGEND

	if(argc == 0)
		av[n++] = "-";
	else
		av[n++] = argv[0];
	av[n] = nil;

	orig_output_fd = dup(1, -1);
	for(i=0; i==0 || i < argc; i++) {
		// Some versions of gcc do not accept -S with multiple files.
		// Run gcc once for each file.
		// Write assembly and stabs debugging to p[1].
		if(pipe(p) < 0)
			sysfatal("pipe: %r");
		dup(p[1], 1);
		close(p[1]);
		if (argc)
			av[n-1] = argv[i];
		pid = spawn(av[0], av);
		dup(orig_output_fd, 1);

		// Read assembly, pulling out .stabs lines.
		bin = Bfdopen(p[0], OREAD);
		while((q = Brdstr(bin, '\n', 1)) != nil) {
			//	.stabs	"float:t(0,12)=r(0,1);4;0;",128,0,0,0
			tofree = q;
			while(*q == ' ' || *q == '\t')
				q++;
			if(strncmp(q, ".stabs", 6) != 0)
				goto Continue;
			q += 6;
			while(*q == ' ' || *q == '\t')
				q++;
			if(*q++ != '\"') {
Bad:
				sysfatal("cannot parse .stabs line:\n%s", tofree);
			}

			r = strchr(q, '\"');
			if(r == nil)
				goto Bad;
			*r++ = '\0';
			if(*r++ != ',')
				goto Bad;
			if(*r < '0' || *r > '9')
				goto Bad;
			if(atoi(r) != 128)	// stabs kind = local symbol
				goto Continue;

			parsestabtype(q);

Continue:
			free(tofree);
		}
		Bterm(bin);
		waitfor(pid);
	}
	close(orig_output_fd);

	// Write defs to standard output.
	bout = Bfdopen(1, OWRITE);
	fmtinstall('T', lang->typefmt);

	// Echo original command line in header.
	Bprint(bout, "//");
	for(i=0; i<oargc; i++)
		Bprint(bout, " %q", oargv[i]);
	Bprint(bout, "\n");
	Bprint(bout, "\n");
	Bprint(bout, "// MACHINE GENERATED - DO NOT EDIT.\n");
	Bprint(bout, "\n");

	if(pkg)
		Bprint(bout, "package %s\n\n", pkg);

	// Constants.
	Bprint(bout, "// Constants\n");
	if(ncon > 0) {
		Bprint(bout, lang->constbegin);
		for(i=0; i<ncon; i++)
			Bprint(bout, lang->constfmt, con[i].name, con[i].value & 0xFFFFFFFF);
		Bprint(bout, lang->constend);
	}
	Bprint(bout, "\n");

	// Types

	// push our names down
	for(i=0; i<ntyp; i++) {
		t = typ[i];
		name = t->name;
		while(t && t->kind == Typedef)
			t = t->type;
		if(t)
			t->name = name;
	}

	Bprint(bout, "// Types\n");

	// Have to turn off structure padding in Plan 9 compiler,
	// mainly because it is more aggressive than gcc tends to be.
	if(lang == &c)
		Bprint(bout, "#pragma pack on\n");

	for(i=0; i<ntyp; i++) {
		Bprint(bout, "\n");
		t = typ[i];
		name = t->name;
		while(t && t->kind == Typedef) {
			if(name == nil && t->name != nil) {
				name = t->name;
				if(t->printed)
					break;
			}
			t = t->type;
		}
		if(name == nil && t->name != nil) {
			name = t->name;
			if(t->printed)
				continue;
			t->printed = 1;
		}
		if(name == nil) {
			fprint(2, "unknown name for %T", typ[i]);
			continue;
		}
		if(name[0] == '$')
			name++;
		npad = 0;
		off = 0;
		switch(t->kind) {
		case 0:
			fprint(2, "unknown type definition for %s\n", name);
			break;
		default:	// numeric, array, or pointer
		case Array:
		case Ptr:
			Bprint(bout, "%s %lT%s", lang->typdef, name, t, lang->typdefend);
			break;
		case Union:
			// In Go, print union as struct with only first element,
			// padded the rest of the way.
			Bprint(bout, lang->unionbegin, name, name, name);
			goto StructBody;
		case Struct:
			Bprint(bout, lang->structbegin, name, name, name);
		StructBody:
			prefix = 0;
			if(lang == &go)
				prefix = prefixlen(t);
			for(j=0; j<t->nf; j++) {
				f = &t->f[j];
				// padding
				if(t->kind == Struct || lang == &go) {
					if(f->offset%8 != 0 || f->size%8 != 0) {
						fprint(2, "ignoring bitfield %s.%s\n", t->name, f->name);
						continue;
					}
					if(f->offset < off)
						sysfatal("%s: struct fields went backward", t->name);
					if(off < f->offset) {
						Bprint(bout, lang->structpadfmt, npad++, (f->offset - off) / 8);
						off = f->offset;
					}
					off += f->size;
				}
				name = f->name;
				if(cutprefix(name))
					name += prefix;
				if(strcmp(name, "") == 0) {
					snprint(nambuf, sizeof nambuf, "Pad%d", npad++);
					name = nambuf;
				}
				Bprint(bout, "\t%#lT;\n", name, f->type);
				if(t->kind == Union && lang == &go)
					break;
			}
			// final padding
			if(t->kind == Struct || lang == &go) {
				if(off/8 < t->size)
					Bprint(bout, lang->structpadfmt, npad++, t->size - off/8);
			}
			Bprint(bout, lang->structend);
		}
	}
	if(lang == &c)
		Bprint(bout, "#pragma pack off\n");
	Bterm(bout);
	exit(0);
}

char *kindnames[] = {
	"void",	// actually unknown, but byte is good for pointers
	"void",
	"int8",
	"uint8",
	"int16",
	"uint16",
	"int32",
	"uint32",
	"int64",
	"uint64",
	"float32",
	"float64",
	"ptr",
	"struct",
	"array",
	"union",
	"typedef",
};

int
ctypefmt(Fmt *f)
{
	char *name, *s;
	Type *t;

	name = nil;
	if(f->flags & FmtLong) {
		name = va_arg(f->args, char*);
		if(name == nil || name[0] == '\0')
			name = "_anon_";
	}
	t = va_arg(f->args, Type*);
	while(t && t->kind == Typedef)
		t = t->type;
	switch(t->kind) {
	case Struct:
	case Union:
		// must be named
		s = t->name;
		if(s == nil) {
			fprint(2, "need name for anonymous struct\n");
			goto bad;
		}
		else if(s[0] != '$')
			fprint(2, "need name for struct %s\n", s);
		else
			s++;
		fmtprint(f, "%s", s);
		if(name)
			fmtprint(f, " %s", name);
		break;

	case Array:
		if(name)
			fmtprint(f, "%T %s[%d]", t->type, name, t->size);
		else
			fmtprint(f, "%T[%d]", t->type, t->size);
		break;

	case Ptr:
		if(name)
			fmtprint(f, "%T *%s", t->type, name);
		else
			fmtprint(f, "%T*", t->type);
		break;

	default:
		fmtprint(f, "%s", kindnames[t->kind]);
		if(name)
			fmtprint(f, " %s", name);
		break;

	bad:
		if(name)
			fmtprint(f, "byte %s[%d]", name, t->size);
		else
			fmtprint(f, "byte[%d]", t->size);
		break;
	}

	return 0;
}

int
gotypefmt(Fmt *f)
{
	char *name, *s;
	Type *t;

	if(f->flags & FmtLong) {
		name = va_arg(f->args, char*);
		if('a' <= name[0] && name[0] <= 'z')
			name[0] += 'A' - 'a';
		if(name[0] == '_' && (f->flags & FmtSharp))
			fmtprint(f, "X");
		fmtprint(f, "%s ", name);
	}
	t = va_arg(f->args, Type*);
	while(t && t->kind == Typedef)
		t = t->type;

	switch(t->kind) {
	case Struct:
	case Union:
		// must be named
		s = t->name;
		if(s == nil) {
			fprint(2, "need name for anonymous struct\n");
			fmtprint(f, "STRUCT");
		}
		else if(s[0] != '$') {
			fprint(2, "warning: missing name for struct %s\n", s);
			fmtprint(f, "[%d]byte /* %s */", t->size, s);
		} else
			fmtprint(f, "%s", s+1);
		break;

	case Array:
		fmtprint(f, "[%d]%T", t->size, t->type);
		break;

	case Ptr:
		fmtprint(f, "*%T", t->type);
		break;

	default:
		s = kindnames[t->kind];
		if(strcmp(s, "void") == 0)
			s = "byte";
		fmtprint(f, "%s", s);
	}

	return 0;
}

// Is this the kind of name we should cut a prefix from?
// The rule is that the name cannot begin with underscore
// and must have an underscore eventually.
int
cutprefix(char *name)
{
	char *p;

	// special case: orig_ in register struct
	if(strncmp(name, "orig_", 5) == 0)
		return 0;

	for(p=name; *p; p++) {
		if(*p == '_')
			return p-name > 0;
	}
	return 0;
}

// Figure out common struct prefix len
int
prefixlen(Type *t)
{
	int i;
	int len;
	char *p, *name;
	Field *f;

	len = 0;
	name = nil;
	for(i=0; i<t->nf; i++) {
		f = &t->f[i];
		if(!cutprefix(f->name))
			continue;
		p = strchr(f->name, '_');
		if(p == nil)
			return 0;
		if(name == nil) {
			name = f->name;
			len = p+1 - name;
		}
		else if(strncmp(f->name, name, len) != 0)
			return 0;
	}
	return len;
}
