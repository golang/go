// Inferno utils/nm/nm.c
// http://code.google.com/p/inferno-os/source/browse/utils/nm/nm.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

/*
 * nm.c -- drive nm
 */
#include <u.h>
#include <libc.h>
#include <ar.h>
#include <bio.h>
#include <mach.h>

enum{
	CHUNK	=	256	/* must be power of 2 */
};

char	*errs;			/* exit status */
char	*filename;		/* current file */
char	symname[]="__.GOSYMDEF";	/* table of contents file name */
int	multifile;		/* processing multiple files */
int	aflag;
int	gflag;
int	hflag;
int	nflag;
int	sflag;
int	Sflag;
int	uflag;
int	Tflag;
int	tflag;

Sym	**fnames;		/* file path translation table */
Sym	**symptr;
int	nsym;
Biobuf	bout;

int	cmp(void*, void*);
void	error(char*, ...);
void	execsyms(int);
void	psym(Sym*, void*);
void	printsyms(Sym**, long);
void	doar(Biobuf*);
void	dofile(Biobuf*);
void	zenter(Sym*);

void
usage(void)
{
	fprint(2, "usage: nm [-aghnsTu] file ...\n");
	exits("usage");
}

void
main(int argc, char *argv[])
{
	int i;
	Biobuf	*bin;

	Binit(&bout, 1, OWRITE);
	argv0 = argv[0];
	ARGBEGIN {
	default:	usage();
	case 'a':	aflag = 1; break;
	case 'g':	gflag = 1; break;
	case 'h':	hflag = 1; break;
	case 'n':	nflag = 1; break;
	case 's':	sflag = 1; break;
	case 'S':	nflag = Sflag = 1; break;
	case 'u':	uflag = 1; break;
	case 't':	tflag = 1; break;
	case 'T':	Tflag = 1; break;
	} ARGEND
	if (argc == 0)
		usage();
	if (argc > 1)
		multifile++;
	for(i=0; i<argc; i++){
		filename = argv[i];
		bin = Bopen(filename, OREAD);
		if(bin == 0){
			error("cannot open %s", filename);
			continue;
		}
		if (isar(bin))
			doar(bin);
		else{
			Bseek(bin, 0, 0);
			dofile(bin);
		}
		Bterm(bin);
	}
	exits(errs);
}

/*
 * read an archive file,
 * processing the symbols for each intermediate file in it.
 */
void
doar(Biobuf *bp)
{
	int offset, size, obj;
	char name[SARNAME];

	multifile = 1;
	for (offset = Boffset(bp);;offset += size) {
		size = nextar(bp, offset, name);
		if (size < 0) {
			error("phase error on ar header %d", offset);
			return;
		}
		if (size == 0)
			return;
		if (strcmp(name, symname) == 0)
			continue;
		obj = objtype(bp, 0);
		if (obj < 0) {
			// perhaps foreign object
			if(strlen(name) > 2 && strcmp(name+strlen(name)-2, ".o") == 0)
				return;
			error("inconsistent file %s in %s",
					name, filename);
			return;
		}
		if (!readar(bp, obj, offset+size, 1)) {
			error("invalid symbol reference in file %s",
					name);
			return;
		}
		filename = name;
		nsym=0;
		objtraverse(psym, 0);
		printsyms(symptr, nsym);
	}
}

/*
 * process symbols in a file
 */
void
dofile(Biobuf *bp)
{
	int obj;

	obj = objtype(bp, 0);
	if (obj < 0)
		execsyms(Bfildes(bp));
	else
	if (readobj(bp, obj)) {
		nsym = 0;
		objtraverse(psym, 0);
		printsyms(symptr, nsym);
	}
}

/*
 * comparison routine for sorting the symbol table
 *	this screws up on 'z' records when aflag == 1
 */
int
cmp(void *vs, void *vt)
{
	Sym **s, **t;

	s = vs;
	t = vt;
	if(nflag)	// sort on address (numeric) order
		if((*s)->value < (*t)->value)
			return -1;
		else
			return (*s)->value > (*t)->value;
	if(sflag)	// sort on file order (sequence)
		return (*s)->sequence - (*t)->sequence;
	return strcmp((*s)->name, (*t)->name);
}
/*
 * enter a symbol in the table of filename elements
 */
void
zenter(Sym *s)
{
	static int maxf = 0;

	if (s->value > maxf) {
		maxf = (s->value+CHUNK-1) &~ (CHUNK-1);
		fnames = realloc(fnames, (maxf+1)*sizeof(*fnames));
		if(fnames == 0) {
			error("out of memory", argv0);
			exits("memory");
		}
	}
	fnames[s->value] = s;
}

/*
 * get the symbol table from an executable file, if it has one
 */
void
execsyms(int fd)
{
	Fhdr f;
	Sym *s;
	int32 n;

	seek(fd, 0, 0);
	if (crackhdr(fd, &f) == 0) {
		error("Can't read header for %s", filename);
		return;
	}
	if (syminit(fd, &f) < 0)
		return;
	s = symbase(&n);
	nsym = 0;
	while(n--)
		psym(s++, 0);

	printsyms(symptr, nsym);
}

void
psym(Sym *s, void* p)
{
	USED(p);
	switch(s->type) {
	case 'T':
	case 'L':
	case 'D':
	case 'B':
		if (uflag)
			return;
		if (!aflag && ((s->name[0] == '.' || s->name[0] == '$')))
			return;
		break;
	case 'b':
	case 'd':
	case 'l':
	case 't':
		if (uflag || gflag)
			return;
		if (!aflag && ((s->name[0] == '.' || s->name[0] == '$')))
			return;
		break;
	case 'U':
		if (gflag)
			return;
		break;
	case 'Z':
		if (!aflag)
			return;
		break;
	case 'm':
		if(!aflag || uflag || gflag)
			return;
		break;
	case 'f':	/* we only see a 'z' when the following is true*/
		if(!aflag || uflag || gflag)
			return;
		zenter(s);
		break;
	case 'a':
	case 'p':
	case 'z':
	default:
		if(!aflag || uflag || gflag)
			return;
		break;
	}
	symptr = realloc(symptr, (nsym+1)*sizeof(Sym*));
	if (symptr == 0) {
		error("out of memory");
		exits("memory");
	}
	symptr[nsym++] = s;
}

void
printsyms(Sym **symptr, long nsym)
{
	int i, j, wid;
	Sym *s;
	char *cp;
	char path[512];

	qsort(symptr, nsym, sizeof(*symptr), (void*)cmp);

	wid = 0;
	for (i=0; i<nsym; i++) {
		s = symptr[i];
		if (s->value && wid == 0)
			wid = 8;
		else if (s->value >= 0x100000000LL && wid == 8)
			wid = 16;
	}
	for (i=0; i<nsym; i++) {
		s = symptr[i];
		if (multifile && !hflag)
			Bprint(&bout, "%s:", filename);
		if (s->type == 'z') {
			fileelem(fnames, (uchar *) s->name, path, 512);
			cp = path;
		} else
			cp = s->name;
		if (Tflag)
			Bprint(&bout, "%8ux ", s->sig);
		if (s->value || s->type == 'a' || s->type == 'p')
			Bprint(&bout, "%*llux ", wid, s->value);
		else
			Bprint(&bout, "%*s ", wid, "");
		if(Sflag) {
			vlong siz;

			siz = 0;
			for(j=i+1; j<nsym; j++) {
				if(symptr[j]->type != 'a' && symptr[j]->type != 'p') {
					siz = symptr[j]->value - s->value;
					break;
				}
			}
			if(siz > 0)
				Bprint(&bout, "%*llud ", wid, siz);
		}
		Bprint(&bout, "%c %s", s->type, cp);
		if(tflag && s->gotype)
			Bprint(&bout, " %*llux", wid, s->gotype);
		Bprint(&bout, "\n");
	}
}

void
error(char *fmt, ...)
{
	Fmt f;
	char buf[128];
	va_list arg;

	fmtfdinit(&f, 2, buf, sizeof buf);
	fmtprint(&f, "%s: ", argv0);
	va_start(arg, fmt);
	fmtvprint(&f, fmt, arg);
	va_end(arg);
	fmtprint(&f, "\n");
	fmtfdflush(&f);
	errs = "errors";
}
