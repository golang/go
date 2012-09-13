// Inferno utils/iar/ar.c
// http://code.google.com/p/inferno-os/source/browse/utils/iar/ar.c
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
 * ar - portable (ascii) format version
 */

/* protect a couple of our names */
#define select your_select
#define rcmd your_rcmd

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>
#include "../../libmach/obj.h"
#include <ar.h>

#undef select
#undef rcmd

/*
 *	The algorithm uses up to 3 temp files.  The "pivot member" is the
 *	archive member specified by and a, b, or i option.  The temp files are
 *	astart - contains existing members up to and including the pivot member.
 *	amiddle - contains new files moved or inserted behind the pivot.
 *	aend - contains the existing members that follow the pivot member.
 *	When all members have been processed, function 'install' streams the
 * 	temp files, in order, back into the archive.
 */

typedef struct	Arsymref
{
	char	*name;
	char *file;
	int	type;
	int	len;
	vlong	offset;
	struct	Arsymref *next;
} Arsymref;

typedef struct	Armember	/* Temp file entry - one per archive member */
{
	struct Armember	*next;
	struct ar_hdr	hdr;
	long		size;
	long		date;
	void		*member;
} Armember;

typedef	struct Arfile		/* Temp file control block - one per tempfile */
{
	char	*fname;		/* paging file name */
	vlong	size;
	Armember *head;		/* head of member chain */
	Armember *tail;		/* tail of member chain */
	Arsymref *sym;		/* head of defined symbol chain */
} Arfile;

typedef struct Hashchain
{
	char	*name;
	char *file;
	struct Hashchain *next;
} Hashchain;

#define	NHASH	1024

/*
 *	macro to portably read/write archive header.
 *	'cmd' is read/write/Bread/Bwrite, etc.
 */
#define	HEADER_IO(cmd, f, h)	cmd(f, h.name, sizeof(h.name)) != sizeof(h.name)\
				|| cmd(f, h.date, sizeof(h.date)) != sizeof(h.date)\
				|| cmd(f, h.uid, sizeof(h.uid)) != sizeof(h.uid)\
				|| cmd(f, h.gid, sizeof(h.gid)) != sizeof(h.gid)\
				|| cmd(f, h.mode, sizeof(h.mode)) != sizeof(h.mode)\
				|| cmd(f, h.size, sizeof(h.size)) != sizeof(h.size)\
				|| cmd(f, h.fmag, sizeof(h.fmag)) != sizeof(h.fmag)

		/* constants and flags */
char	*man =		"mrxtdpq";
char	*opt =		"uvnbailogS";
char	artemp[] =	"/tmp/vXXXXX";
char	movtemp[] =	"/tmp/v1XXXXX";
char	tailtemp[] =	"/tmp/v2XXXXX";
char	symdef[] =	"__.GOSYMDEF";
char	pkgdef[] =	"__.PKGDEF";

int	aflag;				/* command line flags */
int	bflag;
int	cflag;
int	gflag;
int	oflag;
int	uflag;
int	vflag;
int	Pflag;	/* remove leading file prefix */
int	Sflag;	/* force mark Go package as safe */

int	errors;

Arfile *astart, *amiddle, *aend;	/* Temp file control block pointers */
int	allobj = 1;			/* set when all members are object files of the same type */
int	symdefsize;			/* size of symdef file */
char	*pkgstmt;		/* string "package foo" */
char	*objhdr;		/* string "go object darwin 386 release.2010-01-01 2345+" */
int	dupfound;			/* flag for duplicate symbol */
Hashchain	*hash[NHASH];		/* hash table of text symbols */

#define	ARNAMESIZE	sizeof(astart->tail->hdr.name)

char	poname[ARNAMESIZE+1];		/* name of pivot member */
char	*file;				/* current file or member being worked on */
Biobuf	bout;
Biobuf bar;
char	*prefix;
int	pkgdefsafe;		/* was __.PKGDEF marked safe? */

void	arcopy(Biobuf*, Arfile*, Armember*);
int	arcreate(char*);
void	arfree(Arfile*);
void	arinsert(Arfile*, Armember*);
void	*armalloc(int);
char *arstrdup(char*);
void	armove(Biobuf*, Arfile*, Armember*);
void	arread(Biobuf*, Armember*);
void	arstream(int, Arfile*);
int	arwrite(int, Armember*);
int	bamatch(char*, char*);
int	duplicate(char*, char**);
Armember *getdir(Biobuf*);
void	getpkgdef(char**, int*);
void	install(char*, Arfile*, Arfile*, Arfile*, int);
void	loadpkgdata(char*, int);
void	longt(Armember*);
int	match(int, char**);
void	mesg(int, char*);
Arfile	*newtempfile(char*);
Armember *newmember(void);
void	objsym(Sym*, void*);
int	openar(char*, int, int);
void	pmode(long);
void	rl(int);
void	scanobj(Biobuf*, Arfile*, long);
void	scanpkg(Biobuf*, long);
void	select(int*, long);
void	setcom(void(*)(char*, int, char**));
void	skip(Biobuf*, vlong);
void	checksafe(Biobuf*, vlong);
int	symcomp(void*, void*);
void	trim(char*, char*, int);
void	usage(void);
void	wrerr(void);
void	wrsym(Biobuf*, long, Arsymref*);
int	arread_cutprefix(Biobuf*, Armember*);

void	rcmd(char*, int, char**);		/* command processing */
void	dcmd(char*, int, char**);
void	xcmd(char*, int, char**);
void	tcmd(char*, int, char**);
void	pcmd(char*, int, char**);
void	mcmd(char*, int, char**);
void	qcmd(char*, int, char**);
void	(*comfun)(char*, int, char**);

void
main(int argc, char *argv[])
{
	char *cp;

	Binit(&bout, 1, OWRITE);
	if(argc < 3)
		usage();
	for (cp = argv[1]; *cp; cp++) {
		switch(*cp) {
		case 'a':	aflag = 1;	break;
		case 'b':	bflag = 1;	break;
		case 'c':	cflag = 1;	break;
		case 'd':	setcom(dcmd);	break;
		case 'g':	gflag = 1; break;
		case 'i':	bflag = 1;	break;
		case 'l':
				strcpy(artemp, "vXXXXX");
				strcpy(movtemp, "v1XXXXX");
				strcpy(tailtemp, "v2XXXXX");
				break;
		case 'm':	setcom(mcmd);	break;
		case 'o':	oflag = 1;	break;
		case 'p':	setcom(pcmd);	break;
		case 'q':	setcom(qcmd);	break;
		case 'r':	setcom(rcmd);	break;
		case 't':	setcom(tcmd);	break;
		case 'u':	uflag = 1;	break;
		case 'v':	vflag = 1;	break;
		case 'x':	setcom(xcmd);	break;
		case 'S':	Sflag = 1;  break;
		case 'P':	Pflag = 1;  break;
		default:
			fprint(2, "pack: bad option `%c'\n", *cp);
			exits("error");
		}
	}
	if (aflag && bflag) {
		fprint(2, "pack: only one of 'a' and 'b' can be specified\n");
		usage();
	}
	if(aflag || bflag) {
		trim(argv[2], poname, sizeof(poname));
		argv++;
		argc--;
		if(argc < 3)
			usage();
	}
	if(Pflag) {
		if(argc < 4) {
			fprint(2, "pack: P flag requires prefix argument\n");
			usage();
		}
		prefix = argv[2];
		argv++;
		argc--;
	}
	if(comfun == 0) {
		if(uflag == 0) {
			fprint(2, "pack: one of [%s] must be specified\n", man);
			usage();
		}
		setcom(rcmd);
	}
	cp = argv[2];
	argc -= 3;
	argv += 3;
	(*comfun)(cp, argc, argv);	/* do the command */
	if(errors && cflag)
		remove(cp);
	cp = 0;
	while (argc--) {
		if (*argv) {
			fprint(2, "pack: %s not found\n", *argv);
			cp = "error";
		}
		argv++;
	}
	if (errors)
		cp = "error";
	exits(cp);
}
/*
 *	select a command
 */
void
setcom(void (*fun)(char *, int, char**))
{

	if(comfun != 0) {
		fprint(2, "pack: only one of [%s] allowed\n", man);
		usage();
	}
	comfun = fun;
}
/*
 *	perform the 'r' and 'u' commands
 */
void
rcmd(char *arname, int count, char **files)
{
	int fd;
	int i;
	Arfile *ap;
	Armember *bp;
	Dir *d;
	Biobuf *bfile;

	fd = openar(arname, ORDWR, 1);
	if (fd >= 0) {
		Binit(&bar, fd, OREAD);
		Bseek(&bar,seek(fd,0,1), 1);
	}
	astart = newtempfile(artemp);
	ap = astart;
	aend = 0;
	for(i = 0; fd >= 0; i++) {
		bp = getdir(&bar);
		if (!bp)
			break;
		if (bamatch(file, poname)) {		/* check for pivot */
			aend = newtempfile(tailtemp);
			ap = aend;
		}
			/* pitch symdef file */
		if (i == 0 && strcmp(file, symdef) == 0) {
			skip(&bar, bp->size);
			continue;
		}
			/* pitch pkgdef file but remember whether it was marked safe */
		if (gflag && strcmp(file, pkgdef) == 0) {
			checksafe(&bar, bp->size);
			continue;
		}
		/*
		 * the plan 9 ar treats count == 0 as equivalent
		 * to listing all the archive's files on the command line:
		 * it will try to open every file name in the archive
		 * and copy that file into the archive if it exists.
		 * for go we disable that behavior, because we use
		 * r with no files to make changes to the archive itself,
		 * using the S or P flags.
		 */
		if (!match(count, files)) {
			scanobj(&bar, ap, bp->size);
			arcopy(&bar, ap, bp);
			continue;
		}
		bfile = Bopen(file, OREAD);
		if (!bfile) {
			if (count != 0) {
				fprint(2, "pack: cannot open %s\n", file);
				errors++;
			}
			scanobj(&bar, ap, bp->size);
			arcopy(&bar, ap, bp);
			continue;
		}
		d = dirfstat(Bfildes(bfile));
		if(d == nil)
			fprint(2, "pack: cannot stat %s: %r\n", file);
		if (uflag && (d==nil || d->mtime <= bp->date)) {
			scanobj(&bar, ap, bp->size);
			arcopy(&bar, ap, bp);
			Bterm(bfile);
			free(d);
			continue;
		}
		mesg('r', file);
		skip(&bar, bp->size);
		scanobj(bfile, ap, d->length);
		free(d);
		armove(bfile, ap, bp);
		Bterm(bfile);
	}
	if(fd >= 0)
		close(fd);
		/* copy in remaining files named on command line */
	for (i = 0; i < count; i++) {
		file = files[i];
		if(file == 0)
			continue;
		files[i] = 0;
		bfile = Bopen(file, OREAD);
		if (!bfile) {
			fprint(2, "pack: cannot open %s\n", file);
			errors++;
		} else {
			mesg('a', file);
			d = dirfstat(Bfildes(bfile));
			if (d == nil)
				fprint(2, "can't stat %s\n", file);
			else {
				scanobj(bfile, astart, d->length);
				armove(bfile, astart, newmember());
				free(d);
			}
			Bterm(bfile);
		}
	}
	if(fd < 0 && !cflag)
		install(arname, astart, 0, aend, 1);	/* issue 'creating' msg */
	else
		install(arname, astart, 0, aend, 0);
}

void
dcmd(char *arname, int count, char **files)
{
	Armember *bp;
	int fd, i;

	if (!count)
		return;
	fd = openar(arname, ORDWR, 0);
	Binit(&bar, fd, OREAD);
	Bseek(&bar,seek(fd,0,1), 1);
	astart = newtempfile(artemp);
	for (i = 0; bp = getdir(&bar); i++) {
		if(match(count, files)) {
			mesg('d', file);
			skip(&bar, bp->size);
			if (strcmp(file, symdef) == 0)
				allobj = 0;
		} else if (i == 0 && strcmp(file, symdef) == 0) {
			skip(&bar, bp->size);
		} else if (gflag && strcmp(file, pkgdef) == 0) {
			skip(&bar, bp->size);
		} else {
			scanobj(&bar, astart, bp->size);
			arcopy(&bar, astart, bp);
		}
	}
	close(fd);
	install(arname, astart, 0, 0, 0);
}

void
xcmd(char *arname, int count, char **files)
{
	int fd, f, mode, i;
	Armember *bp;
	Dir dx;

	fd = openar(arname, OREAD, 0);
	Binit(&bar, fd, OREAD);
	Bseek(&bar,seek(fd,0,1), 1);
	i = 0;
	while (bp = getdir(&bar)) {
		if(count == 0 || match(count, files)) {
			mode = strtoul(bp->hdr.mode, 0, 8) & 0777;
			f = create(file, OWRITE, mode);
			if(f < 0) {
				fprint(2, "pack: %s cannot create\n", file);
				skip(&bar, bp->size);
			} else {
				mesg('x', file);
				arcopy(&bar, 0, bp);
				if (write(f, bp->member, bp->size) < 0)
					wrerr();
				if(oflag && bp->date != 0) {
					nulldir(&dx);
					dx.atime = bp->date;
					dx.mtime = bp->date;
					if(dirwstat(file, &dx) < 0)
						perror(file);
				}
				free(bp->member);
				close(f);
			}
			free(bp);
			if (count && ++i >= count)
				break;
		} else {
			skip(&bar, bp->size);
			free(bp);
		}
	}
	close(fd);
}
void
pcmd(char *arname, int count, char **files)
{
	int fd;
	Armember *bp;

	fd = openar(arname, OREAD, 0);
	Binit(&bar, fd, OREAD);
	Bseek(&bar,seek(fd,0,1), 1);
	while(bp = getdir(&bar)) {
		if(count == 0 || match(count, files)) {
			if(vflag)
				print("\n<%s>\n\n", file);
			arcopy(&bar, 0, bp);
			if (write(1, bp->member, bp->size) < 0)
				wrerr();
		} else
			skip(&bar, bp->size);
		free(bp);
	}
	close(fd);
}
void
mcmd(char *arname, int count, char **files)
{
	int fd, i;
	Arfile *ap;
	Armember *bp;

	if (count == 0)
		return;
	fd = openar(arname, ORDWR, 0);
	Binit(&bar, fd, OREAD);
	Bseek(&bar,seek(fd,0,1), 1);
	astart = newtempfile(artemp);
	amiddle = newtempfile(movtemp);
	aend = 0;
	ap = astart;
	for (i = 0; bp = getdir(&bar); i++) {
		if (bamatch(file, poname)) {
			aend = newtempfile(tailtemp);
			ap = aend;
		}
		if(match(count, files)) {
			mesg('m', file);
			scanobj(&bar, amiddle, bp->size);
			arcopy(&bar, amiddle, bp);
		} else if (ap == astart && i == 0 && strcmp(file, symdef) == 0) {
			/*
			 * pitch the symdef file if it is at the beginning
			 * of the archive and we aren't inserting in front
			 * of it (ap == astart).
			 */
			skip(&bar, bp->size);
		} else if (ap == astart && gflag && strcmp(file, pkgdef) == 0) {
			/*
			 * pitch the pkgdef file if we aren't inserting in front
			 * of it (ap == astart).
			 */
			skip(&bar, bp->size);
		} else {
			scanobj(&bar, ap, bp->size);
			arcopy(&bar, ap, bp);
		}
	}
	close(fd);
	if (poname[0] && aend == 0)
		fprint(2, "pack: %s not found - files moved to end.\n", poname);
	install(arname, astart, amiddle, aend, 0);
}
void
tcmd(char *arname, int count, char **files)
{
	int fd;
	Armember *bp;
	char name[ARNAMESIZE+1];

	fd = openar(arname, OREAD, 0);
	Binit(&bar, fd, OREAD);
	Bseek(&bar,seek(fd,0,1), 1);
	while(bp = getdir(&bar)) {
		if(count == 0 || match(count, files)) {
			if(vflag)
				longt(bp);
			trim(file, name, ARNAMESIZE);
			Bprint(&bout, "%s\n", name);
		}
		skip(&bar, bp->size);
		free(bp);
	}
	close(fd);
}
void
qcmd(char *arname, int count, char **files)
{
	int fd, i;
	Armember *bp;
	Biobuf *bfile;

	if(aflag || bflag) {
		fprint(2, "pack: abi not allowed with q\n");
		exits("error");
	}
	fd = openar(arname, ORDWR, 1);
	if (fd < 0) {
		if(!cflag)
			fprint(2, "pack: creating %s\n", arname);
		fd = arcreate(arname);
	}
	Binit(&bar, fd, OREAD);
	Bseek(&bar,seek(fd,0,1), 1);
	/* leave note group behind when writing archive; i.e. sidestep interrupts */
	rfork(RFNOTEG);
	Bseek(&bar, 0, 2);
	bp = newmember();
	for(i=0; i<count && files[i]; i++) {
		file = files[i];
		files[i] = 0;
		bfile = Bopen(file, OREAD);
		if(!bfile) {
			fprint(2, "pack: cannot open %s\n", file);
			errors++;
		} else {
			mesg('q', file);
			armove(bfile, 0, bp);
			if (!arwrite(fd, bp))
				wrerr();
			free(bp->member);
			bp->member = 0;
			Bterm(bfile);
		}
	}
	free(bp);
	close(fd);
}

/*
 *	does the object header line p match the last one we saw?
 *	update *lastp if it gets more specific.
 */
int
matchhdr(char *p, char **lastp)
{
	int n;
	char *last;
	
	// no information?
	last = *lastp;
	if(last == nil) {
		*lastp = strdup(p);
		return 1;
	}

	// identical match?
	if(strcmp(last, p) == 0)
		return 1;

	// last has extra fields
	n = strlen(p);
	if(n < strlen(last) && last[n] == ' ')
		return 1;

	// p has extra fields - save in last
	n = strlen(last);
	if(n < strlen(p) && p[n] == ' ') {
		free(last);
		*lastp = strdup(p);
		return 1;
	}
	
	return 0;
}	

/*
 *	extract the symbol references from an object file
 */
void
scanobj(Biobuf *b, Arfile *ap, long size)
{
	int obj, goobject;
	vlong offset, offset1;
	Dir *d;
	static int lastobj = -1;
	uchar buf[4];
	char *p;

	if (!allobj)			/* non-object file encountered */
		return;
	offset = Boffset(b);
	obj = objtype(b, 0);
	if (obj < 0) {			/* not an object file */
		/* maybe a foreign object file */
		Bseek(b, offset, 0);
		memset(buf, 0, sizeof buf);
		Bread(b, buf, 4);
		
		/* maybe a foreign object file?  that's okay */
		if((buf[0] == 0x7F && buf[1] == 'E' && buf[2] == 'L' && buf[3] == 'F') ||   // ELF
		   (buf[0] == 0x4c && buf[1] == 0x01 || buf[0] == 0x64 && buf[1] == 0x86) || // Windows PE
		   (buf[0] == 0xFE && buf[1] == 0xED && buf[2] == 0xFA && (buf[3]&~1) == 0xCE) ||  // Mach-O big-endian
		   (buf[3] == 0xFE && buf[2] == 0xED && buf[1] == 0xFA && (buf[0]&~1) == 0xCE)) {  // Mach-O little-endian
			Bseek(b, offset, 0);
			return;
		}
		
		if (!gflag || strcmp(file, pkgdef) != 0) {  /* don't clear allobj if it's pkg defs */
			fprint(2, "pack: non-object file %s\n", file);
			errors++;
			allobj = 0;
		}
		d = dirfstat(Bfildes(b));
		if (d != nil && d->length == 0) {
			fprint(2, "pack: zero length file %s\n", file);
			errors++;
		}
		free(d);
		Bseek(b, offset, 0);
		return;
	}

	goobject = 1;
	offset1 = Boffset(b);
	Bseek(b, offset, 0);
	p = Brdstr(b, '\n', 1);
	
	// After the go object header comes the Go metadata,
	// followed by ! on a line by itself.  If this is not a Go object,
	// the ! comes immediately.  Catch that so we can avoid
	// the call to scanpkg below, since scanpkg assumes that the
	// Go metadata is present.
	if(Bgetc(b) == '!')
		goobject = 0;

	Bseek(b, offset1, 0);
	if(p == nil || strncmp(p, "go object ", 10) != 0) {
		fprint(2, "pack: malformed object file %s\n", file);
		errors++;
		Bseek(b, offset, 0);
		free(p);
		return;
	}
	
	if (!matchhdr(p, &objhdr)) {
		fprint(2, "pack: inconsistent object file %s: [%s] vs [%s]\n", file, p, objhdr);
		errors++;
		allobj = 0;
		free(p);
		return;
	}
	free(p);

	// Old check.  Should be impossible since objhdrs match, but keep the check anyway.
	if (lastobj >= 0 && obj != lastobj) {
		fprint(2, "pack: inconsistent object file %s\n", file);
		errors++;
		allobj = 0;
		return;
	}
	lastobj = obj;
		
	if (!readar(b, obj, offset+size, 0)) {
		fprint(2, "pack: invalid symbol reference in file %s\n", file);
		errors++;
		allobj = 0;
		Bseek(b, offset, 0);
		return;
	}
	Bseek(b, offset, 0);
	objtraverse(objsym, ap);
	if (gflag && goobject) {
		scanpkg(b, size);
		Bseek(b, offset, 0);
	}
}

/*
 *	does line contain substring (length-limited)
 */
int
strstrn(char *line, int len, char *sub)
{
	int i;
	int sublen;

	sublen = strlen(sub);
	for (i = 0; i < len - sublen; i++)
		if (memcmp(line+i, sub, sublen) == 0)
			return 1;
	return 0;
}

/*
 *	package import data
 */
int	safe = 1;
char*	pkgname;
char*	importblock;

void
getpkgdef(char **datap, int *lenp)
{
	char *tag, *hdr;

	if(pkgname == nil) {
		pkgname = "__emptyarchive__";
		importblock = "";
	}
	
	tag = "";
	if(safe || Sflag)
		tag = "safe";

	hdr = "empty archive";
	if(objhdr != nil)
		hdr = objhdr;

	*datap = smprint("%s\nimport\n$$\npackage %s %s\n%s\n$$\n", hdr, pkgname, tag, importblock);
	*lenp = strlen(*datap);
}

/*
 *	extract the package definition data from an object file.
 *	there can be only one.
 */
void
scanpkg(Biobuf *b, long size)
{
	long n;
	int c;
	long start, end, pkgsize;
	char *data, *line, pkgbuf[1024], *pkg;
	int first;

	/*
	 * scan until $$
	 */
	for (n=0; n<size; ) {
		c = Bgetc(b);
		if(c == Beof)
			break;
		n++;
		if(c != '$')
			continue;
		c = Bgetc(b);
		if(c == Beof)
			break;
		n++;
		if(c != '$')
			continue;
		goto foundstart;
	}
	// fprint(2, "pack: warning: no package import section in %s\n", file);
	if(b != &bar || !pkgdefsafe)
		safe = 0;	// non-Go file (C or assembly)
	return;

foundstart:
	/* found $$; skip rest of line */
	while((c = Bgetc(b)) != '\n')
		if(c == Beof)
			goto bad;

	/* how big is it? */
	first = 1;
	start = end = 0;
	for (n=0; n<size; n+=Blinelen(b)) {
		line = Brdstr(b, '\n', 0);
		if (line == nil)
			goto bad;
		if (first && strstrn(line, Blinelen(b), "package ")) {
			if (Blinelen(b) > sizeof(pkgbuf)-1)
				goto bad;
			memmove(pkgbuf, line, Blinelen(b));
			pkgbuf[Blinelen(b)] = '\0';
			pkg = pkgbuf;
			while(*pkg == ' ' || *pkg == '\t')
				pkg++;
			if(strncmp(pkg, "package ", 8) != 0)
				goto bad;
			pkg += 8;
			data = pkg;
			while(*pkg != ' ' && *pkg != '\n' && *pkg != '\0')
				pkg++;
			pkgname = armalloc(pkg - data + 1);
			memmove(pkgname, data, pkg - data);
			pkgname[pkg-data] = '\0';
			if(strcmp(pkg, " safe\n") != 0 && (b != &bar || !pkgdefsafe))
				safe = 0;
			start = Boffset(b);  // after package statement
			first = 0;
			free(line);
			continue;
		}
		if(line[0] == '$' && line[1] == '$') {
			free(line);
			goto foundend;
		}
		end = Boffset(b);  // before closing $$
		free(line);
	}
bad:
	fprint(2, "pack: bad package import section in %s\n", file);
	errors++;
	return;

foundend:
	if (start == 0)
		return;
	if (end == 0)
		goto bad;
	if(importblock != nil) {
		fprint(2, "pack: multiple Go object files\n");
		errors++;
		return;
	}
	pkgsize = end-start;
	data = armalloc(end - start + 1);
	Bseek(b, start, 0);
	if (Bread(b, data, pkgsize) != pkgsize) {
		fprint(2, "pack: error reading package import section in %s\n", file);
		errors++;
		return;
	}
	data[end-start] = '\0';
	importblock = data;
}

/*
 *	add text and data symbols to the symbol list
 */
void
objsym(Sym *s, void *p)
{
	int n;
	Arsymref *as;
	Arfile *ap;
	char *ofile;

	if (s->type != 'T' &&  s->type != 'D')
		return;
	ap = (Arfile*)p;
	as = armalloc(sizeof(Arsymref));
	as->offset = ap->size;
	as->name = arstrdup(s->name);
	as->file = arstrdup(file);
	if(s->type == 'T' && duplicate(as->name, &ofile)) {
		dupfound = 1;
		fprint(2, "duplicate text symbol: %s and %s: %s\n", as->file, ofile, as->name);
		errors++;
		free(as->name);
		free(as);
		return;
	}
	as->type = s->type;
	n = strlen(s->name);
	symdefsize += 4+(n+1)+1;
	as->len = n;
	as->next = ap->sym;
	ap->sym = as;
}

/*
 *	Check the symbol table for duplicate text symbols
 */
int
hashstr(char *name)
{
	int h;
	char *cp;

	h = 0;
	for(cp = name; *cp; h += *cp++)
		h *= 1119;
	
	// the code used to say
	//	if(h < 0)
	//		h = ~h;
	// but on gcc 4.3 with -O2 on some systems,
	// the if(h < 0) gets compiled away as not possible.
	// use a mask instead, leaving plenty of bits but
	// definitely not the sign bit.

	return h & 0xfffffff;
}

int
duplicate(char *name, char **ofile)
{
	Hashchain *p;
	int h;

	h = hashstr(name) % NHASH;

	for(p = hash[h]; p; p = p->next)
		if(strcmp(p->name, name) == 0) {
			*ofile = p->file;
			return 1;
		}
	p = armalloc(sizeof(Hashchain));
	p->next = hash[h];
	p->name = name;
	p->file = file;
	hash[h] = p;
	*ofile = nil;
	return 0;
}

/*
 *	open an archive and validate its header
 */
int
openar(char *arname, int mode, int errok)
{
	int fd;
	char mbuf[SARMAG];

	fd = open(arname, mode);
	if(fd >= 0){
		if(read(fd, mbuf, SARMAG) != SARMAG || strncmp(mbuf, ARMAG, SARMAG)) {
			fprint(2, "pack: %s not in archive format\n", arname);
			exits("error");
		}
	}else if(!errok){
		fprint(2, "pack: cannot open %s: %r\n", arname);
		exits("error");
	}
	return fd;
}

/*
 *	create an archive and set its header
 */
int
arcreate(char *arname)
{
	int fd;

	fd = create(arname, OWRITE, 0664);
	if(fd < 0){
		fprint(2, "pack: cannot create %s: %r\n", arname);
		exits("error");
	}
	if(write(fd, ARMAG, SARMAG) != SARMAG)
		wrerr();
	return fd;
}

/*
 *		error handling
 */
void
wrerr(void)
{
	perror("pack: write error");
	exits("error");
}

void
rderr(void)
{
	perror("pack: read error");
	exits("error");
}

void
phaseerr(int offset)
{
	fprint(2, "pack: phase error at offset %d\n", offset);
	exits("error");
}

void
usage(void)
{
	fprint(2, "usage: pack [%s][%s][P prefix] archive files ...\n", opt, man);
	exits("error");
}

/*
 *	read the header for the next archive member
 */
Armember *
getdir(Biobuf *b)
{
	Armember *bp;
	char *cp;
	static char name[ARNAMESIZE+1];

	bp = newmember();
	if(HEADER_IO(Bread, b, bp->hdr)) {
		free(bp);
		return 0;
	}
	if(strncmp(bp->hdr.fmag, ARFMAG, sizeof(bp->hdr.fmag)))
		phaseerr(Boffset(b));
	strncpy(name, bp->hdr.name, sizeof(bp->hdr.name));
	cp = name+sizeof(name)-1;
	while(*--cp==' ')
		;
	cp[1] = '\0';
	file = arstrdup(name);
	bp->date = strtol(bp->hdr.date, 0, 0);
	bp->size = strtol(bp->hdr.size, 0, 0);
	return bp;
}

/*
 *	Copy the file referenced by fd to the temp file
 */
void
armove(Biobuf *b, Arfile *ap, Armember *bp)
{
	char *cp;
	Dir *d;
	vlong n;

	d = dirfstat(Bfildes(b));
	if (d == nil) {
		fprint(2, "pack: cannot stat %s\n", file);
		return;
	}

	trim(file, bp->hdr.name, sizeof(bp->hdr.name));
	for (cp = strchr(bp->hdr.name, 0);		/* blank pad on right */
		cp < bp->hdr.name+sizeof(bp->hdr.name); cp++)
			*cp = ' ';
	sprint(bp->hdr.date, "%-12ld", 0L);  // was d->mtime but removed for idempotent builds
	sprint(bp->hdr.uid, "%-6d", 0);
	sprint(bp->hdr.gid, "%-6d", 0);
	sprint(bp->hdr.mode, "%-8lo", d->mode);
	sprint(bp->hdr.size, "%-10lld", d->length);
	strncpy(bp->hdr.fmag, ARFMAG, 2);
	bp->size = d->length;
	arread(b, bp);
	n = bp->size;
	if (n&1)
		n++;
	if (ap) {
		arinsert(ap, bp);
		ap->size += n+SAR_HDR;
	}
	free(d);
}

/*
 *	Copy the archive member at the current offset into the temp file.
 */
void
arcopy(Biobuf *b, Arfile *ap, Armember *bp)
{
	long n;

	arread(b, bp);
	n = bp->size;
	if (n & 01)
		n++;
	if (ap) {
		arinsert(ap, bp);
		ap->size += n+SAR_HDR;
	}
}

/*
 *	Skip an archive member
 */
void
skip(Biobuf *bp, vlong len)
{
	if (len & 01)
		len++;
	Bseek(bp, len, 1);
}

void
checksafe(Biobuf *bp, vlong len)
{
	char *p;
	vlong end;

	if (len & 01)
		len++;
	end = Boffset(bp) + len;

	p = Brdline(bp, '\n');
	if(p == nil || strncmp(p, "go object ", 10) != 0)
		goto done;
	for(;;) {
		p = Brdline(bp, '\n');
		if(p == nil || Boffset(bp) >= end)
			goto done;
		if(strncmp(p, "$$\n", 3) == 0)
			break;
	}
	p = Brdline(bp, '\n');
	if(p == nil || Boffset(bp) > end)
		goto done;
	if(Blinelen(bp) > 8+6 && strncmp(p, "package ", 8) == 0 && strncmp(p+Blinelen(bp)-6, " safe\n", 6) == 0)
		pkgdefsafe = 1;

done:
	Bseek(bp, end, 0);
}

/*
 *	Stream the three temp files to an archive
 */
void
install(char *arname, Arfile *astart, Arfile *amiddle, Arfile *aend, int createflag)
{
	int fd;

	if(allobj && dupfound) {
		fprint(2, "%s not changed\n", arname);
		return;
	}
	/* leave note group behind when copying back; i.e. sidestep interrupts */
	rfork(RFNOTEG);

	if(createflag)
		fprint(2, "pack: creating %s\n", arname);
	fd = arcreate(arname);

	if(allobj)
		rl(fd);

	if (astart) {
		arstream(fd, astart);
		arfree(astart);
	}
	if (amiddle) {
		arstream(fd, amiddle);
		arfree(amiddle);
	}
	if (aend) {
		arstream(fd, aend);
		arfree(aend);
	}
	close(fd);
}

void
rl(int fd)
{
	Biobuf b;
	char *cp;
	struct ar_hdr a;
	long len;
	int headlen;
	char *pkgdefdata;
	int pkgdefsize;

	pkgdefdata = nil;
	pkgdefsize = 0;

	Binit(&b, fd, OWRITE);
	Bseek(&b,seek(fd,0,1), 0);

	len = symdefsize;
	if(len&01)
		len++;
	sprint(a.date, "%-12ld", 0L);  // time(0)
	sprint(a.uid, "%-6d", 0);
	sprint(a.gid, "%-6d", 0);
	sprint(a.mode, "%-8lo", 0644L);
	sprint(a.size, "%-10ld", len);
	strncpy(a.fmag, ARFMAG, 2);
	strcpy(a.name, symdef);
	for (cp = strchr(a.name, 0);		/* blank pad on right */
		cp < a.name+sizeof(a.name); cp++)
			*cp = ' ';
	if(HEADER_IO(Bwrite, &b, a))
			wrerr();

	headlen = Boffset(&b);
	len += headlen;
	if (gflag) {
		getpkgdef(&pkgdefdata, &pkgdefsize);
		len += SAR_HDR + pkgdefsize;
		if (len & 1)
			len++;
	}
	if (astart) {
		wrsym(&b, len, astart->sym);
		len += astart->size;
	}
	if(amiddle) {
		wrsym(&b, len, amiddle->sym);
		len += amiddle->size;
	}
	if(aend)
		wrsym(&b, len, aend->sym);

	if(symdefsize&0x01)
		Bputc(&b, 0);

	if (gflag) {
		len = pkgdefsize;
		sprint(a.date, "%-12ld", 0L);  // time(0)
		sprint(a.uid, "%-6d", 0);
		sprint(a.gid, "%-6d", 0);
		sprint(a.mode, "%-8lo", 0644L);
		sprint(a.size, "%-10ld", (len + 1) & ~1);
		strncpy(a.fmag, ARFMAG, 2);
		strcpy(a.name, pkgdef);
		for (cp = strchr(a.name, 0);		/* blank pad on right */
			cp < a.name+sizeof(a.name); cp++)
				*cp = ' ';
		if(HEADER_IO(Bwrite, &b, a))
				wrerr();

		if (Bwrite(&b, pkgdefdata, pkgdefsize) != pkgdefsize)
			wrerr();
		if(len&0x01)
			Bputc(&b, 0);
	}
	Bterm(&b);
}

/*
 *	Write the defined symbols to the symdef file
 */
void
wrsym(Biobuf *bp, long offset, Arsymref *as)
{
	int off;

	while(as) {
		Bputc(bp, as->type);
		off = as->offset+offset;
		Bputc(bp, off);
		Bputc(bp, off>>8);
		Bputc(bp, off>>16);
		Bputc(bp, off>>24);
		if (Bwrite(bp, as->name, as->len+1) != as->len+1)
			wrerr();
		as = as->next;
	}
}

/*
 *	Check if the archive member matches an entry on the command line.
 */
int
match(int count, char **files)
{
	int i;
	char name[ARNAMESIZE+1];

	for(i=0; i<count; i++) {
		if(files[i] == 0)
			continue;
		trim(files[i], name, ARNAMESIZE);
		if(strncmp(name, file, ARNAMESIZE) == 0) {
			file = files[i];
			files[i] = 0;
			return 1;
		}
	}
	return 0;
}

/*
 *	compare the current member to the name of the pivot member
 */
int
bamatch(char *file, char *pivot)
{
	static int state = 0;

	switch(state)
	{
	case 0:			/* looking for position file */
		if (aflag) {
			if (strncmp(file, pivot, ARNAMESIZE) == 0)
				state = 1;
		} else if (bflag) {
			if (strncmp(file, pivot, ARNAMESIZE) == 0) {
				state = 2;	/* found */
				return 1;
			}
		}
		break;
	case 1:			/* found - after previous file */
		state = 2;
		return 1;
	case 2:			/* already found position file */
		break;
	}
	return 0;
}

/*
 *	output a message, if 'v' option was specified
 */
void
mesg(int c, char *file)
{

	if(vflag)
		Bprint(&bout, "%c - %s\n", c, file);
}

/*
 *	isolate file name by stripping leading directories and trailing slashes
 */
void
trim(char *s, char *buf, int n)
{
	char *p, *q;

	for(;;) {
		p = strrchr(s, '/');
		q = strrchr(s, '\\');
		if (q > p)
			p = q;
		if (!p) {		/* no (back)slash in name */
			strncpy(buf, s, n);
			return;
		}
		if (p[1] != 0) {	/* p+1 is first char of file name */
			strncpy(buf, p+1, n);
			return;
		}
		*p = 0;			/* strip trailing (back)slash */
	}
}

/*
 *	utilities for printing long form of 't' command
 */
#define	SUID	04000
#define	SGID	02000
#define	ROWN	0400
#define	WOWN	0200
#define	XOWN	0100
#define	RGRP	040
#define	WGRP	020
#define	XGRP	010
#define	ROTH	04
#define	WOTH	02
#define	XOTH	01
#define	STXT	01000

void
longt(Armember *bp)
{
	char *cp;

	pmode(strtoul(bp->hdr.mode, 0, 8));
	Bprint(&bout, "%3ld/%1ld", strtol(bp->hdr.uid, 0, 0), strtol(bp->hdr.gid, 0, 0));
	Bprint(&bout, "%7ld", bp->size);
	cp = ctime(bp->date);
	Bprint(&bout, " %-12.12s %-4.4s ", cp+4, cp+24);
}

int	m1[] = { 1, ROWN, 'r', '-' };
int	m2[] = { 1, WOWN, 'w', '-' };
int	m3[] = { 2, SUID, 's', XOWN, 'x', '-' };
int	m4[] = { 1, RGRP, 'r', '-' };
int	m5[] = { 1, WGRP, 'w', '-' };
int	m6[] = { 2, SGID, 's', XGRP, 'x', '-' };
int	m7[] = { 1, ROTH, 'r', '-' };
int	m8[] = { 1, WOTH, 'w', '-' };
int	m9[] = { 2, STXT, 't', XOTH, 'x', '-' };

int	*m[] = { m1, m2, m3, m4, m5, m6, m7, m8, m9};

void
pmode(long mode)
{
	int **mp;

	for(mp = &m[0]; mp < &m[9];)
		select(*mp++, mode);
}

void
select(int *ap, long mode)
{
	int n;

	n = *ap++;
	while(--n>=0 && (mode&*ap++)==0)
		ap++;
	Bputc(&bout, *ap);
}

/*
 *	Temp file I/O subsystem.  We attempt to cache all three temp files in
 *	core.  When we run out of memory we spill to disk.
 *	The I/O model assumes that temp files:
 *		1) are only written on the end
 *		2) are only read from the beginning
 *		3) are only read after all writing is complete.
 *	The architecture uses one control block per temp file.  Each control
 *	block anchors a chain of buffers, each containing an archive member.
 */
Arfile *
newtempfile(char *name)		/* allocate a file control block */
{
	Arfile *ap;

	ap = armalloc(sizeof(Arfile));
	ap->fname = name;
	return ap;
}

Armember *
newmember(void)			/* allocate a member buffer */
{
	return armalloc(sizeof(Armember));
}

void
arread(Biobuf *b, Armember *bp)	/* read an image into a member buffer */
{
	int i;
	vlong off;

	bp->member = armalloc(bp->size);
	
	// If P flag is set, let arread_cutprefix try.
	// If it succeeds, we're done.  If not, fall back
	// to a direct copy.
	off = Boffset(b);
	if(Pflag && arread_cutprefix(b, bp))
		return;
	Bseek(b, off, 0);

	i = Bread(b, bp->member, bp->size);
	if (i < 0) {
		free(bp->member);
		bp->member = 0;
		rderr();
	}
	if(bp->size&1)
		Bgetc(b);
}

/*
 * insert a member buffer into the member chain
 */
void
arinsert(Arfile *ap, Armember *bp)
{
	bp->next = 0;
	if (!ap->tail)
		ap->head = bp;
	else
		ap->tail->next = bp;
	ap->tail = bp;
}

/*
 *	stream the members in a temp file to the file referenced by 'fd'.
 */
void
arstream(int fd, Arfile *ap)
{
	Armember *bp;

	/* dump the in-core buffers */
	for (bp = ap->head; bp; bp = bp->next) {
		if (!arwrite(fd, bp))
			wrerr();
	}
}

/*
 *	write a member to 'fd'.
 */
int
arwrite(int fd, Armember *bp)
{
	int len;

	if(HEADER_IO(write, fd, bp->hdr))
		return 0;
	len = bp->size;
	if (len & 01)
		len++;
	if (write(fd, bp->member, len) != len)
		return 0;
	return 1;
}

void
arfree(Arfile *ap)		/* free a member buffer */
{
	Armember *bp, *next;

	for (bp = ap->head; bp; bp = next) {
		next = bp->next;
		if (bp->member)
			free(bp->member);
		free(bp);
	}
	free(ap);
}

/*
 *	allocate space for a control block or member buffer.  if the malloc
 *	fails we try to reclaim space by spilling previously allocated
 *	member buffers.
 */
void *
armalloc(int n)
{
	char *cp;

	// bump so that arwrite can do the same
	if(n&1)
		n++;

	cp = malloc(n);
	if (cp) {
		memset(cp, 0, n);
		return cp;
	}
	fprint(2, "pack: out of memory\n");
	exits("malloc");
	return 0;
}

char *
arstrdup(char *s)
{
	char *t;

	t = armalloc(strlen(s) + 1);
	strcpy(t, s);
	return t;
}


/*
 *	Parts of libmach we're not supposed
 *	to look at but need for arread_cutprefix.
 */
extern int _read5(Biobuf*, Prog*);
extern int _read6(Biobuf*, Prog*);
extern int _read8(Biobuf*, Prog*);
int (*reader[256])(Biobuf*, Prog*) = {
	[ObjArm] = _read5,
	[ObjAmd64] = _read6,
	[Obj386] = _read8,
};

/*
 *	copy b into bp->member but rewrite object
 *	during copy to drop prefix from all file names.
 *	return 1 if b was recognized as an object file
 *	and copied successfully, 0 otherwise.
 */
int
arread_cutprefix(Biobuf *b, Armember *bp)
{
	vlong offset, o, end;
	int n, t;
	int (*rd)(Biobuf*, Prog*);
	char *w, *inprefix;
	Prog p;
	
	offset = Boffset(b);
	end = offset + bp->size;
	t = objtype(b, nil);
	if(t < 0)
		return 0;
	if((rd = reader[t]) == nil)
		return 0;
	
	// copy header
	w = bp->member;
	n = Boffset(b) - offset;
	Bseek(b, -n, 1);
	if(Bread(b, w, n) != n)
		return 0;
	offset += n;
	w += n;
	
	// read object file one pseudo-instruction at a time,
	// eliding the file name instructions that refer to
	// the prefix.
	memset(&p, 0, sizeof p);
	inprefix = nil;
	while(Boffset(b) < end && rd(b, &p)) {
		if(p.kind == aName && p.type == UNKNOWN && p.sym == 1 && p.id[0] == '<') {
			// part of a file path.
			// we'll keep continuing (skipping the copy)
			// around the loop until either we get to a
			// name piece that should be kept or we see
			// the whole prefix.

			if(inprefix == nil && prefix[0] == '/' && p.id[1] == '/' && p.id[2] == '\0') {
				// leading /
				inprefix = prefix+1;
			} else if(inprefix != nil) {
				// handle subsequent elements
				n = strlen(p.id+1);
				if(strncmp(p.id+1, inprefix, n) == 0 && (inprefix[n] == '/' || inprefix[n] == '\0')) {
					inprefix += n;
					if(inprefix[0] == '/')
						inprefix++;
				}
			}
			
			if(inprefix && inprefix[0] == '\0') {
				// reached end of prefix.
				// if we another path element follows,
				// nudge the offset to skip over the prefix we saw.
				// if not, leave offset alone, to emit the whole name.
				// additional name elements will not be skipped
				// because inprefix is now nil and we won't see another
				// leading / in this name.
				inprefix = nil;
				o = Boffset(b);
				if(o < end && rd(b, &p) && p.kind == aName && p.type == UNKNOWN && p.sym == 1 && p.id[0] == '<') {
					// print("skip %lld-%lld\n", offset, o);
					offset = o;
				}
			}
		} else {
			// didn't find the whole prefix.
			// give up and let it emit the entire name.
			inprefix = nil;
		}

		// copy instructions
		if(!inprefix) {
			n = Boffset(b) - offset;
			Bseek(b, -n, 1);
			if(Bread(b, w, n) != n)
				return 0;
			offset += n;
			w += n;
		}
	}
	bp->size = w - (char*)bp->member;
	sprint(bp->hdr.size, "%-10lld", (vlong)bp->size);
	strncpy(bp->hdr.fmag, ARFMAG, 2);
	Bseek(b, end, 0);
	if(Boffset(b)&1)
		Bgetc(b);
	return 1;
}
