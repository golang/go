// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing and reading of Go object files.
//
// Originally, Go object files were Plan 9 object files, but no longer.
// Now they are more like standard object files, in that each symbol is defined
// by an associated memory image (bytes) and a list of relocations to apply
// during linking. We do not (yet?) use a standard file format, however.
// For now, the format is chosen to be as simple as possible to read and write.
// It may change for reasons of efficiency, or we may even switch to a
// standard file format if there are compelling benefits to doing so.
// See golang.org/s/go13linker for more background.
//
// The file format is:
//
//	- magic header: "\x00\x00go13ld"
//	- byte 1 - version number
//	- sequence of strings giving dependencies (imported packages)
//	- empty string (marks end of sequence)
//	- sequence of defined symbols
//	- byte 0xff (marks end of sequence)
//	- magic footer: "\xff\xffgo13ld"
//
// All integers are stored in a zigzag varint format.
// See golang.org/s/go12symtab for a definition.
//
// Data blocks and strings are both stored as an integer
// followed by that many bytes.
//
// A symbol reference is a string name followed by a version.
// An empty name corresponds to a nil LSym* pointer.
//
// Each symbol is laid out as the following fields (taken from LSym*):
//
//	- byte 0xfe (sanity check for synchronization)
//	- type [int]
//	- name [string]
//	- version [int]
//	- flags [int]
//		1 dupok
//	- size [int]
//	- gotype [symbol reference]
//	- p [data block]
//	- nr [int]
//	- r [nr relocations, sorted by off]
//
// If type == STEXT, there are a few more fields:
//
//	- args [int]
//	- locals [int]
//	- nosplit [int]
//	- flags [int]
//		1 leaf
//		2 C function
//	- nlocal [int]
//	- local [nlocal automatics]
//	- pcln [pcln table]
//
// Each relocation has the encoding:
//
//	- off [int]
//	- siz [int]
//	- type [int]
//	- add [int]
//	- xadd [int]
//	- sym [symbol reference]
//	- xsym [symbol reference]
//
// Each local has the encoding:
//
//	- asym [symbol reference]
//	- offset [int]
//	- type [int]
//	- gotype [symbol reference]
//
// The pcln table has the encoding:
//
//	- pcsp [data block]
//	- pcfile [data block]
//	- pcline [data block]
//	- npcdata [int]
//	- pcdata [npcdata data blocks]
//	- nfuncdata [int]
//	- funcdata [nfuncdata symbol references]
//	- funcdatasym [nfuncdata ints]
//	- nfile [int]
//	- file [nfile symbol references]
//
// The file layout and meaning of type integers are architecture-independent.
//
// TODO(rsc): The file format is good for a first pass but needs work.
//	- There are SymID in the object file that should really just be strings.
//	- The actual symbol memory images are interlaced with the symbol
//	  metadata. They should be separated, to reduce the I/O required to
//	  load just the metadata.
//	- The symbol references should be shortened, either with a symbol
//	  table or by using a simple backward index to an earlier mentioned symbol.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>
#include "../cmd/ld/textflag.h"
#include "../runtime/funcdata.h"

static void writesym(Link*, Biobuf*, LSym*);
static void wrint(Biobuf*, int64);
static void wrstring(Biobuf*, char*);
static void wrpath(Link *, Biobuf*, char*);
static void wrdata(Biobuf*, void*, int);
static void wrsym(Biobuf*, LSym*);
static void wrpathsym(Link *ctxt, Biobuf *b, LSym *s);

static void readsym(Link*, Biobuf*, char*, char*);
static int64 rdint(Biobuf*);
static char *rdstring(Biobuf*);
static void rddata(Biobuf*, uchar**, int*);
static LSym *rdsym(Link*, Biobuf*, char*);

// The Go and C compilers, and the assembler, call writeobj to write
// out a Go object file.  The linker does not call this; the linker
// does not write out object files.
void
writeobj(Link *ctxt, Biobuf *b)
{
	int flag, found;
	Hist *h;
	LSym *s, *text, *etext, *curtext, *data, *edata;
	Plist *pl;
	Prog *p, *plink;
	Auto *a;

	// Build list of symbols, and assign instructions to lists.
	// Ignore ctxt->plist boundaries. There are no guarantees there,
	// and the C compilers and assemblers just use one big list.
	text = nil;
	curtext = nil;
	data = nil;
	etext = nil;
	edata = nil;
	for(pl = ctxt->plist; pl != nil; pl = pl->link) {
		for(p = pl->firstpc; p != nil; p = plink) {
			plink = p->link;
			p->link = nil;

			if(p->as == ctxt->arch->AEND)
				continue;

			if(p->as == ctxt->arch->ATYPE) {
				// Assume each TYPE instruction describes
				// a different local variable or parameter,
				// so no dedup.
				// Using only the TYPE instructions means
				// that we discard location information about local variables
				// in C and assembly functions; that information is inferred
				// from ordinary references, because there are no TYPE
				// instructions there. Without the type information, gdb can't
				// use the locations, so we don't bother to save them.
				// If something else could use them, we could arrange to
				// preserve them.
				if(curtext == nil)
					continue;
				a = emallocz(sizeof *a);
				a->asym = p->from.sym;
				a->aoffset = p->from.offset;
				a->type = ctxt->arch->symtype(&p->from);
				a->gotype = p->from.gotype;
				a->link = curtext->autom;
				curtext->autom = a;
				continue;
			}

			if(p->as == ctxt->arch->AGLOBL) {
				s = p->from.sym;
				if(s->seenglobl++)
					print("duplicate %P\n", p);
				if(s->onlist)
					sysfatal("symbol %s listed multiple times", s->name);
				s->onlist = 1;
				if(data == nil)
					data = s;
				else
					edata->next = s;
				s->next = nil;
				s->size = p->to.offset;
				if(s->type == 0 || s->type == SXREF)
					s->type = SBSS;
				flag = ctxt->arch->textflag(p);
				if(flag & DUPOK)
					s->dupok = 1;
				if(flag & RODATA)
					s->type = SRODATA;
				else if(flag & NOPTR)
					s->type = SNOPTRBSS;
				edata = s;
				continue;
			}

			if(p->as == ctxt->arch->ADATA) {
				savedata(ctxt, p->from.sym, p, "<input>");
				continue;
			}

			if(p->as == ctxt->arch->ATEXT) {
				s = p->from.sym;
				if(s == nil) {
					// func _() { }
					curtext = nil;
					continue;
				}
				if(s->text != nil)
					sysfatal("duplicate TEXT for %s", s->name);
				if(s->onlist)
					sysfatal("symbol %s listed multiple times", s->name);
				s->onlist = 1;
				if(text == nil)
					text = s;
				else
					etext->next = s;
				etext = s;
				flag = ctxt->arch->textflag(p);
				if(flag & DUPOK)
					s->dupok = 1;
				if(flag & NOSPLIT)
					s->nosplit = 1;
				s->next = nil;
				s->type = STEXT;
				s->text = p;
				s->etext = p;
				curtext = s;
				continue;
			}
			
			if(p->as == ctxt->arch->AFUNCDATA) {
				// Rewrite reference to go_args_stackmap(SB) to the Go-provided declaration information.
				if(curtext == nil) // func _() {}
					continue;
				if(strcmp(p->to.sym->name, "go_args_stackmap") == 0) {
					if(p->from.type != ctxt->arch->D_CONST || p->from.offset != FUNCDATA_ArgsPointerMaps)
						ctxt->diag("FUNCDATA use of go_args_stackmap(SB) without FUNCDATA_ArgsPointerMaps");
					p->to.sym = linklookup(ctxt, smprint("%s.args_stackmap", curtext->name), curtext->version);
				}
			}
			
			if(curtext == nil)
				continue;
			s = curtext;
			s->etext->link = p;
			s->etext = p;
		}
	}
	
	// Add reference to Go arguments for C or assembly functions without them.
	for(s = text; s != nil; s = s->next) {
		if(strncmp(s->name, "\"\".", 3) != 0)
			continue;
		found = 0;
		for(p = s->text; p != nil; p = p->link) {
			if(p->as == ctxt->arch->AFUNCDATA && p->from.type == ctxt->arch->D_CONST && p->from.offset == FUNCDATA_ArgsPointerMaps) {
				found = 1;
				break;
			}
		}
		if(!found) {
			p = appendp(ctxt, s->text);
			p->as = ctxt->arch->AFUNCDATA;
			p->from.type = ctxt->arch->D_CONST;
			p->from.offset = FUNCDATA_ArgsPointerMaps;
			if(ctxt->arch->thechar == '6' || ctxt->arch->thechar == '8')
				p->to.type = ctxt->arch->D_EXTERN;
			else {
				p->to.type = ctxt->arch->D_OREG;
				p->to.name = ctxt->arch->D_EXTERN;
			}
			p->to.sym = linklookup(ctxt, smprint("%s.args_stackmap", s->name), s->version);
		}
	}

	// Turn functions into machine code images.
	for(s = text; s != nil; s = s->next) {
		mkfwd(s);
		linkpatch(ctxt, s);
		ctxt->arch->follow(ctxt, s);
		ctxt->arch->addstacksplit(ctxt, s);
		ctxt->arch->assemble(ctxt, s);
		linkpcln(ctxt, s);
	}

	// Emit header.
	Bputc(b, 0);
	Bputc(b, 0);
	Bprint(b, "go13ld");
	Bputc(b, 1); // version

	// Emit autolib.
	for(h = ctxt->hist; h != nil; h = h->link)
		if(h->offset < 0)
			wrstring(b, h->name);
	wrstring(b, "");

	// Emit symbols.
	for(s = text; s != nil; s = s->next)
		writesym(ctxt, b, s);
	for(s = data; s != nil; s = s->next)
		writesym(ctxt, b, s);

	// Emit footer.
	Bputc(b, 0xff);
	Bputc(b, 0xff);
	Bprint(b, "go13ld");
}

static void
writesym(Link *ctxt, Biobuf *b, LSym *s)
{
	Reloc *r;
	int i, j, c, n;
	Pcln *pc;
	Prog *p;
	Auto *a;
	char *name;

	if(ctxt->debugasm) {
		Bprint(ctxt->bso, "%s ", s->name);
		if(s->version)
			Bprint(ctxt->bso, "v=%d ", s->version);
		if(s->type)
			Bprint(ctxt->bso, "t=%d ", s->type);
		if(s->dupok)
			Bprint(ctxt->bso, "dupok ");
		if(s->cfunc)
			Bprint(ctxt->bso, "cfunc ");
		if(s->nosplit)
			Bprint(ctxt->bso, "nosplit ");
		Bprint(ctxt->bso, "size=%lld value=%lld", (vlong)s->size, (vlong)s->value);
		if(s->type == STEXT) {
			Bprint(ctxt->bso, " args=%#llux locals=%#llux", (uvlong)s->args, (uvlong)s->locals);
			if(s->leaf)
				Bprint(ctxt->bso, " leaf");
		}
		Bprint(ctxt->bso, "\n");
		for(p=s->text; p != nil; p = p->link)
			Bprint(ctxt->bso, "\t%#06ux %P\n", (int)p->pc, p);
		for(i=0; i<s->np; ) {
			Bprint(ctxt->bso, "\t%#06ux", i);
			for(j=i; j<i+16 && j<s->np; j++)
				Bprint(ctxt->bso, " %02ux", s->p[j]);
			for(; j<i+16; j++)
				Bprint(ctxt->bso, "   ");
			Bprint(ctxt->bso, "  ");
			for(j=i; j<i+16 && j<s->np; j++) {
				c = s->p[j];
				if(' ' <= c && c <= 0x7e)
					Bprint(ctxt->bso, "%c", c);
				else
					Bprint(ctxt->bso, ".");
			}
			Bprint(ctxt->bso, "\n");
			i += 16;
		}
		for(i=0; i<s->nr; i++) {
			r = &s->r[i];
			name = "";
			if(r->sym != nil)
				name = r->sym->name;
			Bprint(ctxt->bso, "\trel %d+%d t=%d %s+%lld\n", (int)r->off, r->siz, r->type, name, (vlong)r->add);
		}
	}

	Bputc(b, 0xfe);
	wrint(b, s->type);
	wrstring(b, s->name);
	wrint(b, s->version);
	wrint(b, s->dupok);
	wrint(b, s->size);
	wrsym(b, s->gotype);
	wrdata(b, s->p, s->np);

	wrint(b, s->nr);
	for(i=0; i<s->nr; i++) {
		r = &s->r[i];
		wrint(b, r->off);
		wrint(b, r->siz);
		wrint(b, r->type);
		wrint(b, r->add);
		wrint(b, r->xadd);
		wrsym(b, r->sym);
		wrsym(b, r->xsym);
	}
	
	if(s->type == STEXT) {
		wrint(b, s->args);
		wrint(b, s->locals);
		wrint(b, s->nosplit);
		wrint(b, s->leaf | s->cfunc<<1);
		n = 0;
		for(a = s->autom; a != nil; a = a->link)
			n++;
		wrint(b, n);
		for(a = s->autom; a != nil; a = a->link) {
			wrsym(b, a->asym);
			wrint(b, a->aoffset);
			if(a->type == ctxt->arch->D_AUTO)
				wrint(b, A_AUTO);
			else if(a->type == ctxt->arch->D_PARAM)
				wrint(b, A_PARAM);
			else
				sysfatal("%s: invalid local variable type %d", s->name, a->type);
			wrsym(b, a->gotype);
		}

		pc = s->pcln;
		wrdata(b, pc->pcsp.p, pc->pcsp.n);
		wrdata(b, pc->pcfile.p, pc->pcfile.n);
		wrdata(b, pc->pcline.p, pc->pcline.n);
		wrint(b, pc->npcdata);
		for(i=0; i<pc->npcdata; i++)
			wrdata(b, pc->pcdata[i].p, pc->pcdata[i].n);
		wrint(b, pc->nfuncdata);
		for(i=0; i<pc->nfuncdata; i++)
			wrsym(b, pc->funcdata[i]);
		for(i=0; i<pc->nfuncdata; i++)
			wrint(b, pc->funcdataoff[i]);
		wrint(b, pc->nfile);
		for(i=0; i<pc->nfile; i++)
			wrpathsym(ctxt, b, pc->file[i]);
	}
}

static void
wrint(Biobuf *b, int64 sval)
{
	uint64 uv, v;
	uchar buf[10], *p;

	uv = ((uint64)sval<<1) ^ (uint64)(int64)(sval>>63);

	p = buf;
	for(v = uv; v >= 0x80; v >>= 7)
		*p++ = v | 0x80;
	*p++ = v;
	
	Bwrite(b, buf, p - buf);
}

static void
wrstring(Biobuf *b, char *s)
{
	wrdata(b, s, strlen(s));
}

// wrpath writes a path just like a string, but on windows, it
// translates '\\' to '/' in the process.
static void
wrpath(Link *ctxt, Biobuf *b, char *p)
{
	int i, n;
	if (!ctxt->windows || strchr(p, '\\') == nil) {
		wrstring(b, p);
		return;
	} else {
		n = strlen(p);
		wrint(b, n);
		for (i = 0; i < n; i++)
			Bputc(b, p[i] == '\\' ? '/' : p[i]);
	}
}

static void
wrdata(Biobuf *b, void *v, int n)
{
	wrint(b, n);
	Bwrite(b, v, n);
}

static void
wrpathsym(Link *ctxt, Biobuf *b, LSym *s)
{
	if(s == nil) {
		wrint(b, 0);
		wrint(b, 0);
		return;
	}
	wrpath(ctxt, b, s->name);
	wrint(b, s->version);
}

static void
wrsym(Biobuf *b, LSym *s)
{
	if(s == nil) {
		wrint(b, 0);
		wrint(b, 0);
		return;
	}
	wrstring(b, s->name);
	wrint(b, s->version);
}

static char startmagic[] = "\x00\x00go13ld";
static char endmagic[] = "\xff\xffgo13ld";

void
ldobjfile(Link *ctxt, Biobuf *f, char *pkg, int64 len, char *pn)
{
	int c;
	uchar buf[8];
	int64 start;
	char *lib;

	start = Boffset(f);
	ctxt->version++;
	memset(buf, 0, sizeof buf);
	Bread(f, buf, sizeof buf);
	if(memcmp(buf, startmagic, sizeof buf) != 0)
		sysfatal("%s: invalid file start %x %x %x %x %x %x %x %x", pn, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
	if((c = Bgetc(f)) != 1)
		sysfatal("%s: invalid file version number %d", pn, c);

	for(;;) {
		lib = rdstring(f);
		if(lib[0] == 0)
			break;
		addlib(ctxt, pkg, pn, lib);
	}
	
	for(;;) {
		c = Bgetc(f);
		Bungetc(f);
		if(c == 0xff)
			break;
		readsym(ctxt, f, pkg, pn);
	}
	
	memset(buf, 0, sizeof buf);
	Bread(f, buf, sizeof buf);
	if(memcmp(buf, endmagic, sizeof buf) != 0)
		sysfatal("%s: invalid file end", pn);
	
	if(Boffset(f) != start+len)
		sysfatal("%s: unexpected end at %lld, want %lld", pn, (vlong)Boffset(f), (vlong)(start+len));
}

static void
readsym(Link *ctxt, Biobuf *f, char *pkg, char *pn)
{
	int i, j, c, t, v, n, size, dupok;
	static int ndup;
	char *name;
	Reloc *r;
	LSym *s, *dup, *typ;
	Pcln *pc;
	Auto *a;
	
	if(Bgetc(f) != 0xfe)
		sysfatal("readsym out of sync");
	t = rdint(f);
	name = expandpkg(rdstring(f), pkg);
	v = rdint(f);
	if(v != 0 && v != 1)
		sysfatal("invalid symbol version %d", v);
	dupok = rdint(f);
	dupok &= 1;
	size = rdint(f);
	
	if(v != 0)
		v = ctxt->version;
	s = linklookup(ctxt, name, v);
	dup = nil;
	if(s->type != 0 && s->type != SXREF) {
		if(s->type != SBSS && s->type != SNOPTRBSS && !dupok && !s->dupok)
			sysfatal("duplicate symbol %s (types %d and %d) in %s and %s", s->name, s->type, t, s->file, pn);
		if(s->np > 0) {
			dup = s;
			s = linknewsym(ctxt, ".dup", ndup++); // scratch
		}
	}
	s->file = pkg;
	s->dupok = dupok;
	if(t == SXREF)
		sysfatal("bad sxref");
	if(t == 0)
		sysfatal("missing type for %s in %s", name, pn);
	s->type = t;
	if(s->size < size)
		s->size = size;
	typ = rdsym(ctxt, f, pkg);
	if(typ != nil) // if bss sym defined multiple times, take type from any one def
		s->gotype = typ;
	if(dup != nil && typ != nil)
		dup->gotype = typ;
	rddata(f, &s->p, &s->np);
	s->maxp = s->np;
	n = rdint(f);
	if(n > 0) {
		s->r = emallocz(n * sizeof s->r[0]);
		s->nr = n;
		s->maxr = n;
		for(i=0; i<n; i++) {
			r = &s->r[i];
			r->off = rdint(f);
			r->siz = rdint(f);
			r->type = rdint(f);
			r->add = rdint(f);
			r->xadd = rdint(f);
			r->sym = rdsym(ctxt, f, pkg);
			r->xsym = rdsym(ctxt, f, pkg);
		}
	}
	
	if(s->np > 0 && dup != nil && dup->np > 0 && strncmp(s->name, "gclocals·", 10) == 0) {
		// content-addressed garbage collection liveness bitmap symbol.
		// double check for hash collisions.
		if(s->np != dup->np || memcmp(s->p, dup->p, s->np) != 0)
			sysfatal("dupok hash collision for %s in %s and %s", s->name, s->file, pn);
	}
	
	if(s->type == STEXT) {
		s->args = rdint(f);
		s->locals = rdint(f);
		s->nosplit = rdint(f);
		v = rdint(f);
		s->leaf = v&1;
		s->cfunc = v&2;
		n = rdint(f);
		for(i=0; i<n; i++) {
			a = emallocz(sizeof *a);
			a->asym = rdsym(ctxt, f, pkg);
			a->aoffset = rdint(f);
			a->type = rdint(f);
			a->gotype = rdsym(ctxt, f, pkg);
			a->link = s->autom;
			s->autom = a;
		}

		s->pcln = emallocz(sizeof *s->pcln);
		pc = s->pcln;
		rddata(f, &pc->pcsp.p, &pc->pcsp.n);
		rddata(f, &pc->pcfile.p, &pc->pcfile.n);
		rddata(f, &pc->pcline.p, &pc->pcline.n);
		n = rdint(f);
		pc->pcdata = emallocz(n * sizeof pc->pcdata[0]);
		pc->npcdata = n;
		for(i=0; i<n; i++)
			rddata(f, &pc->pcdata[i].p, &pc->pcdata[i].n);
		n = rdint(f);
		pc->funcdata = emallocz(n * sizeof pc->funcdata[0]);
		pc->funcdataoff = emallocz(n * sizeof pc->funcdataoff[0]);
		pc->nfuncdata = n;
		for(i=0; i<n; i++)
			pc->funcdata[i] = rdsym(ctxt, f, pkg);
		for(i=0; i<n; i++)
			pc->funcdataoff[i] = rdint(f);
		n = rdint(f);
		pc->file = emallocz(n * sizeof pc->file[0]);
		pc->nfile = n;
		for(i=0; i<n; i++)
			pc->file[i] = rdsym(ctxt, f, pkg);

		if(dup == nil) {
			if(s->onlist)
				sysfatal("symbol %s listed multiple times", s->name);
			s->onlist = 1;
			if(ctxt->etextp)
				ctxt->etextp->next = s;
			else
				ctxt->textp = s;
			ctxt->etextp = s;
		}
	}

	if(ctxt->debugasm) {
		Bprint(ctxt->bso, "%s ", s->name);
		if(s->version)
			Bprint(ctxt->bso, "v=%d ", s->version);
		if(s->type)
			Bprint(ctxt->bso, "t=%d ", s->type);
		if(s->dupok)
			Bprint(ctxt->bso, "dupok ");
		if(s->cfunc)
			Bprint(ctxt->bso, "cfunc ");
		if(s->nosplit)
			Bprint(ctxt->bso, "nosplit ");
		Bprint(ctxt->bso, "size=%lld value=%lld", (vlong)s->size, (vlong)s->value);
		if(s->type == STEXT)
			Bprint(ctxt->bso, " args=%#llux locals=%#llux", (uvlong)s->args, (uvlong)s->locals);
		Bprint(ctxt->bso, "\n");
		for(i=0; i<s->np; ) {
			Bprint(ctxt->bso, "\t%#06ux", i);
			for(j=i; j<i+16 && j<s->np; j++)
				Bprint(ctxt->bso, " %02ux", s->p[j]);
			for(; j<i+16; j++)
				Bprint(ctxt->bso, "   ");
			Bprint(ctxt->bso, "  ");
			for(j=i; j<i+16 && j<s->np; j++) {
				c = s->p[j];
				if(' ' <= c && c <= 0x7e)
					Bprint(ctxt->bso, "%c", c);
				else
					Bprint(ctxt->bso, ".");
			}
			Bprint(ctxt->bso, "\n");
			i += 16;
		}
		for(i=0; i<s->nr; i++) {
			r = &s->r[i];
			Bprint(ctxt->bso, "\trel %d+%d t=%d %s+%lld\n", (int)r->off, r->siz, r->type, r->sym->name, (vlong)r->add);
		}
	}
}

static int64
rdint(Biobuf *f)
{
	int c;
	uint64 uv;
	int shift;
	
	uv = 0;
	for(shift = 0;; shift += 7) {
		if(shift >= 64)
			sysfatal("corrupt input");
		c = Bgetc(f);
		uv |= (uint64)(c & 0x7F) << shift;
		if(!(c & 0x80))
			break;
	}

	return (int64)(uv>>1) ^ ((int64)((uint64)uv<<63)>>63);
}

static char*
rdstring(Biobuf *f)
{
	int n;
	char *p;
	
	n = rdint(f);
	p = emallocz(n+1);
	Bread(f, p, n);
	return p;
}

static void
rddata(Biobuf *f, uchar **pp, int *np)
{
	*np = rdint(f);
	*pp = emallocz(*np);
	Bread(f, *pp, *np);
}

static LSym*
rdsym(Link *ctxt, Biobuf *f, char *pkg)
{
	int n, v;
	char *p;
	LSym *s;
	
	n = rdint(f);
	if(n == 0) {
		rdint(f);
		return nil;
	}
	p = emallocz(n+1);
	Bread(f, p, n);
	v = rdint(f);
	if(v != 0)
		v = ctxt->version;
	s = linklookup(ctxt, expandpkg(p, pkg), v);
	
	if(v == 0 && s->name[0] == '$' && s->type == 0) {
		if(strncmp(s->name, "$f32.", 5) == 0) {
			int32 i32;
			i32 = strtoul(s->name+5, nil, 16);
			s->type = SRODATA;
			adduint32(ctxt, s, i32);
			s->reachable = 0;
		} else if(strncmp(s->name, "$f64.", 5) == 0) {
			int64 i64;
			i64 = strtoull(s->name+5, nil, 16);
			s->type = SRODATA;
			adduint64(ctxt, s, i64);
			s->reachable = 0;
		}
	}

	return s;
}
