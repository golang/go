/*
Derived from Plan 9 from User Space's src/libmach/elf.h, elf.c
http://code.swtch.com/plan9port/src/tip/src/libmach/

	Copyright © 2004 Russ Cox.
	Portions Copyright © 2008-2010 Google Inc.
	Portions Copyright © 2010 The Go Authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include	"l.h"
#include	"lib.h"
#include	"../ld/elf.h"

enum
{
	ElfClassNone = 0,
	ElfClass32,
	ElfClass64,

	ElfDataNone = 0,
	ElfDataLsb,
	ElfDataMsb,

	ElfTypeNone = 0,
	ElfTypeRelocatable,
	ElfTypeExecutable,
	ElfTypeSharedObject,
	ElfTypeCore,
	/* 0xFF00 - 0xFFFF reserved for processor-specific types */

	ElfMachNone = 0,
	ElfMach32100,		/* AT&T WE 32100 */
	ElfMachSparc,		/* SPARC */
	ElfMach386,		/* Intel 80386 */
	ElfMach68000,		/* Motorola 68000 */
	ElfMach88000,		/* Motorola 88000 */
	ElfMach486,		/* Intel 80486, no longer used */
	ElfMach860,		/* Intel 80860 */
	ElfMachMips,		/* MIPS RS3000 */
	ElfMachS370,		/* IBM System/370 */
	ElfMachMipsLe,	/* MIPS RS3000 LE */
	ElfMachParisc = 15,		/* HP PA RISC */
	ElfMachVpp500 = 17,	/* Fujitsu VPP500 */
	ElfMachSparc32Plus,	/* SPARC V8+ */
	ElfMach960,		/* Intel 80960 */
	ElfMachPower,		/* PowerPC */
	ElfMachPower64,	/* PowerPC 64-bit */
	ElfMachS390,		/* IBM System/390 */
	ElfMachV800 = 36,	/* NEC V800 */
	ElfMachFr20,		/* Fujitsu FR20 */
	ElfMachRh32,		/* TRW RH-32 */
	ElfMachRce,		/* Motorola RCE */
	ElfMachArm,		/* ARM */
	ElfMachAlpha,		/* Digital Alpha */
	ElfMachSH,		/* Hitachi SH */
	ElfMachSparc9,		/* SPARC V9 */
	ElfMachAmd64 = 62,
	/* and the list goes on... */

	ElfAbiNone = 0,
	ElfAbiSystemV = 0,	/* [sic] */
	ElfAbiHPUX,
	ElfAbiNetBSD,
	ElfAbiLinux,
	ElfAbiSolaris = 6,
	ElfAbiAix,
	ElfAbiIrix,
	ElfAbiFreeBSD,
	ElfAbiTru64,
	ElfAbiModesto,
	ElfAbiOpenBSD,
	ElfAbiARM = 97,
	ElfAbiEmbedded = 255,

	/* some of sections 0xFF00 - 0xFFFF reserved for various things */
	ElfSectNone = 0,
	ElfSectProgbits,
	ElfSectSymtab,
	ElfSectStrtab,
	ElfSectRela,
	ElfSectHash,
	ElfSectDynamic,
	ElfSectNote,
	ElfSectNobits,
	ElfSectRel,
	ElfSectShlib,
	ElfSectDynsym,

	ElfSectFlagWrite = 0x1,
	ElfSectFlagAlloc = 0x2,
	ElfSectFlagExec = 0x4,
	/* 0xF0000000 are reserved for processor specific */

	ElfSymBindLocal = 0,
	ElfSymBindGlobal,
	ElfSymBindWeak,
	/* 13-15 reserved */

	ElfSymTypeNone = 0,
	ElfSymTypeObject,
	ElfSymTypeFunc,
	ElfSymTypeSection,
	ElfSymTypeFile,
	/* 13-15 reserved */

	ElfSymShnNone = 0,
	ElfSymShnAbs = 0xFFF1,
	ElfSymShnCommon = 0xFFF2,
	/* 0xFF00-0xFF1F reserved for processors */
	/* 0xFF20-0xFF3F reserved for operating systems */

	ElfProgNone = 0,
	ElfProgLoad,
	ElfProgDynamic,
	ElfProgInterp,
	ElfProgNote,
	ElfProgShlib,
	ElfProgPhdr,

	ElfProgFlagExec = 0x1,
	ElfProgFlagWrite = 0x2,
	ElfProgFlagRead = 0x4,

	ElfNotePrStatus = 1,
	ElfNotePrFpreg = 2,
	ElfNotePrPsinfo = 3,
	ElfNotePrTaskstruct = 4,
	ElfNotePrAuxv = 6,
	ElfNotePrXfpreg = 0x46e62b7f	/* for gdb/386 */
};

typedef struct ElfHdrBytes ElfHdrBytes;
typedef struct ElfSectBytes ElfSectBytes;
typedef struct ElfProgBytes ElfProgBytes;
typedef struct ElfSymBytes ElfSymBytes;

typedef struct ElfHdrBytes64 ElfHdrBytes64;
typedef struct ElfSectBytes64 ElfSectBytes64;
typedef struct ElfProgBytes64 ElfProgBytes64;
typedef struct ElfSymBytes64 ElfSymBytes64;

struct ElfHdrBytes
{
	uchar	ident[16];
	uchar	type[2];
	uchar	machine[2];
	uchar	version[4];
	uchar	entry[4];
	uchar	phoff[4];
	uchar	shoff[4];
	uchar	flags[4];
	uchar	ehsize[2];
	uchar	phentsize[2];
	uchar	phnum[2];
	uchar	shentsize[2];
	uchar	shnum[2];
	uchar	shstrndx[2];
};

struct ElfHdrBytes64
{
	uchar	ident[16];
	uchar	type[2];
	uchar	machine[2];
	uchar	version[4];
	uchar	entry[8];
	uchar	phoff[8];
	uchar	shoff[8];
	uchar	flags[4];
	uchar	ehsize[2];
	uchar	phentsize[2];
	uchar	phnum[2];
	uchar	shentsize[2];
	uchar	shnum[2];
	uchar	shstrndx[2];
};

struct ElfSectBytes
{
	uchar	name[4];
	uchar	type[4];
	uchar	flags[4];
	uchar	addr[4];
	uchar	off[4];
	uchar	size[4];
	uchar	link[4];
	uchar	info[4];
	uchar	align[4];
	uchar	entsize[4];
};

struct ElfSectBytes64
{
	uchar	name[4];
	uchar	type[4];
	uchar	flags[8];
	uchar	addr[8];
	uchar	off[8];
	uchar	size[8];
	uchar	link[4];
	uchar	info[4];
	uchar	align[8];
	uchar	entsize[8];
};

struct ElfSymBytes
{
	uchar	name[4];
	uchar	value[4];
	uchar	size[4];
	uchar	info;	/* top4: bind, bottom4: type */
	uchar	other;
	uchar	shndx[2];
};

struct ElfSymBytes64
{
	uchar	name[4];
	uchar	info;	/* top4: bind, bottom4: type */
	uchar	other;
	uchar	shndx[2];
	uchar	value[8];
	uchar	size[8];
};

typedef struct ElfSect ElfSect;
typedef struct ElfObj ElfObj;
typedef struct ElfSym ElfSym;

struct ElfSect
{
	char		*name;
	uint32	type;
	uint64	flags;
	uint64	addr;
	uint64	off;
	uint64	size;
	uint32	link;
	uint32	info;
	uint64	align;
	uint64	entsize;
	uchar	*base;
	LSym	*sym;
};

struct ElfObj
{
	Biobuf	*f;
	int64	base;	// offset in f where ELF begins
	int64	len;		// length of ELF
	int	is64;
	char	*name;

	Endian	*e;
	ElfSect	*sect;
	uint		nsect;
	char		*shstrtab;
	int		nsymtab;
	ElfSect	*symtab;
	ElfSect	*symstr;

	uint32	type;
	uint32	machine;
	uint32	version;
	uint64	entry;
	uint64	phoff;
	uint64	shoff;
	uint32	flags;
	uint32	ehsize;
	uint32	phentsize;
	uint32	phnum;
	uint32	shentsize;
	uint32	shnum;
	uint32	shstrndx;
};

struct ElfSym
{
	char*	name;
	uint64	value;
	uint64	size;
	uchar	bind;
	uchar	type;
	uchar	other;
	uint16	shndx;
	LSym*	sym;
};

uchar ElfMagic[4] = { 0x7F, 'E', 'L', 'F' };

static ElfSect*	section(ElfObj*, char*);
static int	map(ElfObj*, ElfSect*);
static int	readsym(ElfObj*, int i, ElfSym*, int);
static int	reltype(char*, int, uchar*);

int
valuecmp(LSym *a, LSym *b)
{
	if(a->value < b->value)
		return -1;
	if(a->value > b->value)
		return +1;
	return 0;
}

void
ldelf(Biobuf *f, char *pkg, int64 len, char *pn)
{
	int32 base;
	uint64 add, info;
	char *name;
	int i, j, rela, is64, n;
	uchar hdrbuf[64];
	uchar *p;
	ElfHdrBytes *hdr;
	ElfObj *obj;
	ElfSect *sect, *rsect;
	ElfSym sym;
	Endian *e;
	Reloc *r, *rp;
	LSym *s;
	LSym **symbols;

	symbols = nil;

	if(debug['v'])
		Bprint(&bso, "%5.2f ldelf %s\n", cputime(), pn);

	ctxt->version++;
	base = Boffset(f);

	if(Bread(f, hdrbuf, sizeof hdrbuf) != sizeof hdrbuf)
		goto bad;
	hdr = (ElfHdrBytes*)hdrbuf;
	if(memcmp(hdr->ident, ElfMagic, 4) != 0)
		goto bad;
	switch(hdr->ident[5]) {
	case ElfDataLsb:
		e = &le;
		break;
	case ElfDataMsb:
		e = &be;
		break;
	default:
		goto bad;
	}

	// read header
	obj = mal(sizeof *obj);
	obj->e = e;
	obj->f = f;
	obj->base = base;
	obj->len = len;
	obj->name = pn;
	
	is64 = 0;
	if(hdr->ident[4] == ElfClass64) {
		ElfHdrBytes64* hdr;

		is64 = 1;
		hdr = (ElfHdrBytes64*)hdrbuf;
		obj->type = e->e16(hdr->type);
		obj->machine = e->e16(hdr->machine);
		obj->version = e->e32(hdr->version);
		obj->phoff = e->e64(hdr->phoff);
		obj->shoff = e->e64(hdr->shoff);
		obj->flags = e->e32(hdr->flags);
		obj->ehsize = e->e16(hdr->ehsize);
		obj->phentsize = e->e16(hdr->phentsize);
		obj->phnum = e->e16(hdr->phnum);
		obj->shentsize = e->e16(hdr->shentsize);
		obj->shnum = e->e16(hdr->shnum);
		obj->shstrndx = e->e16(hdr->shstrndx);
	} else {
		obj->type = e->e16(hdr->type);
		obj->machine = e->e16(hdr->machine);
		obj->version = e->e32(hdr->version);
		obj->entry = e->e32(hdr->entry);
		obj->phoff = e->e32(hdr->phoff);
		obj->shoff = e->e32(hdr->shoff);
		obj->flags = e->e32(hdr->flags);
		obj->ehsize = e->e16(hdr->ehsize);
		obj->phentsize = e->e16(hdr->phentsize);
		obj->phnum = e->e16(hdr->phnum);
		obj->shentsize = e->e16(hdr->shentsize);
		obj->shnum = e->e16(hdr->shnum);
		obj->shstrndx = e->e16(hdr->shstrndx);
	}
	obj->is64 = is64;
	
	if(hdr->ident[6] != obj->version)
		goto bad;

	if(e->e16(hdr->type) != ElfTypeRelocatable) {
		diag("%s: elf but not elf relocatable object", pn);
		return;
	}

	switch(thechar) {
	default:
		diag("%s: elf %s unimplemented", pn, thestring);
		return;
	case '5':
		if(e != &le || obj->machine != ElfMachArm || hdr->ident[4] != ElfClass32) {
			diag("%s: elf object but not arm", pn);
			return;
		}
		break;
	case '6':
		if(e != &le || obj->machine != ElfMachAmd64 || hdr->ident[4] != ElfClass64) {
			diag("%s: elf object but not amd64", pn);
			return;
		}
		break;
	case '8':
		if(e != &le || obj->machine != ElfMach386 || hdr->ident[4] != ElfClass32) {
			diag("%s: elf object but not 386", pn);
			return;
		}
		break;
	case '9':
		if(obj->machine != ElfMachPower64 || hdr->ident[4] != ElfClass64) {
			diag("%s: elf object but not ppc64", pn);
			return;
		}
		break;
	}

	// load section list into memory.
	obj->sect = mal(obj->shnum*sizeof obj->sect[0]);
	obj->nsect = obj->shnum;
	for(i=0; i<obj->nsect; i++) {
		if(Bseek(f, base+obj->shoff+i*obj->shentsize, 0) < 0)
			goto bad;
		sect = &obj->sect[i];
		if(is64) {
			ElfSectBytes64 b;

			werrstr("short read");
			if(Bread(f, &b, sizeof b) != sizeof b)
				goto bad;

			sect->name = (char*)(uintptr)e->e32(b.name);
			sect->type = e->e32(b.type);
			sect->flags = e->e64(b.flags);
			sect->addr = e->e64(b.addr);
			sect->off = e->e64(b.off);
			sect->size = e->e64(b.size);
			sect->link = e->e32(b.link);
			sect->info = e->e32(b.info);
			sect->align = e->e64(b.align);
			sect->entsize = e->e64(b.entsize);
		} else {
			ElfSectBytes b;

			werrstr("short read");
			if(Bread(f, &b, sizeof b) != sizeof b)
				goto bad;
		
			sect->name = (char*)(uintptr)e->e32(b.name);
			sect->type = e->e32(b.type);
			sect->flags = e->e32(b.flags);
			sect->addr = e->e32(b.addr);
			sect->off = e->e32(b.off);
			sect->size = e->e32(b.size);
			sect->link = e->e32(b.link);
			sect->info = e->e32(b.info);
			sect->align = e->e32(b.align);
			sect->entsize = e->e32(b.entsize);
		}
	}

	// read section string table and translate names
	if(obj->shstrndx >= obj->nsect) {
		werrstr("shstrndx out of range %d >= %d", obj->shstrndx, obj->nsect);
		goto bad;
	}
	sect = &obj->sect[obj->shstrndx];
	if(map(obj, sect) < 0)
		goto bad;
	for(i=0; i<obj->nsect; i++)
		if(obj->sect[i].name != nil)
			obj->sect[i].name = (char*)sect->base + (uintptr)obj->sect[i].name;
	
	// load string table for symbols into memory.
	obj->symtab = section(obj, ".symtab");
	if(obj->symtab == nil) {
		// our work is done here - no symbols means nothing can refer to this file
		return;
	}
	if(obj->symtab->link <= 0 || obj->symtab->link >= obj->nsect) {
		diag("%s: elf object has symbol table with invalid string table link", pn);
		return;
	}
	obj->symstr = &obj->sect[obj->symtab->link];
	if(is64)
		obj->nsymtab = obj->symtab->size / sizeof(ElfSymBytes64);
	else
		obj->nsymtab = obj->symtab->size / sizeof(ElfSymBytes);
	
	if(map(obj, obj->symtab) < 0)
		goto bad;
	if(map(obj, obj->symstr) < 0)
		goto bad;

	// load text and data segments into memory.
	// they are not as small as the section lists, but we'll need
	// the memory anyway for the symbol images, so we might
	// as well use one large chunk.
	
	// create symbols for mapped sections
	for(i=0; i<obj->nsect; i++) {
		sect = &obj->sect[i];
		if((sect->type != ElfSectProgbits && sect->type != ElfSectNobits) || !(sect->flags&ElfSectFlagAlloc))
			continue;
		if(sect->type != ElfSectNobits && map(obj, sect) < 0)
			goto bad;
		
		name = smprint("%s(%s)", pkg, sect->name);
		s = linklookup(ctxt, name, ctxt->version);
		free(name);
		switch((int)sect->flags&(ElfSectFlagAlloc|ElfSectFlagWrite|ElfSectFlagExec)) {
		default:
			werrstr("unexpected flags for ELF section %s", sect->name);
			goto bad;
		case ElfSectFlagAlloc:
			s->type = SRODATA;
			break;
		case ElfSectFlagAlloc + ElfSectFlagWrite:
			s->type = SNOPTRDATA;
			break;
		case ElfSectFlagAlloc + ElfSectFlagExec:
			s->type = STEXT;
			break;
		}
		if(sect->type == ElfSectProgbits) {
			s->p = sect->base;
			s->np = sect->size;
		}
		s->size = sect->size;
		s->align = sect->align;
		sect->sym = s;
	}

	// enter sub-symbols into symbol table.
	// symbol 0 is the null symbol.
	symbols = malloc(obj->nsymtab * sizeof(symbols[0]));
	if(symbols == nil) {
		diag("out of memory");
		errorexit();
	}
	for(i=1; i<obj->nsymtab; i++) {
		if(readsym(obj, i, &sym, 1) < 0)
			goto bad;
		symbols[i] = sym.sym;
		if(sym.type != ElfSymTypeFunc && sym.type != ElfSymTypeObject && sym.type != ElfSymTypeNone)
			continue;
		if(sym.shndx == ElfSymShnCommon) {
			s = sym.sym;
			if(s->size < sym.size)
				s->size = sym.size;
			if(s->type == 0 || s->type == SXREF)
				s->type = SNOPTRBSS;
			continue;
		}
		if(sym.shndx >= obj->nsect || sym.shndx == 0)
			continue;
		// even when we pass needSym == 1 to readsym, it might still return nil to skip some unwanted symbols
		if(sym.sym == S)
			continue;
		sect = obj->sect+sym.shndx;
		if(sect->sym == nil) {
			if(strncmp(sym.name, ".Linfo_string", 13) == 0) // clang does this
				continue;
			diag("%s: sym#%d: ignoring %s in section %d (type %d)", pn, i, sym.name, sym.shndx, sym.type);
			continue;
		}
		s = sym.sym;
		if(s->outer != S) {
			if(s->dupok)
				continue;
			diag("%s: duplicate symbol reference: %s in both %s and %s", pn, s->name, s->outer->name, sect->sym->name);
			errorexit();
		}
		s->sub = sect->sym->sub;
		sect->sym->sub = s;
		s->type = sect->sym->type | (s->type&~SMASK) | SSUB;
		if(!(s->cgoexport & CgoExportDynamic))
			s->dynimplib = nil;  // satisfy dynimport
		s->value = sym.value;
		s->size = sym.size;
		s->outer = sect->sym;
		if(sect->sym->type == STEXT) {
			if(s->external && !s->dupok)
					diag("%s: duplicate definition of %s", pn, s->name);
			s->external = 1;
		}
	}
	
	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for(i=0; i<obj->nsect; i++) {
		s = obj->sect[i].sym;
		if(s == S)
			continue;
		if(s->sub)
			s->sub = listsort(s->sub, valuecmp, offsetof(LSym, sub));
		if(s->type == STEXT) {
			if(s->onlist)
				sysfatal("symbol %s listed multiple times", s->name);
			s->onlist = 1;
			if(ctxt->etextp)
				ctxt->etextp->next = s;
			else
				ctxt->textp = s;
			ctxt->etextp = s;
			for(s = s->sub; s != S; s = s->sub) {
				if(s->onlist)
					sysfatal("symbol %s listed multiple times", s->name);
				s->onlist = 1;
				ctxt->etextp->next = s;
				ctxt->etextp = s;
			}
		}
	}

	// load relocations
	for(i=0; i<obj->nsect; i++) {
		rsect = &obj->sect[i];
		if(rsect->type != ElfSectRela && rsect->type != ElfSectRel)
			continue;
		if(rsect->info >= obj->nsect || obj->sect[rsect->info].base == nil)
			continue;
		sect = &obj->sect[rsect->info];
		if(map(obj, rsect) < 0)
			goto bad;
		rela = rsect->type == ElfSectRela;
		n = rsect->size/(4+4*is64)/(2+rela);
		r = mal(n*sizeof r[0]);
		p = rsect->base;
		for(j=0; j<n; j++) {
			add = 0;
			rp = &r[j];
			if(is64) {
				// 64-bit rel/rela
				rp->off = e->e64(p);
				p += 8;
				info = e->e64(p);
				p += 8;
				if(rela) {
					add = e->e64(p);
					p += 8;
				}
			} else {
				// 32-bit rel/rela
				rp->off = e->e32(p);
				p += 4;
				info = e->e32(p);
				info = info>>8<<32 | (info&0xff);	// convert to 64-bit info
				p += 4;
				if(rela) {
					add = e->e32(p);
					p += 4;
				}
			}
			if((info & 0xffffffff) == 0) { // skip R_*_NONE relocation
				j--;
				n--;
				continue;
			}
			if((info >> 32) == 0) { // absolute relocation, don't bother reading the null symbol
				rp->sym = S;
			} else {
				if(readsym(obj, info>>32, &sym, 0) < 0)
					goto bad;
				sym.sym = symbols[info>>32];
				if(sym.sym == nil) {
					werrstr("%s#%d: reloc of invalid sym #%d %s shndx=%d type=%d",
						sect->sym->name, j, (int)(info>>32), sym.name, sym.shndx, sym.type);
					goto bad;
				}
				rp->sym = sym.sym;
			}
			rp->type = reltype(pn, (uint32)info, &rp->siz);
			if(rela)
				rp->add = add;
			else {
				// load addend from image
				if(rp->siz == 4)
					rp->add = e->e32(sect->base+rp->off);
				else if(rp->siz == 8)
					rp->add = e->e64(sect->base+rp->off);
				else
					diag("invalid rela size %d", rp->siz);
			}
			if(rp->siz == 4)
				rp->add = (int32)rp->add;
			//print("rel %s %d %d %s %#llx\n", sect->sym->name, rp->type, rp->siz, rp->sym->name, rp->add);
		}
		qsort(r, n, sizeof r[0], rbyoff);	// just in case
		
		s = sect->sym;
		s->r = r;
		s->nr = n;
	}
	free(symbols);

	return;

bad:
	diag("%s: malformed elf file: %r", pn);
	free(symbols);
}

static ElfSect*
section(ElfObj *obj, char *name)
{
	int i;
	
	for(i=0; i<obj->nsect; i++)
		if(obj->sect[i].name && name && strcmp(obj->sect[i].name, name) == 0)
			return &obj->sect[i];
	return nil;
}

static int
map(ElfObj *obj, ElfSect *sect)
{
	if(sect->base != nil)
		return 0;

	if(sect->off+sect->size > obj->len) {
		werrstr("elf section past end of file");
		return -1;
	}

	sect->base = mal(sect->size);
	werrstr("short read");
	if(Bseek(obj->f, obj->base+sect->off, 0) < 0 || Bread(obj->f, sect->base, sect->size) != sect->size)
		return -1;
	
	return 0;
}

static int
readsym(ElfObj *obj, int i, ElfSym *sym, int needSym)
{
	LSym *s;

	if(i >= obj->nsymtab || i < 0) {
		werrstr("invalid elf symbol index");
		return -1;
	}
	if(i == 0) {
		diag("readym: read null symbol!");
	}

	if(obj->is64) {
		ElfSymBytes64 *b;
		
		b = (ElfSymBytes64*)(obj->symtab->base + i*sizeof *b);
		sym->name = (char*)obj->symstr->base + obj->e->e32(b->name);
		sym->value = obj->e->e64(b->value);
		sym->size = obj->e->e64(b->size);
		sym->shndx = obj->e->e16(b->shndx);
		sym->bind = b->info>>4;
		sym->type = b->info&0xf;
		sym->other = b->other;
	} else {
		ElfSymBytes *b;
		
		b = (ElfSymBytes*)(obj->symtab->base + i*sizeof *b);
		sym->name = (char*)obj->symstr->base + obj->e->e32(b->name);
		sym->value = obj->e->e32(b->value);
		sym->size = obj->e->e32(b->size);
		sym->shndx = obj->e->e16(b->shndx);
		sym->bind = b->info>>4;
		sym->type = b->info&0xf;
		sym->other = b->other;
	}

	s = nil;
	if(strcmp(sym->name, "_GLOBAL_OFFSET_TABLE_") == 0)
		sym->name = ".got";
	switch(sym->type) {
	case ElfSymTypeSection:
		s = obj->sect[sym->shndx].sym;
		break;
	case ElfSymTypeObject:
	case ElfSymTypeFunc:
	case ElfSymTypeNone:
		switch(sym->bind) {
		case ElfSymBindGlobal:
			if(needSym) {
				s = linklookup(ctxt, sym->name, 0);
				// for global scoped hidden symbols we should insert it into
				// symbol hash table, but mark them as hidden.
				// __i686.get_pc_thunk.bx is allowed to be duplicated, to
				// workaround that we set dupok.
				// TODO(minux): correctly handle __i686.get_pc_thunk.bx without
				// set dupok generally. See http://codereview.appspot.com/5823055/
				// comment #5 for details.
				if(s && sym->other == 2) {
					s->type |= SHIDDEN;
					s->dupok = 1;
				}
			}
			break;
		case ElfSymBindLocal:
			if(thechar == '5' && (strncmp(sym->name, "$a", 2) == 0 || strncmp(sym->name, "$d", 2) == 0)) {
				// binutils for arm generate these mapping
				// symbols, ignore these
				break;
			}
			if(needSym) {
				// local names and hidden visiblity global names are unique
				// and should only reference by its index, not name, so we
				// don't bother to add them into hash table
				s = linknewsym(ctxt, sym->name, ctxt->version);
				s->type |= SHIDDEN;
			}
			break;
		case ElfSymBindWeak:
			if(needSym) {
				s = linknewsym(ctxt, sym->name, 0);
				if(sym->other == 2)
					s->type |= SHIDDEN;
			}
			break;
		default:
			werrstr("%s: invalid symbol binding %d", sym->name, sym->bind);
			return -1;
		}
		break;
	}
	if(s != nil && s->type == 0 && sym->type != ElfSymTypeSection)
		s->type = SXREF;
	sym->sym = s;

	return 0;
}

int
rbyoff(const void *va, const void *vb)
{
	Reloc *a, *b;
	
	a = (Reloc*)va;
	b = (Reloc*)vb;
	if(a->off < b->off)
		return -1;
	if(a->off > b->off)
		return +1;
	return 0;
}

#define R(x, y) ((x)|((y)<<24))

static int
reltype(char *pn, int elftype, uchar *siz)
{
	switch(R(thechar, elftype)) {
	default:
		diag("%s: unknown relocation type %d; compiled without -fpic?", pn, elftype);
	case R('5', R_ARM_ABS32):
	case R('5', R_ARM_GOT32):
	case R('5', R_ARM_PLT32):
	case R('5', R_ARM_GOTOFF):
	case R('5', R_ARM_GOTPC):
	case R('5', R_ARM_THM_PC22):
	case R('5', R_ARM_REL32):
	case R('5', R_ARM_CALL):
	case R('5', R_ARM_V4BX):
	case R('5', R_ARM_GOT_PREL):
	case R('5', R_ARM_PC24):
	case R('5', R_ARM_JUMP24):
	case R('6', R_X86_64_PC32):
	case R('6', R_X86_64_PLT32):
	case R('6', R_X86_64_GOTPCREL):
	case R('8', R_386_32):
	case R('8', R_386_PC32):
	case R('8', R_386_GOT32):
	case R('8', R_386_PLT32):
	case R('8', R_386_GOTOFF):
	case R('8', R_386_GOTPC):
		*siz = 4;
		break;
	case R('6', R_X86_64_64):
		*siz = 8;
		break;
	}

	return 256+elftype;
}
