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

enum {
	MACHO_FAKE_GOTPCREL = 100,	// from macho.h
	
	N_EXT = 0x01,
	N_TYPE = 0x1e,
	N_STAB = 0xe0,
};

typedef struct MachoObj MachoObj;
typedef struct MachoCmd MachoCmd;
typedef struct MachoSeg MachoSeg;
typedef struct MachoSect MachoSect;
typedef struct MachoRel MachoRel;
typedef struct MachoSymtab MachoSymtab;
typedef struct MachoSym MachoSym;
typedef struct MachoDysymtab MachoDysymtab;

enum
{
	MachoCpuVax = 1,
	MachoCpu68000 = 6,
	MachoCpu386 = 7,
	MachoCpuAmd64 = 0x1000007,
	MachoCpuMips = 8,
	MachoCpu98000 = 10,
	MachoCpuHppa = 11,
	MachoCpuArm = 12,
	MachoCpu88000 = 13,
	MachoCpuSparc = 14,
	MachoCpu860 = 15,
	MachoCpuAlpha = 16,
	MachoCpuPower = 18,

	MachoCmdSegment = 1,
	MachoCmdSymtab = 2,
	MachoCmdSymseg = 3,
	MachoCmdThread = 4,
	MachoCmdDysymtab = 11,
	MachoCmdSegment64 = 25,

	MachoFileObject = 1,
	MachoFileExecutable = 2,
	MachoFileFvmlib = 3,
	MachoFileCore = 4,
	MachoFilePreload = 5,
};

struct MachoSeg
{
	char name[16+1];
	uint64 vmaddr;
	uint64 vmsize;
	uint32 fileoff;
	uint32 filesz;
	uint32 maxprot;
	uint32 initprot;
	uint32 nsect;
	uint32 flags;
	MachoSect *sect;
};

struct MachoSect
{
	char	name[16+1];
	char	segname[16+1];
	uint64 addr;
	uint64 size;
	uint32 off;
	uint32 align;
	uint32 reloff;
	uint32 nreloc;
	uint32 flags;
	uint32 res1;
	uint32 res2;
	Sym *sym;
	
	MachoRel *rel;
};

struct MachoRel
{
	uint32 addr;
	uint32 symnum;
	uint8 pcrel;
	uint8 length;
	uint8 extrn;
	uint8 type;
	uint8 scattered;
	uint32 value;
};

struct MachoSymtab
{
	uint32 symoff;
	uint32 nsym;
	uint32 stroff;
	uint32 strsize;
	
	char *str;
	MachoSym *sym;
};

struct MachoSym
{
	char *name;
	uint8 type;
	uint8 sectnum;
	uint16 desc;
	char kind;
	uint64 value;
	Sym *sym;
};

struct MachoDysymtab
{
	uint32 ilocalsym;
	uint32 nlocalsym;
	uint32 iextdefsym;
	uint32 nextdefsym;
	uint32 iundefsym;
	uint32 nundefsym;
	uint32 tocoff;
	uint32 ntoc;
	uint32 modtaboff;
	uint32 nmodtab;
	uint32 extrefsymoff;
	uint32 nextrefsyms;
	uint32 indirectsymoff;
	uint32 nindirectsyms;
	uint32 extreloff;
	uint32 nextrel;
	uint32 locreloff;
	uint32 nlocrel;
	uint32 *indir;
};

struct MachoCmd
{
	int type;
	uint32 off;
	uint32 size;
	MachoSeg seg;
	MachoSymtab sym;
	MachoDysymtab dsym;
};

struct MachoObj
{
	Biobuf	*f;
	int64	base;	// off in f where Mach-O begins
	int64	len;		// length of Mach-O
	int is64;
	char	*name;

	Endian	*e;
	uint cputype;
	uint subcputype;
	uint32 filetype;
	uint32 flags;
	MachoCmd *cmd;
	uint ncmd;
};

static int
unpackcmd(uchar *p, MachoObj *m, MachoCmd *c, uint type, uint sz)
{
	uint32 (*e4)(uchar*);
	uint64 (*e8)(uchar*);
	MachoSect *s;
	int i;

	e4 = m->e->e32;
	e8 = m->e->e64;

	c->type = type;
	c->size = sz;
	switch(type){
	default:
		return -1;
	case MachoCmdSegment:
		if(sz < 56)
			return -1;
		strecpy(c->seg.name, c->seg.name+sizeof c->seg.name, (char*)p+8);
		c->seg.vmaddr = e4(p+24);
		c->seg.vmsize = e4(p+28);
		c->seg.fileoff = e4(p+32);
		c->seg.filesz = e4(p+36);
		c->seg.maxprot = e4(p+40);
		c->seg.initprot = e4(p+44);
		c->seg.nsect = e4(p+48);
		c->seg.flags = e4(p+52);
		c->seg.sect = mal(c->seg.nsect * sizeof c->seg.sect[0]);
		if(sz < 56+c->seg.nsect*68)
			return -1;
		p += 56;
		for(i=0; i<c->seg.nsect; i++) {
			s = &c->seg.sect[i];
			strecpy(s->name, s->name+sizeof s->name, (char*)p+0);
			strecpy(s->segname, s->segname+sizeof s->segname, (char*)p+16);
			s->addr = e4(p+32);
			s->size = e4(p+36);
			s->off = e4(p+40);
			s->align = e4(p+44);
			s->reloff = e4(p+48);
			s->nreloc = e4(p+52);
			s->flags = e4(p+56);
			s->res1 = e4(p+60);
			s->res2 = e4(p+64);
			p += 68;
		}
		break;
	case MachoCmdSegment64:
		if(sz < 72)
			return -1;
		strecpy(c->seg.name, c->seg.name+sizeof c->seg.name, (char*)p+8);
		c->seg.vmaddr = e8(p+24);
		c->seg.vmsize = e8(p+32);
		c->seg.fileoff = e8(p+40);
		c->seg.filesz = e8(p+48);
		c->seg.maxprot = e4(p+56);
		c->seg.initprot = e4(p+60);
		c->seg.nsect = e4(p+64);
		c->seg.flags = e4(p+68);
		c->seg.sect = mal(c->seg.nsect * sizeof c->seg.sect[0]);
		if(sz < 72+c->seg.nsect*80)
			return -1;
		p += 72;
		for(i=0; i<c->seg.nsect; i++) {
			s = &c->seg.sect[i];
			strecpy(s->name, s->name+sizeof s->name, (char*)p+0);
			strecpy(s->segname, s->segname+sizeof s->segname, (char*)p+16);
			s->addr = e8(p+32);
			s->size = e8(p+40);
			s->off = e4(p+48);
			s->align = e4(p+52);
			s->reloff = e4(p+56);
			s->nreloc = e4(p+60);
			s->flags = e4(p+64);
			s->res1 = e4(p+68);
			s->res2 = e4(p+72);
			// p+76 is reserved
			p += 80;
		}
		break;
	case MachoCmdSymtab:
		if(sz < 24)
			return -1;
		c->sym.symoff = e4(p+8);
		c->sym.nsym = e4(p+12);
		c->sym.stroff = e4(p+16);
		c->sym.strsize = e4(p+20);
		break;
	case MachoCmdDysymtab:
		if(sz < 80)
			return -1;
		c->dsym.ilocalsym = e4(p+8);
		c->dsym.nlocalsym = e4(p+12);
		c->dsym.iextdefsym = e4(p+16);
		c->dsym.nextdefsym = e4(p+20);
		c->dsym.iundefsym = e4(p+24);
		c->dsym.nundefsym = e4(p+28);
		c->dsym.tocoff = e4(p+32);
		c->dsym.ntoc = e4(p+36);
		c->dsym.modtaboff = e4(p+40);
		c->dsym.nmodtab = e4(p+44);
		c->dsym.extrefsymoff = e4(p+48);
		c->dsym.nextrefsyms = e4(p+52);
		c->dsym.indirectsymoff = e4(p+56);
		c->dsym.nindirectsyms = e4(p+60);
		c->dsym.extreloff = e4(p+64);
		c->dsym.nextrel = e4(p+68);
		c->dsym.locreloff = e4(p+72);
		c->dsym.nlocrel = e4(p+76);
		break;
	}
	return 0;
}

static int
macholoadrel(MachoObj *m, MachoSect *sect)
{
	MachoRel *rel, *r;
	uchar *buf, *p;
	int i, n;
	uint32 v;
	
	if(sect->rel != nil || sect->nreloc == 0)
		return 0;
	rel = mal(sect->nreloc * sizeof r[0]);
	n = sect->nreloc * 8;
	buf = mal(n);
	if(Bseek(m->f, m->base + sect->reloff, 0) < 0 || Bread(m->f, buf, n) != n)
		return -1;
	for(i=0; i<sect->nreloc; i++) {
		r = &rel[i];
		p = buf+i*8;
		r->addr = m->e->e32(p);
		
		// TODO(rsc): Wrong interpretation for big-endian bitfields?
		if(r->addr & 0x80000000) {
			// scatterbrained relocation
			r->scattered = 1;
			v = r->addr >> 24;
			r->addr &= 0xFFFFFF;
			r->type = v & 0xF;
			v >>= 4;
			r->length = 1<<(v&3);
			v >>= 2;
			r->pcrel = v & 1;
			r->value = m->e->e32(p+4);
		} else {
			v = m->e->e32(p+4);
			r->symnum = v & 0xFFFFFF;
			v >>= 24;
			r->pcrel = v&1;
			v >>= 1;
			r->length = 1<<(v&3);
			v >>= 2;
			r->extrn = v&1;
			v >>= 1;
			r->type = v;
		}
	}
	sect->rel = rel;
	return 0;
}

static int
macholoaddsym(MachoObj *m, MachoDysymtab *d)
{
	uchar *p;
	int i, n;
	
	n = d->nindirectsyms;
	
	p = mal(n*4);
	if(Bseek(m->f, m->base + d->indirectsymoff, 0) < 0 || Bread(m->f, p, n*4) != n*4)
		return -1;
	
	d->indir = (uint32*)p;
	for(i=0; i<n; i++)
		d->indir[i] = m->e->e32(p+4*i);
	return 0;
}

static int 
macholoadsym(MachoObj *m, MachoSymtab *symtab)
{
	char *strbuf;
	uchar *symbuf, *p;
	int i, n, symsize;
	MachoSym *sym, *s;
	uint32 v;

	if(symtab->sym != nil)
		return 0;

	strbuf = mal(symtab->strsize);
	if(Bseek(m->f, m->base + symtab->stroff, 0) < 0 || Bread(m->f, strbuf, symtab->strsize) != symtab->strsize)
		return -1;
	
	symsize = 12;
	if(m->is64)
		symsize = 16;
	n = symtab->nsym * symsize;
	symbuf = mal(n);
	if(Bseek(m->f, m->base + symtab->symoff, 0) < 0 || Bread(m->f, symbuf, n) != n)
		return -1;
	sym = mal(symtab->nsym * sizeof sym[0]);
	p = symbuf;
	for(i=0; i<symtab->nsym; i++) {
		s = &sym[i];
		v = m->e->e32(p);
		if(v >= symtab->strsize)
			return -1;
		s->name = strbuf + v;
		s->type = p[4];
		s->sectnum = p[5];
		s->desc = m->e->e16(p+6);
		if(m->is64)
			s->value = m->e->e64(p+8);
		else
			s->value = m->e->e32(p+8);
		p += symsize;
	}
	symtab->str = strbuf;
	symtab->sym = sym;
	return 0;
}

void
ldmacho(Biobuf *f, char *pkg, int64 len, char *pn)
{
	int i, j, is64;
	uint64 secaddr;
	uchar hdr[7*4], *cmdp;
	uchar tmp[4];
	uchar *dat;
	ulong ncmd, cmdsz, ty, sz, off;
	MachoObj *m;
	Endian *e;
	int64 base;
	MachoSect *sect;
	MachoRel *rel;
	Sym *s, *outer;
	MachoCmd *c;
	MachoSymtab *symtab;
	MachoDysymtab *dsymtab;
	MachoSym *sym;
	Reloc *r, *rp;
	char *name;

	version++;
	base = Boffset(f);
	if(Bread(f, hdr, sizeof hdr) != sizeof hdr)
		goto bad;

	if((be.e32(hdr)&~1) == 0xFEEDFACE){
		e = &be;
	}else if((le.e32(hdr)&~1) == 0xFEEDFACE){
		e = &le;
	}else{
		werrstr("bad magic - not mach-o file");
		goto bad;
	}

	is64 = e->e32(hdr) == 0xFEEDFACF;
	ncmd = e->e32(hdr+4*4);
	cmdsz = e->e32(hdr+5*4);
	if(ncmd > 0x10000 || cmdsz >= 0x01000000){
		werrstr("implausible mach-o header ncmd=%lud cmdsz=%lud", ncmd, cmdsz);
		goto bad;
	}
	if(is64)
		Bread(f, tmp, 4);	// skip reserved word in header

	m = mal(sizeof(*m)+ncmd*sizeof(MachoCmd)+cmdsz);
	m->f = f;
	m->e = e;
	m->cputype = e->e32(hdr+1*4);
	m->subcputype = e->e32(hdr+2*4);
	m->filetype = e->e32(hdr+3*4);
	m->ncmd = ncmd;
	m->flags = e->e32(hdr+6*4);
	m->is64 = is64;
	m->base = base;
	m->len = len;
	m->name = pn;
	
	switch(thechar) {
	default:
		diag("%s: mach-o %s unimplemented", pn, thestring);
		return;
	case '6':
		if(e != &le || m->cputype != MachoCpuAmd64) {
			diag("%s: mach-o object but not amd64", pn);
			return;
		}
		break;
	case '8':
		if(e != &le || m->cputype != MachoCpu386) {
			diag("%s: mach-o object but not 386", pn);
			return;
		}
		break;
	}

	m->cmd = (MachoCmd*)(m+1);
	off = sizeof hdr;
	cmdp = (uchar*)(m->cmd+ncmd);
	if(Bread(f, cmdp, cmdsz) != cmdsz){
		werrstr("reading cmds: %r");
		goto bad;
	}

	// read and parse load commands
	c = nil;
	symtab = nil;
	dsymtab = nil;
	for(i=0; i<ncmd; i++){
		ty = e->e32(cmdp);
		sz = e->e32(cmdp+4);
		m->cmd[i].off = off;
		unpackcmd(cmdp, m, &m->cmd[i], ty, sz);
		cmdp += sz;
		off += sz;
		if(ty == MachoCmdSymtab) {
			if(symtab != nil) {
				werrstr("multiple symbol tables");
				goto bad;
			}
			symtab = &m->cmd[i].sym;
			macholoadsym(m, symtab);
		}
		if(ty == MachoCmdDysymtab) {
			dsymtab = &m->cmd[i].dsym;
			macholoaddsym(m, dsymtab);
		}
		if((is64 && ty == MachoCmdSegment64) || (!is64 && ty == MachoCmdSegment)) {
			if(c != nil) {
				werrstr("multiple load commands");
				goto bad;
			}
			c = &m->cmd[i];
		}
	}

	// load text and data segments into memory.
	// they are not as small as the load commands, but we'll need
	// the memory anyway for the symbol images, so we might
	// as well use one large chunk.
	if(c == nil) {
		werrstr("no load command");
		goto bad;
	}
	if(symtab == nil) {
		// our work is done here - no symbols means nothing can refer to this file
		return;
	}

	if(c->seg.fileoff+c->seg.filesz >= len) {
		werrstr("load segment out of range");
		goto bad;
	}

	dat = mal(c->seg.filesz);
	if(Bseek(f, m->base + c->seg.fileoff, 0) < 0 || Bread(f, dat, c->seg.filesz) != c->seg.filesz) {
		werrstr("cannot load object data: %r");
		goto bad;
	}
	
	for(i=0; i<c->seg.nsect; i++) {
		sect = &c->seg.sect[i];
		if(strcmp(sect->segname, "__TEXT") != 0 && strcmp(sect->segname, "__DATA") != 0)
			continue;
		if(strcmp(sect->name, "__eh_frame") == 0)
			continue;
		name = smprint("%s(%s/%s)", pkg, sect->segname, sect->name);
		s = lookup(name, version);
		if(s->type != 0) {
			werrstr("duplicate %s/%s", sect->segname, sect->name);
			goto bad;
		}
		free(name);

		s->np = sect->size;
		s->size = s->np;
		if((sect->flags & 0xff) == 1) // S_ZEROFILL
			s->p = mal(s->size);
		else {
			s->p = dat + sect->addr - c->seg.vmaddr;
		}
		
		if(strcmp(sect->segname, "__TEXT") == 0) {
			if(strcmp(sect->name, "__text") == 0)
				s->type = STEXT;
			else
				s->type = SRODATA;
		} else {
			if (strcmp(sect->name, "__bss") == 0) {
				s->type = SBSS;
				s->np = 0;
			} else
				s->type = SDATA;
		}
		sect->sym = s;
	}
	
	// enter sub-symbols into symbol table.
	// have to guess sizes from next symbol.
	for(i=0; i<symtab->nsym; i++) {
		int v;
		sym = &symtab->sym[i];
		if(sym->type&N_STAB)
			continue;
		// TODO: check sym->type against outer->type.
		name = sym->name;
		if(name[0] == '_' && name[1] != '\0')
			name++;
		v = 0;
		if(!(sym->type&N_EXT))
			v = version;
		s = lookup(name, v);
		sym->sym = s;
		if(sym->sectnum == 0)	// undefined
			continue;
		if(sym->sectnum > c->seg.nsect) {
			werrstr("reference to invalid section %d", sym->sectnum);
			goto bad;
		}
		sect = &c->seg.sect[sym->sectnum-1];
		outer = sect->sym;
		if(outer == nil) {
			werrstr("reference to invalid section %s/%s", sect->segname, sect->name);
			continue;
		}
		if(s->outer != S) {
			if(s->dupok)
				continue;
			diag("%s: duplicate symbol reference: %s in both %s and %s", pn, s->name, s->outer->name, sect->sym->name);
			errorexit();
		}
		s->type = outer->type | SSUB;
		s->sub = outer->sub;
		outer->sub = s;
		s->outer = outer;
		s->value = sym->value - sect->addr;
		if(i+1 < symtab->nsym)
			s->size = (sym+1)->value - sym->value;
		else
			s->size = sect->addr + sect->size - sym->value;
		if(!(s->cgoexport & CgoExportDynamic))
			s->dynimplib = nil;	// satisfy dynimport
		if(outer->type == STEXT) {
			Prog *p;

			if(s->text != P)
				diag("%s sym#%d: duplicate definition of %s", pn, i, s->name);
			// build a TEXT instruction with a unique pc
			// just to make the rest of the linker happy.
			// TODO: this is too 6l-specific ?
			p = prg();
			p->as = ATEXT;
			p->from.type = D_EXTERN;
			p->from.sym = s;
			p->textflag = 7;
			p->to.type = D_CONST;
			p->link = nil;
			p->pc = pc++;
			s->text = p;
		}
		sym->sym = s;
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for(i=0; i<c->seg.nsect; i++) {
		sect = &c->seg.sect[i];
		if((s = sect->sym) == S)
			continue;
		if(s->sub)
			s->sub = listsort(s->sub, valuecmp, offsetof(Sym, sub));
		if(s->type == STEXT) {
			if(etextp)
				etextp->next = s;
			else
				textp = s;
			etextp = s;
			for(s = s->sub; s != S; s = s->sub) {
				etextp->next = s;
				etextp = s;
			}
		}
	}

	// load relocations
	for(i=0; i<c->seg.nsect; i++) {
		sect = &c->seg.sect[i];
		if((s = sect->sym) == S)
			continue;
		macholoadrel(m, sect);
		if(sect->rel == nil)
			continue;
		r = mal(sect->nreloc*sizeof r[0]);
		rp = r;
		rel = sect->rel;
		for(j=0; j<sect->nreloc; j++, rel++) {
			if(rel->scattered) {
				int k;
				MachoSect *ks;

				if(thechar != '8') {
					// mach-o only uses scattered relocation on 32-bit platforms
					diag("unexpected scattered relocation");
					continue;
				}

				// on 386, rewrite scattered 4/1 relocation and some
				// scattered 2/1 relocation into the pseudo-pc-relative
				// reference that it is.
				// assume that the second in the pair is in this section
				// and use that as the pc-relative base.
				if(j+1 >= sect->nreloc) {
					werrstr("unsupported scattered relocation %d", (int)rel->type);
					goto bad;
				}
				if(!(rel+1)->scattered || (rel+1)->type != 1 ||
				   (rel->type != 4 && rel->type != 2) ||
				   (rel+1)->value < sect->addr || (rel+1)->value >= sect->addr+sect->size) {
					werrstr("unsupported scattered relocation %d/%d", (int)rel->type, (int)(rel+1)->type);
					goto bad;
				}

				rp->siz = rel->length;
				rp->off = rel->addr;
				
				// NOTE(rsc): I haven't worked out why (really when)
				// we should ignore the addend on a
				// scattered relocation, but it seems that the
				// common case is we ignore it.
				// It's likely that this is not strictly correct
				// and that the math should look something
				// like the non-scattered case below.
				rp->add = 0;
				
				// want to make it pc-relative aka relative to rp->off+4
				// but the scatter asks for relative to off = (rel+1)->value - sect->addr.
				// adjust rp->add accordingly.
				rp->type = D_PCREL;
				rp->add += (rp->off+4) - ((rel+1)->value - sect->addr);
				
				// now consider the desired symbol.
				// find the section where it lives.
				for(k=0; k<c->seg.nsect; k++) {
					ks = &c->seg.sect[k];
					if(ks->addr <= rel->value && rel->value < ks->addr+ks->size)
						goto foundk;
				}
				werrstr("unsupported scattered relocation: invalid address %#ux", rel->addr);
				goto bad;
			foundk:
				if(ks->sym != S) {
					rp->sym = ks->sym;
					rp->add += rel->value - ks->addr;
				} else if(strcmp(ks->segname, "__IMPORT") == 0 && strcmp(ks->name, "__pointers") == 0) {
					// handle reference to __IMPORT/__pointers.
					// how much worse can this get?
					// why are we supporting 386 on the mac anyway?
					rp->type = 512 + MACHO_FAKE_GOTPCREL;
					// figure out which pointer this is a reference to.
					k = ks->res1 + (rel->value - ks->addr) / 4;
					// load indirect table for __pointers
					// fetch symbol number
					if(dsymtab == nil || k < 0 || k >= dsymtab->nindirectsyms || dsymtab->indir == nil) {
						werrstr("invalid scattered relocation: indirect symbol reference out of range");
						goto bad;
					}
					k = dsymtab->indir[k];
					if(k < 0 || k >= symtab->nsym) {
						werrstr("invalid scattered relocation: symbol reference out of range");
						goto bad;
					}
					rp->sym = symtab->sym[k].sym;
				} else {
					werrstr("unsupported scattered relocation: reference to %s/%s", ks->segname, ks->name);
					goto bad;
				}
				rp++;
				// skip #1 of 2 rel; continue skips #2 of 2.
				rel++;
				j++;
				continue;
			}

			rp->siz = rel->length;
			rp->type = 512 + (rel->type<<1) + rel->pcrel;
			rp->off = rel->addr;

			// Handle X86_64_RELOC_SIGNED referencing a section (rel->extrn == 0).
			if (thechar == '6' && rel->extrn == 0 && rel->type == 1) {
				// Calculate the addend as the offset into the section.
				//
				// The rip-relative offset stored in the object file is encoded
				// as follows:
				//    
				//    movsd	0x00000360(%rip),%xmm0
				//
				// To get the absolute address of the value this rip-relative address is pointing
				// to, we must add the address of the next instruction to it. This is done by
				// taking the address of the relocation and adding 4 to it (since the rip-relative
				// offset can at most be 32 bits long).  To calculate the offset into the section the
				// relocation is referencing, we subtract the vaddr of the start of the referenced
				// section found in the original object file.
				//
				// [For future reference, see Darwin's /usr/include/mach-o/x86_64/reloc.h]
				secaddr = c->seg.sect[rel->symnum-1].addr;
				rp->add = (int32)e->e32(s->p+rp->off) + rp->off + 4 - secaddr;
			} else
				rp->add = (int32)e->e32(s->p+rp->off);

			// For i386 Mach-O PC-relative, the addend is written such that
			// it *is* the PC being subtracted.  Use that to make
			// it match our version of PC-relative.
			if(rel->pcrel && thechar == '8')
				rp->add += rp->off+rp->siz;
			if(!rel->extrn) {
				if(rel->symnum < 1 || rel->symnum > c->seg.nsect) {
					werrstr("invalid relocation: section reference out of range %d vs %d", rel->symnum, c->seg.nsect);
					goto bad;
				}
				rp->sym = c->seg.sect[rel->symnum-1].sym;
				if(rp->sym == nil) {
					werrstr("invalid relocation: %s", c->seg.sect[rel->symnum-1].name);
					goto bad;
				}
				// References to symbols in other sections
				// include that information in the addend.
				// We only care about the delta from the 
				// section base.
				if(thechar == '8')
					rp->add -= c->seg.sect[rel->symnum-1].addr;
			} else {
				if(rel->symnum >= symtab->nsym) {
					werrstr("invalid relocation: symbol reference out of range");
					goto bad;
				}
				rp->sym = symtab->sym[rel->symnum].sym;
			}
			rp++;
		}			
		qsort(r, rp - r, sizeof r[0], rbyoff);
		s->r = r;
		s->nr = rp - r;
	}
	return;

bad:
	diag("%s: malformed mach-o file: %r", pn);
}
