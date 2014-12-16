// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"l.h"
#include	"lib.h"
#include	"../ld/elf.h"

/*
 * We use the 64-bit data structures on both 32- and 64-bit machines
 * in order to write the code just once.  The 64-bit data structure is
 * written in the 32-bit format on the 32-bit machines.
 */
#define	NSECT	48

int	iself;

static	int	elf64;
static	ElfEhdr	hdr;
static	ElfPhdr	*phdr[NSECT];
static	ElfShdr	*shdr[NSECT];
static	char	*interp;

typedef struct Elfstring Elfstring;
struct Elfstring
{
	char *s;
	int off;
};

static Elfstring elfstr[100];
static int nelfstr;

static char buildinfo[32];

/*
 Initialize the global variable that describes the ELF header. It will be updated as
 we write section and prog headers.
 */
void
elfinit(void)
{
	iself = 1;

	switch(thechar) {
	// 64-bit architectures
	case '9':
		if(ctxt->arch->endian == BigEndian)
			hdr.flags = 1;		/* Version 1 ABI */
		else
			hdr.flags = 2;		/* Version 2 ABI */
		// fallthrough
	case '6':
		elf64 = 1;
		hdr.phoff = ELF64HDRSIZE;	/* Must be be ELF64HDRSIZE: first PHdr must follow ELF header */
		hdr.shoff = ELF64HDRSIZE;	/* Will move as we add PHeaders */
		hdr.ehsize = ELF64HDRSIZE;	/* Must be ELF64HDRSIZE */
		hdr.phentsize = ELF64PHDRSIZE;	/* Must be ELF64PHDRSIZE */
		hdr.shentsize = ELF64SHDRSIZE;	/* Must be ELF64SHDRSIZE */
		break;

	// 32-bit architectures
	case '5':
		// we use EABI on both linux/arm and freebsd/arm.
		if(HEADTYPE == Hlinux || HEADTYPE == Hfreebsd)
			hdr.flags = 0x5000002; // has entry point, Version5 EABI
		// fallthrough
	default:
		hdr.phoff = ELF32HDRSIZE;	/* Must be be ELF32HDRSIZE: first PHdr must follow ELF header */
		hdr.shoff = ELF32HDRSIZE;	/* Will move as we add PHeaders */
		hdr.ehsize = ELF32HDRSIZE;	/* Must be ELF32HDRSIZE */
		hdr.phentsize = ELF32PHDRSIZE;	/* Must be ELF32PHDRSIZE */
		hdr.shentsize = ELF32SHDRSIZE;	/* Must be ELF32SHDRSIZE */
	}
}

void
elf64phdr(ElfPhdr *e)
{
	LPUT(e->type);
	LPUT(e->flags);
	VPUT(e->off);
	VPUT(e->vaddr);
	VPUT(e->paddr);
	VPUT(e->filesz);
	VPUT(e->memsz);
	VPUT(e->align);
}

void
elf32phdr(ElfPhdr *e)
{
	int frag;
	
	if(e->type == PT_LOAD) {
		// Correct ELF loaders will do this implicitly,
		// but buggy ELF loaders like the one in some
		// versions of QEMU won't.
		frag = e->vaddr&(e->align-1);
		e->off -= frag;
		e->vaddr -= frag;
		e->paddr -= frag;
		e->filesz += frag;
		e->memsz += frag;
	}
	LPUT(e->type);
	LPUT(e->off);
	LPUT(e->vaddr);
	LPUT(e->paddr);
	LPUT(e->filesz);
	LPUT(e->memsz);
	LPUT(e->flags);
	LPUT(e->align);
}

void
elf64shdr(ElfShdr *e)
{
	LPUT(e->name);
	LPUT(e->type);
	VPUT(e->flags);
	VPUT(e->addr);
	VPUT(e->off);
	VPUT(e->size);
	LPUT(e->link);
	LPUT(e->info);
	VPUT(e->addralign);
	VPUT(e->entsize);
}

void
elf32shdr(ElfShdr *e)
{
	LPUT(e->name);
	LPUT(e->type);
	LPUT(e->flags);
	LPUT(e->addr);
	LPUT(e->off);
	LPUT(e->size);
	LPUT(e->link);
	LPUT(e->info);
	LPUT(e->addralign);
	LPUT(e->entsize);
}

uint32
elfwriteshdrs(void)
{
	int i;

	if (elf64) {
		for (i = 0; i < hdr.shnum; i++)
			elf64shdr(shdr[i]);
		return hdr.shnum * ELF64SHDRSIZE;
	}
	for (i = 0; i < hdr.shnum; i++)
		elf32shdr(shdr[i]);
	return hdr.shnum * ELF32SHDRSIZE;
}

void
elfsetstring(char *s, int off)
{
	if(nelfstr >= nelem(elfstr)) {
		diag("too many elf strings");
		errorexit();
	}
	elfstr[nelfstr].s = s;
	elfstr[nelfstr].off = off;
	nelfstr++;
}

uint32
elfwritephdrs(void)
{
	int i;

	if (elf64) {
		for (i = 0; i < hdr.phnum; i++)
			elf64phdr(phdr[i]);
		return hdr.phnum * ELF64PHDRSIZE;
	}
	for (i = 0; i < hdr.phnum; i++)
		elf32phdr(phdr[i]);
	return hdr.phnum * ELF32PHDRSIZE;
}

ElfPhdr*
newElfPhdr(void)
{
	ElfPhdr *e;

	e = mal(sizeof *e);
	if (hdr.phnum >= NSECT)
		diag("too many phdrs");
	else
		phdr[hdr.phnum++] = e;
	if (elf64)
		hdr.shoff += ELF64PHDRSIZE;
	else
		hdr.shoff += ELF32PHDRSIZE;
	return e;
}

ElfShdr*
newElfShdr(vlong name)
{
	ElfShdr *e;

	e = mal(sizeof *e);
	e->name = name;
	e->shnum = hdr.shnum;
	if (hdr.shnum >= NSECT) {
		diag("too many shdrs");
	} else {
		shdr[hdr.shnum++] = e;
	}
	return e;
}

ElfEhdr*
getElfEhdr(void)
{
	return &hdr;
}

uint32
elf64writehdr(void)
{
	int i;

	for (i = 0; i < EI_NIDENT; i++)
		cput(hdr.ident[i]);
	WPUT(hdr.type);
	WPUT(hdr.machine);
	LPUT(hdr.version);
	VPUT(hdr.entry);
	VPUT(hdr.phoff);
	VPUT(hdr.shoff);
	LPUT(hdr.flags);
	WPUT(hdr.ehsize);
	WPUT(hdr.phentsize);
	WPUT(hdr.phnum);
	WPUT(hdr.shentsize);
	WPUT(hdr.shnum);
	WPUT(hdr.shstrndx);
	return ELF64HDRSIZE;
}

uint32
elf32writehdr(void)
{
	int i;

	for (i = 0; i < EI_NIDENT; i++)
		cput(hdr.ident[i]);
	WPUT(hdr.type);
	WPUT(hdr.machine);
	LPUT(hdr.version);
	LPUT(hdr.entry);
	LPUT(hdr.phoff);
	LPUT(hdr.shoff);
	LPUT(hdr.flags);
	WPUT(hdr.ehsize);
	WPUT(hdr.phentsize);
	WPUT(hdr.phnum);
	WPUT(hdr.shentsize);
	WPUT(hdr.shnum);
	WPUT(hdr.shstrndx);
	return ELF32HDRSIZE;
}

uint32
elfwritehdr(void)
{
	if(elf64)
		return elf64writehdr();
	return elf32writehdr();
}

/* Taken directly from the definition document for ELF64 */
uint32
elfhash(uchar *name)
{
	uint32 h = 0, g;
	while (*name) {
		h = (h << 4) + *name++;
		if (g = h & 0xf0000000)
			h ^= g >> 24;
		h &= 0x0fffffff;
	}
	return h;
}

void
elfwritedynent(LSym *s, int tag, uint64 val)
{
	if(elf64) {
		adduint64(ctxt, s, tag);
		adduint64(ctxt, s, val);
	} else {
		adduint32(ctxt, s, tag);
		adduint32(ctxt, s, val);
	}
}

void
elfwritedynentsym(LSym *s, int tag, LSym *t)
{
	elfwritedynentsymplus(s, tag, t, 0);
}

void
elfwritedynentsymplus(LSym *s, int tag, LSym *t, vlong add)
{
	if(elf64)
		adduint64(ctxt, s, tag);
	else
		adduint32(ctxt, s, tag);
	addaddrplus(ctxt, s, t, add);
}

void
elfwritedynentsymsize(LSym *s, int tag, LSym *t)
{
	if(elf64)
		adduint64(ctxt, s, tag);
	else
		adduint32(ctxt, s, tag);
	addsize(ctxt, s, t);
}

int
elfinterp(ElfShdr *sh, uint64 startva, uint64 resoff, char *p)
{
	int n;

	interp = p;
	n = strlen(interp)+1;
	sh->addr = startva + resoff - n;
	sh->off = resoff - n;
	sh->size = n;

	return n;
}

int
elfwriteinterp(void)
{
	ElfShdr *sh;
	
	sh = elfshname(".interp");
	cseek(sh->off);
	cwrite(interp, sh->size);
	return sh->size;
}

int
elfnote(ElfShdr *sh, uint64 startva, uint64 resoff, int sz)
{
	uint64 n;

	n = sizeof(Elf_Note) + sz + resoff % 4;

	sh->type = SHT_NOTE;
	sh->flags = SHF_ALLOC;
	sh->addralign = 4;
	sh->addr = startva + resoff - n;
	sh->off = resoff - n;
	sh->size = n - resoff % 4;

	return n;
}

ElfShdr *
elfwritenotehdr(char *str, uint32 namesz, uint32 descsz, uint32 tag)
{
	ElfShdr *sh;
	
	sh = elfshname(str);

	// Write Elf_Note header.
	cseek(sh->off);
	LPUT(namesz);
	LPUT(descsz);
	LPUT(tag);

	return sh;
}

// NetBSD Signature (as per sys/exec_elf.h)
#define ELF_NOTE_NETBSD_NAMESZ		7
#define ELF_NOTE_NETBSD_DESCSZ		4
#define ELF_NOTE_NETBSD_TAG		1
#define ELF_NOTE_NETBSD_NAME		"NetBSD\0\0"
#define ELF_NOTE_NETBSD_VERSION		599000000	/* NetBSD 5.99 */

int
elfnetbsdsig(ElfShdr *sh, uint64 startva, uint64 resoff)
{
	int n;

	n = rnd(ELF_NOTE_NETBSD_NAMESZ, 4) + rnd(ELF_NOTE_NETBSD_DESCSZ, 4);
	return elfnote(sh, startva, resoff, n);
}

int
elfwritenetbsdsig(void)
{
	ElfShdr *sh;

	// Write Elf_Note header.
	sh = elfwritenotehdr(".note.netbsd.ident", ELF_NOTE_NETBSD_NAMESZ, ELF_NOTE_NETBSD_DESCSZ, ELF_NOTE_NETBSD_TAG);
	if(sh == nil)
		return 0;

	// Followed by NetBSD string and version.
	cwrite(ELF_NOTE_NETBSD_NAME, ELF_NOTE_NETBSD_NAMESZ + 1);
	LPUT(ELF_NOTE_NETBSD_VERSION);

	return sh->size;
}

// OpenBSD Signature
#define ELF_NOTE_OPENBSD_NAMESZ		8
#define ELF_NOTE_OPENBSD_DESCSZ		4
#define ELF_NOTE_OPENBSD_TAG		1
#define ELF_NOTE_OPENBSD_NAME		"OpenBSD\0"
#define ELF_NOTE_OPENBSD_VERSION	0

int
elfopenbsdsig(ElfShdr *sh, uint64 startva, uint64 resoff)
{
	int n;

	n = ELF_NOTE_OPENBSD_NAMESZ + ELF_NOTE_OPENBSD_DESCSZ;
	return elfnote(sh, startva, resoff, n);
}

int
elfwriteopenbsdsig(void)
{
	ElfShdr *sh;

	// Write Elf_Note header.
	sh = elfwritenotehdr(".note.openbsd.ident", ELF_NOTE_OPENBSD_NAMESZ, ELF_NOTE_OPENBSD_DESCSZ, ELF_NOTE_OPENBSD_TAG);
	if(sh == nil)
		return 0;

	// Followed by OpenBSD string and version.
	cwrite(ELF_NOTE_OPENBSD_NAME, ELF_NOTE_OPENBSD_NAMESZ);
	LPUT(ELF_NOTE_OPENBSD_VERSION);

	return sh->size;
}

void
addbuildinfo(char *val)
{
	char *ov;
	int i, b, j;

	if(val[0] != '0' || val[1] != 'x') {
		fprint(2, "%s: -B argument must start with 0x: %s\n", argv0, val);
		exits("usage");
	}
	ov = val;
	val += 2;
	i = 0;
	while(*val != '\0') {
		if(val[1] == '\0') {
			fprint(2, "%s: -B argument must have even number of digits: %s\n", argv0, ov);
			exits("usage");
		}
		b = 0;
		for(j = 0; j < 2; j++, val++) {
			b *= 16;
		  	if(*val >= '0' && *val <= '9')
				b += *val - '0';
			else if(*val >= 'a' && *val <= 'f')
				b += *val - 'a' + 10;
			else if(*val >= 'A' && *val <= 'F')
				b += *val - 'A' + 10;
			else {
				fprint(2, "%s: -B argument contains invalid hex digit %c: %s\n", argv0, *val, ov);
				exits("usage");
			}
		}
		if(i >= nelem(buildinfo)) {
			fprint(2, "%s: -B option too long (max %d digits): %s\n", argv0, (int)nelem(buildinfo), ov);
			exits("usage");
		}
		buildinfo[i++] = b;
	}
	buildinfolen = i;
}

// Build info note
#define ELF_NOTE_BUILDINFO_NAMESZ	4
#define ELF_NOTE_BUILDINFO_TAG		3
#define ELF_NOTE_BUILDINFO_NAME		"GNU\0"

int
elfbuildinfo(ElfShdr *sh, uint64 startva, uint64 resoff)
{
	int n;

	n = ELF_NOTE_BUILDINFO_NAMESZ + rnd(buildinfolen, 4);
	return elfnote(sh, startva, resoff, n);
}

int
elfwritebuildinfo(void)
{
	ElfShdr *sh;

	sh = elfwritenotehdr(".note.gnu.build-id", ELF_NOTE_BUILDINFO_NAMESZ, buildinfolen, ELF_NOTE_BUILDINFO_TAG);
	if(sh == nil)
		return 0;

	cwrite(ELF_NOTE_BUILDINFO_NAME, ELF_NOTE_BUILDINFO_NAMESZ);
	cwrite(buildinfo, buildinfolen);
	cwrite("\0\0\0", rnd(buildinfolen, 4) - buildinfolen);

	return sh->size;
}

extern int nelfsym;
int elfverneed;

typedef struct Elfaux Elfaux;
typedef struct Elflib Elflib;

struct Elflib
{
	Elflib *next;
	Elfaux *aux;
	char *file;
};

struct Elfaux
{
	Elfaux *next;
	int num;
	char *vers;
};

Elfaux*
addelflib(Elflib **list, char *file, char *vers)
{
	Elflib *lib;
	Elfaux *aux;
	
	for(lib=*list; lib; lib=lib->next)
		if(strcmp(lib->file, file) == 0)
			goto havelib;
	lib = mal(sizeof *lib);
	lib->next = *list;
	lib->file = file;
	*list = lib;
havelib:
	for(aux=lib->aux; aux; aux=aux->next)
		if(strcmp(aux->vers, vers) == 0)
			goto haveaux;
	aux = mal(sizeof *aux);
	aux->next = lib->aux;
	aux->vers = vers;
	lib->aux = aux;
haveaux:
	return aux;
}

void
elfdynhash(void)
{
	LSym *s, *sy, *dynstr;
	int i, j, nbucket, b, nfile;
	uint32 hc, *chain, *buckets;
	int nsym;
	char *name;
	Elfaux **need;
	Elflib *needlib;
	Elflib *l;
	Elfaux *x;
	
	if(!iself)
		return;

	nsym = nelfsym;
	s = linklookup(ctxt, ".hash", 0);
	s->type = SELFROSECT;
	s->reachable = 1;

	i = nsym;
	nbucket = 1;
	while(i > 0) {
		++nbucket;
		i >>= 1;
	}

	needlib = nil;
	need = malloc(nsym * sizeof need[0]);
	chain = malloc(nsym * sizeof chain[0]);
	buckets = malloc(nbucket * sizeof buckets[0]);
	if(need == nil || chain == nil || buckets == nil) {
		ctxt->cursym = nil;
		diag("out of memory");
		errorexit();
	}
	memset(need, 0, nsym * sizeof need[0]);
	memset(chain, 0, nsym * sizeof chain[0]);
	memset(buckets, 0, nbucket * sizeof buckets[0]);
	for(sy=ctxt->allsym; sy!=S; sy=sy->allsym) {
		if (sy->dynid <= 0)
			continue;

		if(sy->dynimpvers)
			need[sy->dynid] = addelflib(&needlib, sy->dynimplib, sy->dynimpvers);

		name = sy->extname;
		hc = elfhash((uchar*)name);

		b = hc % nbucket;
		chain[sy->dynid] = buckets[b];
		buckets[b] = sy->dynid;
	}

	adduint32(ctxt, s, nbucket);
	adduint32(ctxt, s, nsym);
	for(i = 0; i<nbucket; i++)
		adduint32(ctxt, s, buckets[i]);
	for(i = 0; i<nsym; i++)
		adduint32(ctxt, s, chain[i]);

	free(chain);
	free(buckets);
	
	// version symbols
	dynstr = linklookup(ctxt, ".dynstr", 0);
	s = linklookup(ctxt, ".gnu.version_r", 0);
	i = 2;
	nfile = 0;
	for(l=needlib; l; l=l->next) {
		nfile++;
		// header
		adduint16(ctxt, s, 1);  // table version
		j = 0;
		for(x=l->aux; x; x=x->next)
			j++;
		adduint16(ctxt, s, j);	// aux count
		adduint32(ctxt, s, addstring(dynstr, l->file));  // file string offset
		adduint32(ctxt, s, 16);  // offset from header to first aux
		if(l->next)
			adduint32(ctxt, s, 16+j*16);  // offset from this header to next
		else
			adduint32(ctxt, s, 0);
		
		for(x=l->aux; x; x=x->next) {
			x->num = i++;
			// aux struct
			adduint32(ctxt, s, elfhash((uchar*)x->vers));  // hash
			adduint16(ctxt, s, 0);  // flags
			adduint16(ctxt, s, x->num);  // other - index we refer to this by
			adduint32(ctxt, s, addstring(dynstr, x->vers));  // version string offset
			if(x->next)
				adduint32(ctxt, s, 16);  // offset from this aux to next
			else
				adduint32(ctxt, s, 0);
		}
	}

	// version references
	s = linklookup(ctxt, ".gnu.version", 0);
	for(i=0; i<nsym; i++) {
		if(i == 0)
			adduint16(ctxt, s, 0); // first entry - no symbol
		else if(need[i] == nil)
			adduint16(ctxt, s, 1); // global
		else
			adduint16(ctxt, s, need[i]->num);
	}

	free(need);

	s = linklookup(ctxt, ".dynamic", 0);
	elfverneed = nfile;
	if(elfverneed) {
		elfwritedynentsym(s, DT_VERNEED, linklookup(ctxt, ".gnu.version_r", 0));
		elfwritedynent(s, DT_VERNEEDNUM, nfile);
		elfwritedynentsym(s, DT_VERSYM, linklookup(ctxt, ".gnu.version", 0));
	}

	if(thechar == '6' || thechar == '9') {
		sy = linklookup(ctxt, ".rela.plt", 0);
		if(sy->size > 0) {
			elfwritedynent(s, DT_PLTREL, DT_RELA);
			elfwritedynentsymsize(s, DT_PLTRELSZ, sy);
			elfwritedynentsym(s, DT_JMPREL, sy);
		}
	} else {
		sy = linklookup(ctxt, ".rel.plt", 0);
		if(sy->size > 0) {
			elfwritedynent(s, DT_PLTREL, DT_REL);
			elfwritedynentsymsize(s, DT_PLTRELSZ, sy);
			elfwritedynentsym(s, DT_JMPREL, sy);
		}
	}

	elfwritedynent(s, DT_NULL, 0);
}

ElfPhdr*
elfphload(Segment *seg)
{
	ElfPhdr *ph;
	
	ph = newElfPhdr();
	ph->type = PT_LOAD;
	if(seg->rwx & 4)
		ph->flags |= PF_R;
	if(seg->rwx & 2)
		ph->flags |= PF_W;
	if(seg->rwx & 1)
		ph->flags |= PF_X;
	ph->vaddr = seg->vaddr;
	ph->paddr = seg->vaddr;
	ph->memsz = seg->len;
	ph->off = seg->fileoff;
	ph->filesz = seg->filelen;
	ph->align = INITRND;
	
	return ph;
}

ElfShdr*
elfshname(char *name)
{
	int i, off;
	ElfShdr *sh;
	
	for(i=0; i<nelfstr; i++) {
		if(strcmp(name, elfstr[i].s) == 0) {
			off = elfstr[i].off;
			goto found;
		}
	}
	diag("cannot find elf name %s", name);
	errorexit();
	return nil;

found:
	for(i=0; i<hdr.shnum; i++) {
		sh = shdr[i];
		if(sh->name == off)
			return sh;
	}
	
	sh = newElfShdr(off);
	return sh;
}

ElfShdr*
elfshalloc(Section *sect)
{
	ElfShdr *sh;
	
	sh = elfshname(sect->name);
	sect->elfsect = sh;
	return sh;
}

ElfShdr*
elfshbits(Section *sect)
{
	ElfShdr *sh;
	
	sh = elfshalloc(sect);
	if(sh->type > 0)
		return sh;

	if(sect->vaddr < sect->seg->vaddr + sect->seg->filelen)
		sh->type = SHT_PROGBITS;
	else
		sh->type = SHT_NOBITS;
	sh->flags = SHF_ALLOC;
	if(sect->rwx & 1)
		sh->flags |= SHF_EXECINSTR;
	if(sect->rwx & 2)
		sh->flags |= SHF_WRITE;
	if(strcmp(sect->name, ".tbss") == 0) {
		if(strcmp(goos, "android") != 0)
			sh->flags |= SHF_TLS; // no TLS on android
		sh->type = SHT_NOBITS;
	}
	if(linkmode != LinkExternal)
		sh->addr = sect->vaddr;
	sh->addralign = sect->align;
	sh->size = sect->len;
	sh->off = sect->seg->fileoff + sect->vaddr - sect->seg->vaddr;

	return sh;
}

ElfShdr*
elfshreloc(Section *sect)
{
	int typ;
	ElfShdr *sh;
	char *prefix;
	char buf[100];
	
	// If main section is SHT_NOBITS, nothing to relocate.
	// Also nothing to relocate in .shstrtab.
	if(sect->vaddr >= sect->seg->vaddr + sect->seg->filelen)
		return nil;
	if(strcmp(sect->name, ".shstrtab") == 0 || strcmp(sect->name, ".tbss") == 0)
		return nil;

	if(thechar == '6' || thechar == '9') {
		prefix = ".rela";
		typ = SHT_RELA;
	} else {
		prefix = ".rel";
		typ = SHT_REL;
	}

	snprint(buf, sizeof buf, "%s%s", prefix, sect->name);
	sh = elfshname(buf);
	sh->type = typ;
	sh->entsize = RegSize*(2+(typ==SHT_RELA));
	sh->link = elfshname(".symtab")->shnum;
	sh->info = sect->elfsect->shnum;
	sh->off = sect->reloff;
	sh->size = sect->rellen;
	sh->addralign = RegSize;
	return sh;
}

void
elfrelocsect(Section *sect, LSym *first)
{
	LSym *sym;
	int32 eaddr;
	Reloc *r;

	// If main section is SHT_NOBITS, nothing to relocate.
	// Also nothing to relocate in .shstrtab.
	if(sect->vaddr >= sect->seg->vaddr + sect->seg->filelen)
		return;
	if(strcmp(sect->name, ".shstrtab") == 0)
		return;

	sect->reloff = cpos();
	for(sym = first; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= sect->vaddr)
			break;
	}
	
	eaddr = sect->vaddr + sect->len;
	for(; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= eaddr)
			break;
		ctxt->cursym = sym;
		
		for(r = sym->r; r < sym->r+sym->nr; r++) {
			if(r->done)
				continue;
			if(r->xsym == nil) {
				diag("missing xsym in relocation");
				continue;
			}
			if(r->xsym->elfsym == 0)
				diag("reloc %d to non-elf symbol %s (outer=%s) %d", r->type, r->sym->name, r->xsym->name, r->sym->type);
			if(elfreloc1(r, sym->value+r->off - sect->vaddr) < 0)
				diag("unsupported obj reloc %d/%d to %s", r->type, r->siz, r->sym->name);
		}
	}
		
	sect->rellen = cpos() - sect->reloff;
}	
	
void
elfemitreloc(void)
{
	Section *sect;

	while(cpos()&7)
		cput(0);

	elfrelocsect(segtext.sect, ctxt->textp);
	for(sect=segtext.sect->next; sect!=nil; sect=sect->next)
		elfrelocsect(sect, datap);	
	for(sect=segrodata.sect; sect!=nil; sect=sect->next)
		elfrelocsect(sect, datap);	
	for(sect=segdata.sect; sect!=nil; sect=sect->next)
		elfrelocsect(sect, datap);	
}

void
doelf(void)
{
	LSym *s, *shstrtab, *dynstr;

	if(!iself)
		return;

	/* predefine strings we need for section headers */
	shstrtab = linklookup(ctxt, ".shstrtab", 0);
	shstrtab->type = SELFROSECT;
	shstrtab->reachable = 1;

	addstring(shstrtab, "");
	addstring(shstrtab, ".text");
	addstring(shstrtab, ".noptrdata");
	addstring(shstrtab, ".data");
	addstring(shstrtab, ".bss");
	addstring(shstrtab, ".noptrbss");
	// generate .tbss section (except for OpenBSD where it's not supported)
	// for dynamic internal linker or external linking, so that various
	// binutils could correctly calculate PT_TLS size.
	// see http://golang.org/issue/5200.
	if(HEADTYPE != Hopenbsd)
	if(!debug['d'] || linkmode == LinkExternal)
		addstring(shstrtab, ".tbss");
	if(HEADTYPE == Hnetbsd)
		addstring(shstrtab, ".note.netbsd.ident");
	if(HEADTYPE == Hopenbsd)
		addstring(shstrtab, ".note.openbsd.ident");
	if(buildinfolen > 0)
		addstring(shstrtab, ".note.gnu.build-id");
	addstring(shstrtab, ".elfdata");
	addstring(shstrtab, ".rodata");
	addstring(shstrtab, ".typelink");
	addstring(shstrtab, ".gosymtab");
	addstring(shstrtab, ".gopclntab");
	
	if(linkmode == LinkExternal) {
		debug_s = debug['s'];
		debug['s'] = 0;
		debug['d'] = 1;

		if(thechar == '6' || thechar == '9') {
			addstring(shstrtab, ".rela.text");
			addstring(shstrtab, ".rela.rodata");
			addstring(shstrtab, ".rela.typelink");
			addstring(shstrtab, ".rela.gosymtab");
			addstring(shstrtab, ".rela.gopclntab");
			addstring(shstrtab, ".rela.noptrdata");
			addstring(shstrtab, ".rela.data");
		} else {
			addstring(shstrtab, ".rel.text");
			addstring(shstrtab, ".rel.rodata");
			addstring(shstrtab, ".rel.typelink");
			addstring(shstrtab, ".rel.gosymtab");
			addstring(shstrtab, ".rel.gopclntab");
			addstring(shstrtab, ".rel.noptrdata");
			addstring(shstrtab, ".rel.data");
		}
		// add a .note.GNU-stack section to mark the stack as non-executable
		addstring(shstrtab, ".note.GNU-stack");
	}

	if(flag_shared) {
		addstring(shstrtab, ".init_array");
		if(thechar == '6' || thechar == '9')
			addstring(shstrtab, ".rela.init_array");
		else
			addstring(shstrtab, ".rel.init_array");
	}

	if(!debug['s']) {
		addstring(shstrtab, ".symtab");
		addstring(shstrtab, ".strtab");
		dwarfaddshstrings(shstrtab);
	}
	addstring(shstrtab, ".shstrtab");

	if(!debug['d']) {	/* -d suppresses dynamic loader format */
		addstring(shstrtab, ".interp");
		addstring(shstrtab, ".hash");
		addstring(shstrtab, ".got");
		if(thechar == '9')
			addstring(shstrtab, ".glink");
		addstring(shstrtab, ".got.plt");
		addstring(shstrtab, ".dynamic");
		addstring(shstrtab, ".dynsym");
		addstring(shstrtab, ".dynstr");
		if(thechar == '6' || thechar == '9') {
			addstring(shstrtab, ".rela");
			addstring(shstrtab, ".rela.plt");
		} else {
			addstring(shstrtab, ".rel");
			addstring(shstrtab, ".rel.plt");
		}
		addstring(shstrtab, ".plt");
		addstring(shstrtab, ".gnu.version");
		addstring(shstrtab, ".gnu.version_r");

		/* dynamic symbol table - first entry all zeros */
		s = linklookup(ctxt, ".dynsym", 0);
		s->type = SELFROSECT;
		s->reachable = 1;
		if(thechar == '6' || thechar == '9')
			s->size += ELF64SYMSIZE;
		else
			s->size += ELF32SYMSIZE;

		/* dynamic string table */
		s = linklookup(ctxt, ".dynstr", 0);
		s->type = SELFROSECT;
		s->reachable = 1;
		if(s->size == 0)
			addstring(s, "");
		dynstr = s;

		/* relocation table */
		if(thechar == '6' || thechar == '9')
			s = linklookup(ctxt, ".rela", 0);
		else
			s = linklookup(ctxt, ".rel", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		/* global offset table */
		s = linklookup(ctxt, ".got", 0);
		s->reachable = 1;
		s->type = SELFGOT; // writable

		/* ppc64 glink resolver */
		if(thechar == '9') {
			s = linklookup(ctxt, ".glink", 0);
			s->reachable = 1;
			s->type = SELFRXSECT;
		}

		/* hash */
		s = linklookup(ctxt, ".hash", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		s = linklookup(ctxt, ".got.plt", 0);
		s->reachable = 1;
		s->type = SELFSECT; // writable

		s = linklookup(ctxt, ".plt", 0);
		s->reachable = 1;
		if(thechar == '9')
			// In the ppc64 ABI, .plt is a data section
			// written by the dynamic linker.
			s->type = SELFSECT;
		else
			s->type = SELFRXSECT;
		
		elfsetupplt();
		
		if(thechar == '6' || thechar == '9')
			s = linklookup(ctxt, ".rela.plt", 0);
		else
			s = linklookup(ctxt, ".rel.plt", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		
		s = linklookup(ctxt, ".gnu.version", 0);
		s->reachable = 1;
		s->type = SELFROSECT;
		
		s = linklookup(ctxt, ".gnu.version_r", 0);
		s->reachable = 1;
		s->type = SELFROSECT;

		/* define dynamic elf table */
		s = linklookup(ctxt, ".dynamic", 0);
		s->reachable = 1;
		s->type = SELFSECT; // writable

		/*
		 * .dynamic table
		 */
		elfwritedynentsym(s, DT_HASH, linklookup(ctxt, ".hash", 0));
		elfwritedynentsym(s, DT_SYMTAB, linklookup(ctxt, ".dynsym", 0));
		if(thechar == '6' || thechar == '9')
			elfwritedynent(s, DT_SYMENT, ELF64SYMSIZE);
		else
			elfwritedynent(s, DT_SYMENT, ELF32SYMSIZE);
		elfwritedynentsym(s, DT_STRTAB, linklookup(ctxt, ".dynstr", 0));
		elfwritedynentsymsize(s, DT_STRSZ, linklookup(ctxt, ".dynstr", 0));
		if(thechar == '6' || thechar == '9') {
			elfwritedynentsym(s, DT_RELA, linklookup(ctxt, ".rela", 0));
			elfwritedynentsymsize(s, DT_RELASZ, linklookup(ctxt, ".rela", 0));
			elfwritedynent(s, DT_RELAENT, ELF64RELASIZE);
		} else {
			elfwritedynentsym(s, DT_REL, linklookup(ctxt, ".rel", 0));
			elfwritedynentsymsize(s, DT_RELSZ, linklookup(ctxt, ".rel", 0));
			elfwritedynent(s, DT_RELENT, ELF32RELSIZE);
		}
		if(rpath)
			elfwritedynent(s, DT_RUNPATH, addstring(dynstr, rpath));

		if(thechar == '9')
			elfwritedynentsym(s, DT_PLTGOT, linklookup(ctxt, ".plt", 0));
		else
			elfwritedynentsym(s, DT_PLTGOT, linklookup(ctxt, ".got.plt", 0));

		if(thechar == '9')
			elfwritedynent(s, DT_PPC64_OPT, 0);

		// Solaris dynamic linker can't handle an empty .rela.plt if
		// DT_JMPREL is emitted so we have to defer generation of DT_PLTREL,
		// DT_PLTRELSZ, and DT_JMPREL dynamic entries until after we know the
		// size of .rel(a).plt section.
		elfwritedynent(s, DT_DEBUG, 0);

		// Do not write DT_NULL.  elfdynhash will finish it.
	}
}

void
shsym(ElfShdr *sh, LSym *s)
{
	vlong addr;
	addr = symaddr(s);
	if(sh->flags&SHF_ALLOC)
		sh->addr = addr;
	sh->off = datoff(addr);
	sh->size = s->size;
}

void
phsh(ElfPhdr *ph, ElfShdr *sh)
{
	ph->vaddr = sh->addr;
	ph->paddr = ph->vaddr;
	ph->off = sh->off;
	ph->filesz = sh->size;
	ph->memsz = sh->size;
	ph->align = sh->addralign;
}

void
asmbelfsetup(void)
{
	Section *sect;

	/* This null SHdr must appear before all others */
	elfshname("");
	
	for(sect=segtext.sect; sect!=nil; sect=sect->next)
		elfshalloc(sect);
	for(sect=segrodata.sect; sect!=nil; sect=sect->next)
		elfshalloc(sect);
	for(sect=segdata.sect; sect!=nil; sect=sect->next)
		elfshalloc(sect);
}

void
asmbelf(vlong symo)
{
	vlong a, o;
	vlong startva, resoff;
	ElfEhdr *eh;
	ElfPhdr *ph, *pph, *pnote;
	ElfShdr *sh;
	Section *sect;

	eh = getElfEhdr();
	switch(thechar) {
	default:
		diag("unknown architecture in asmbelf");
		errorexit();
	case '5':
		eh->machine = EM_ARM;
		break;
	case '6':
		eh->machine = EM_X86_64;
		break;
	case '8':
		eh->machine = EM_386;
		break;
	case '9':
		eh->machine = EM_PPC64;
		break;
	}

	startva = INITTEXT - HEADR;
	resoff = ELFRESERVE;
	
	pph = nil;
	if(linkmode == LinkExternal) {
		/* skip program headers */
		eh->phoff = 0;
		eh->phentsize = 0;
		goto elfobj;
	}

	/* program header info */
	pph = newElfPhdr();
	pph->type = PT_PHDR;
	pph->flags = PF_R;
	pph->off = eh->ehsize;
	pph->vaddr = INITTEXT - HEADR + pph->off;
	pph->paddr = INITTEXT - HEADR + pph->off;
	pph->align = INITRND;

	/*
	 * PHDR must be in a loaded segment. Adjust the text
	 * segment boundaries downwards to include it.
	 * Except on NaCl where it must not be loaded.
	 */
	if(HEADTYPE != Hnacl) {
		o = segtext.vaddr - pph->vaddr;
		segtext.vaddr -= o;
		segtext.len += o;
		o = segtext.fileoff - pph->off;
		segtext.fileoff -= o;
		segtext.filelen += o;
	}

	if(!debug['d']) {
		/* interpreter */
		sh = elfshname(".interp");
		sh->type = SHT_PROGBITS;
		sh->flags = SHF_ALLOC;
		sh->addralign = 1;
		if(interpreter == nil) {
			switch(HEADTYPE) {
			case Hlinux:
				interpreter = linuxdynld;
				break;
			case Hfreebsd:
				interpreter = freebsddynld;
				break;
			case Hnetbsd:
				interpreter = netbsddynld;
				break;
			case Hopenbsd:
				interpreter = openbsddynld;
				break;
			case Hdragonfly:
				interpreter = dragonflydynld;
				break;
			case Hsolaris:
				interpreter = solarisdynld;
				break;
			}
		}
		resoff -= elfinterp(sh, startva, resoff, interpreter);

		ph = newElfPhdr();
		ph->type = PT_INTERP;
		ph->flags = PF_R;
		phsh(ph, sh);
	}

	pnote = nil;
	if(HEADTYPE == Hnetbsd || HEADTYPE == Hopenbsd) {
		sh = nil;
		switch(HEADTYPE) {
		case Hnetbsd:
			sh = elfshname(".note.netbsd.ident");
			resoff -= elfnetbsdsig(sh, startva, resoff);
			break;
		case Hopenbsd:
			sh = elfshname(".note.openbsd.ident");
			resoff -= elfopenbsdsig(sh, startva, resoff);
			break;
		}

		pnote = newElfPhdr();
		pnote->type = PT_NOTE;
		pnote->flags = PF_R;
		phsh(pnote, sh);
	}

	if(buildinfolen > 0) {
		sh = elfshname(".note.gnu.build-id");
		resoff -= elfbuildinfo(sh, startva, resoff);

		if(pnote == nil) {
			pnote = newElfPhdr();
			pnote->type = PT_NOTE;
			pnote->flags = PF_R;
		}
		phsh(pnote, sh);
	}

	// Additions to the reserved area must be above this line.
	USED(resoff);

	elfphload(&segtext);
	if(segrodata.sect != nil)
		elfphload(&segrodata);
	elfphload(&segdata);

	/* Dynamic linking sections */
	if(!debug['d']) {	/* -d suppresses dynamic loader format */
		sh = elfshname(".dynsym");
		sh->type = SHT_DYNSYM;
		sh->flags = SHF_ALLOC;
		if(elf64)
			sh->entsize = ELF64SYMSIZE;
		else
			sh->entsize = ELF32SYMSIZE;
		sh->addralign = RegSize;
		sh->link = elfshname(".dynstr")->shnum;
		// sh->info = index of first non-local symbol (number of local symbols)
		shsym(sh, linklookup(ctxt, ".dynsym", 0));

		sh = elfshname(".dynstr");
		sh->type = SHT_STRTAB;
		sh->flags = SHF_ALLOC;
		sh->addralign = 1;
		shsym(sh, linklookup(ctxt, ".dynstr", 0));

		if(elfverneed) {
			sh = elfshname(".gnu.version");
			sh->type = SHT_GNU_VERSYM;
			sh->flags = SHF_ALLOC;
			sh->addralign = 2;
			sh->link = elfshname(".dynsym")->shnum;
			sh->entsize = 2;
			shsym(sh, linklookup(ctxt, ".gnu.version", 0));
			
			sh = elfshname(".gnu.version_r");
			sh->type = SHT_GNU_VERNEED;
			sh->flags = SHF_ALLOC;
			sh->addralign = RegSize;
			sh->info = elfverneed;
			sh->link = elfshname(".dynstr")->shnum;
			shsym(sh, linklookup(ctxt, ".gnu.version_r", 0));
		}

		switch(eh->machine) {
		case EM_X86_64:
		case EM_PPC64:
			sh = elfshname(".rela.plt");
			sh->type = SHT_RELA;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF64RELASIZE;
			sh->addralign = RegSize;
			sh->link = elfshname(".dynsym")->shnum;
			sh->info = elfshname(".plt")->shnum;
			shsym(sh, linklookup(ctxt, ".rela.plt", 0));

			sh = elfshname(".rela");
			sh->type = SHT_RELA;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF64RELASIZE;
			sh->addralign = 8;
			sh->link = elfshname(".dynsym")->shnum;
			shsym(sh, linklookup(ctxt, ".rela", 0));
			break;
		
		default:
			sh = elfshname(".rel.plt");
			sh->type = SHT_REL;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF32RELSIZE;
			sh->link = elfshname(".dynsym")->shnum;
			shsym(sh, linklookup(ctxt, ".rel.plt", 0));

			sh = elfshname(".rel");
			sh->type = SHT_REL;
			sh->flags = SHF_ALLOC;
			sh->entsize = ELF32RELSIZE;
			sh->addralign = 4;
			sh->link = elfshname(".dynsym")->shnum;
			shsym(sh, linklookup(ctxt, ".rel", 0));
			break;
		}

		if(eh->machine == EM_PPC64) {
			sh = elfshname(".glink");
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_EXECINSTR;
			sh->addralign = 4;
			shsym(sh, linklookup(ctxt, ".glink", 0));
		}

		sh = elfshname(".plt");
		sh->type = SHT_PROGBITS;
		sh->flags = SHF_ALLOC+SHF_EXECINSTR;
		if(eh->machine == EM_X86_64)
			sh->entsize = 16;
		else if(eh->machine == EM_PPC64) {
			// On ppc64, this is just a table of addresses
			// filled by the dynamic linker
			sh->type = SHT_NOBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = 8;
		} else
			sh->entsize = 4;
		sh->addralign = sh->entsize;
		shsym(sh, linklookup(ctxt, ".plt", 0));

		// On ppc64, .got comes from the input files, so don't
		// create it here, and .got.plt is not used.
		if(eh->machine != EM_PPC64) {
			sh = elfshname(".got");
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = RegSize;
			sh->addralign = RegSize;
			shsym(sh, linklookup(ctxt, ".got", 0));

			sh = elfshname(".got.plt");
			sh->type = SHT_PROGBITS;
			sh->flags = SHF_ALLOC+SHF_WRITE;
			sh->entsize = RegSize;
			sh->addralign = RegSize;
			shsym(sh, linklookup(ctxt, ".got.plt", 0));
		}
		
		sh = elfshname(".hash");
		sh->type = SHT_HASH;
		sh->flags = SHF_ALLOC;
		sh->entsize = 4;
		sh->addralign = RegSize;
		sh->link = elfshname(".dynsym")->shnum;
		shsym(sh, linklookup(ctxt, ".hash", 0));

		/* sh and PT_DYNAMIC for .dynamic section */
		sh = elfshname(".dynamic");
		sh->type = SHT_DYNAMIC;
		sh->flags = SHF_ALLOC+SHF_WRITE;
		sh->entsize = 2*RegSize;
		sh->addralign = RegSize;
		sh->link = elfshname(".dynstr")->shnum;
		shsym(sh, linklookup(ctxt, ".dynamic", 0));
		ph = newElfPhdr();
		ph->type = PT_DYNAMIC;
		ph->flags = PF_R + PF_W;
		phsh(ph, sh);
		
		/*
		 * Thread-local storage segment (really just size).
		 */
		// Do not emit PT_TLS for OpenBSD since ld.so(1) does
		// not currently support it. This is handled
		// appropriately in runtime/cgo.
		if(ctxt->tlsoffset != 0 && HEADTYPE != Hopenbsd) {
			ph = newElfPhdr();
			ph->type = PT_TLS;
			ph->flags = PF_R;
			ph->memsz = -ctxt->tlsoffset;
			ph->align = RegSize;
		}
	}

	if(HEADTYPE == Hlinux) {
		ph = newElfPhdr();
		ph->type = PT_GNU_STACK;
		ph->flags = PF_W+PF_R;
		ph->align = RegSize;
		
		ph = newElfPhdr();
		ph->type = PT_PAX_FLAGS;
		ph->flags = 0x2a00; // mprotect, randexec, emutramp disabled
		ph->align = RegSize;
	}

elfobj:
	sh = elfshname(".shstrtab");
	sh->type = SHT_STRTAB;
	sh->addralign = 1;
	shsym(sh, linklookup(ctxt, ".shstrtab", 0));
	eh->shstrndx = sh->shnum;

	// put these sections early in the list
	if(!debug['s']) {
		elfshname(".symtab");
		elfshname(".strtab");
	}

	for(sect=segtext.sect; sect!=nil; sect=sect->next)
		elfshbits(sect);
	for(sect=segrodata.sect; sect!=nil; sect=sect->next)
		elfshbits(sect);
	for(sect=segdata.sect; sect!=nil; sect=sect->next)
		elfshbits(sect);

	if(linkmode == LinkExternal) {
		for(sect=segtext.sect; sect!=nil; sect=sect->next)
			elfshreloc(sect);
		for(sect=segrodata.sect; sect!=nil; sect=sect->next)
			elfshreloc(sect);
		for(sect=segdata.sect; sect!=nil; sect=sect->next)
			elfshreloc(sect);
		// add a .note.GNU-stack section to mark the stack as non-executable
		sh = elfshname(".note.GNU-stack");
		sh->type = SHT_PROGBITS;
		sh->addralign = 1;
		sh->flags = 0;
	}

	// generate .tbss section for dynamic internal linking (except for OpenBSD)
	// external linking generates .tbss in data.c
	if(linkmode == LinkInternal && !debug['d'] && HEADTYPE != Hopenbsd) {
		sh = elfshname(".tbss");
		sh->type = SHT_NOBITS;
		sh->addralign = RegSize;
		sh->size = -ctxt->tlsoffset;
		sh->flags = SHF_ALLOC | SHF_TLS | SHF_WRITE;
	}

	if(!debug['s']) {
		sh = elfshname(".symtab");
		sh->type = SHT_SYMTAB;
		sh->off = symo;
		sh->size = symsize;
		sh->addralign = RegSize;
		sh->entsize = 8+2*RegSize;
		sh->link = elfshname(".strtab")->shnum;
		sh->info = elfglobalsymndx;

		sh = elfshname(".strtab");
		sh->type = SHT_STRTAB;
		sh->off = symo+symsize;
		sh->size = elfstrsize;
		sh->addralign = 1;

		dwarfaddelfheaders();
	}

	/* Main header */
	eh->ident[EI_MAG0] = '\177';
	eh->ident[EI_MAG1] = 'E';
	eh->ident[EI_MAG2] = 'L';
	eh->ident[EI_MAG3] = 'F';
	if(HEADTYPE == Hfreebsd)
		eh->ident[EI_OSABI] = ELFOSABI_FREEBSD;
	else if(HEADTYPE == Hnetbsd)
		eh->ident[EI_OSABI] = ELFOSABI_NETBSD;
	else if(HEADTYPE == Hopenbsd)
		eh->ident[EI_OSABI] = ELFOSABI_OPENBSD;
	else if(HEADTYPE == Hdragonfly)
		eh->ident[EI_OSABI] = ELFOSABI_NONE;
	if(elf64)
		eh->ident[EI_CLASS] = ELFCLASS64;
	else
		eh->ident[EI_CLASS] = ELFCLASS32;
	if(ctxt->arch->endian == BigEndian)
		eh->ident[EI_DATA] = ELFDATA2MSB;
	else
		eh->ident[EI_DATA] = ELFDATA2LSB;
	eh->ident[EI_VERSION] = EV_CURRENT;

	if(linkmode == LinkExternal)
		eh->type = ET_REL;
	else
		eh->type = ET_EXEC;

	if(linkmode != LinkExternal)
		eh->entry = entryvalue();

	eh->version = EV_CURRENT;

	if(pph != nil) {
		pph->filesz = eh->phnum * eh->phentsize;
		pph->memsz = pph->filesz;
	}

	cseek(0);
	a = 0;
	a += elfwritehdr();
	a += elfwritephdrs();
	a += elfwriteshdrs();
	if(!debug['d'])
		a += elfwriteinterp();
	if(linkmode != LinkExternal) {
		if(HEADTYPE == Hnetbsd)
			a += elfwritenetbsdsig();
		if(HEADTYPE == Hopenbsd)
			a += elfwriteopenbsdsig();
		if(buildinfolen > 0)
			a += elfwritebuildinfo();
	}
	if(a > ELFRESERVE)	
		diag("ELFRESERVE too small: %lld > %d", a, ELFRESERVE);
}
