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
#define	NSECT	32

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
	case '6':
		elf64 = 1;
		hdr.phoff = ELF64HDRSIZE;	/* Must be be ELF64HDRSIZE: first PHdr must follow ELF header */
		hdr.shoff = ELF64HDRSIZE;	/* Will move as we add PHeaders */
		hdr.ehsize = ELF64HDRSIZE;	/* Must be ELF64HDRSIZE */
		hdr.phentsize = ELF64PHDRSIZE;	/* Must be ELF64PHDRSIZE */
		hdr.shentsize = ELF64SHDRSIZE;	/* Must be ELF64SHDRSIZE */
		break;

	// 32-bit architectures
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
newElfShstrtab(vlong name)
{
	hdr.shstrndx = hdr.shnum;
	return newElfShdr(name);
}

ElfShdr*
newElfShdr(vlong name)
{
	ElfShdr *e;

	e = mal(sizeof *e);
	e->name = name;
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
elfwritedynent(Sym *s, int tag, uint64 val)
{
	if(elf64) {
		adduint64(s, tag);
		adduint64(s, val);
	} else {
		adduint32(s, tag);
		adduint32(s, val);
	}
}

void
elfwritedynentsym(Sym *s, int tag, Sym *t)
{
	if(elf64)
		adduint64(s, tag);
	else
		adduint32(s, tag);
	addaddr(s, t);
}

void
elfwritedynentsymsize(Sym *s, int tag, Sym *t)
{
	if(elf64)
		adduint64(s, tag);
	else
		adduint32(s, tag);
	addsize(s, t);
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
elfwriteinterp(vlong stridx)
{
	ElfShdr *sh = nil;
	int i;

	for(i = 0; i < hdr.shnum; i++)
		if(shdr[i]->name == stridx)
			sh = shdr[i];
	if(sh == nil || interp == nil)
		return 0;

	cseek(sh->off);
	cwrite(interp, sh->size);
	return sh->size;
}

// Defined in NetBSD's sys/exec_elf.h
#define ELF_NOTE_TYPE_NETBSD_TAG	1
#define ELF_NOTE_NETBSD_NAMESZ		7
#define ELF_NOTE_NETBSD_DESCSZ		4
#define ELF_NOTE_NETBSD_NAME		"NetBSD\0\0"
#define ELF_NOTE_NETBSD_VERSION		599000000	/* NetBSD 5.99 */

int
elfnetbsdsig(ElfShdr *sh, uint64 startva, uint64 resoff)
{
	int n;

	n = sizeof(Elf_Note) + ELF_NOTE_NETBSD_NAMESZ + ELF_NOTE_NETBSD_DESCSZ + 1;
	n += resoff % 4;
	sh->addr = startva + resoff - n;
	sh->off = resoff - n;
	sh->size = n;

	return n;
}

int
elfwritenetbsdsig(vlong stridx) {
	ElfShdr *sh = nil;
	int i;

	for(i = 0; i < hdr.shnum; i++)
		if(shdr[i]->name == stridx)
			sh = shdr[i];
	if(sh == nil)
		return 0;

	// Write Elf_Note header followed by NetBSD string.
	cseek(sh->off);
	LPUT(ELF_NOTE_NETBSD_NAMESZ);
	LPUT(ELF_NOTE_NETBSD_DESCSZ);
	LPUT(ELF_NOTE_TYPE_NETBSD_TAG);
	cwrite(ELF_NOTE_NETBSD_NAME, 8);
	LPUT(ELF_NOTE_NETBSD_VERSION);

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
	Sym *s, *sy, *dynstr;
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
	s = lookup(".hash", 0);
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
		cursym = nil;
		diag("out of memory");
		errorexit();
	}
	memset(need, 0, nsym * sizeof need[0]);
	memset(chain, 0, nsym * sizeof chain[0]);
	memset(buckets, 0, nbucket * sizeof buckets[0]);
	for(sy=allsym; sy!=S; sy=sy->allsym) {
		if (sy->dynid <= 0)
			continue;

		if(sy->dynimpvers)
			need[sy->dynid] = addelflib(&needlib, sy->dynimplib, sy->dynimpvers);

		name = sy->dynimpname;
		if(name == nil)
			name = sy->name;
		hc = elfhash((uchar*)name);

		b = hc % nbucket;
		chain[sy->dynid] = buckets[b];
		buckets[b] = sy->dynid;
	}

	adduint32(s, nbucket);
	adduint32(s, nsym);
	for(i = 0; i<nbucket; i++)
		adduint32(s, buckets[i]);
	for(i = 0; i<nsym; i++)
		adduint32(s, chain[i]);

	free(chain);
	free(buckets);
	
	// version symbols
	dynstr = lookup(".dynstr", 0);
	s = lookup(".gnu.version_r", 0);
	i = 2;
	nfile = 0;
	for(l=needlib; l; l=l->next) {
		nfile++;
		// header
		adduint16(s, 1);  // table version
		j = 0;
		for(x=l->aux; x; x=x->next)
			j++;
		adduint16(s, j);	// aux count
		adduint32(s, addstring(dynstr, l->file));  // file string offset
		adduint32(s, 16);  // offset from header to first aux
		if(l->next)
			adduint32(s, 16+j*16);  // offset from this header to next
		else
			adduint32(s, 0);
		
		for(x=l->aux; x; x=x->next) {
			x->num = i++;
			// aux struct
			adduint32(s, elfhash((uchar*)x->vers));  // hash
			adduint16(s, 0);  // flags
			adduint16(s, x->num);  // other - index we refer to this by
			adduint32(s, addstring(dynstr, x->vers));  // version string offset
			if(x->next)
				adduint32(s, 16);  // offset from this aux to next
			else
				adduint32(s, 0);
		}
	}

	// version references
	s = lookup(".gnu.version", 0);
	for(i=0; i<nsym; i++) {
		if(i == 0)
			adduint16(s, 0); // first entry - no symbol
		else if(need[i] == nil)
			adduint16(s, 1); // global
		else
			adduint16(s, need[i]->num);
	}

	free(need);

	s = lookup(".dynamic", 0);
	elfverneed = nfile;
	if(elfverneed) {
		elfwritedynentsym(s, DT_VERNEED, lookup(".gnu.version_r", 0));
		elfwritedynent(s, DT_VERNEEDNUM, nfile);
		elfwritedynentsym(s, DT_VERSYM, lookup(".gnu.version", 0));
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
elfshbits(Section *sect)
{
	int i, off;
	ElfShdr *sh;
	
	for(i=0; i<nelfstr; i++) {
		if(strcmp(sect->name, elfstr[i].s) == 0) {
			off = elfstr[i].off;
			goto found;
		}
	}
	diag("cannot find elf name %s", sect->name);
	errorexit();
	return nil;

found:
	for(i=0; i<hdr.shnum; i++) {
		sh = shdr[i];
		if(sh->name == off)
			return sh;
	}

	sh = newElfShdr(off);
	if(sect->vaddr < sect->seg->vaddr + sect->seg->filelen)
		sh->type = SHT_PROGBITS;
	else
		sh->type = SHT_NOBITS;
	sh->flags = SHF_ALLOC;
	if(sect->rwx & 1)
		sh->flags |= SHF_EXECINSTR;
	if(sect->rwx & 2)
		sh->flags |= SHF_WRITE;
	sh->addr = sect->vaddr;
	sh->addralign = PtrSize;
	sh->size = sect->len;
	sh->off = sect->seg->fileoff + sect->vaddr - sect->seg->vaddr;
	
	return sh;
}
