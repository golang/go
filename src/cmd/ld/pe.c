// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PE (Portable Executable) file writing
// http://www.microsoft.com/whdc/system/platform/firmware/PECOFF.mspx

#include "l.h"
#include "../ld/lib.h"
#include "../ld/pe.h"
#include "../ld/dwarf.h"

// DOS stub that prints out
// "This program cannot be run in DOS mode."
static char dosstub[] =
{
	0x4d, 0x5a, 0x90, 0x00, 0x03, 0x00, 0x04, 0x00,
	0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00,
	0x8b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
	0x0e, 0x1f, 0xba, 0x0e, 0x00, 0xb4, 0x09, 0xcd,
	0x21, 0xb8, 0x01, 0x4c, 0xcd, 0x21, 0x54, 0x68,
	0x69, 0x73, 0x20, 0x70, 0x72, 0x6f, 0x67, 0x72,
	0x61, 0x6d, 0x20, 0x63, 0x61, 0x6e, 0x6e, 0x6f,
	0x74, 0x20, 0x62, 0x65, 0x20, 0x72, 0x75, 0x6e,
	0x20, 0x69, 0x6e, 0x20, 0x44, 0x4f, 0x53, 0x20,
	0x6d, 0x6f, 0x64, 0x65, 0x2e, 0x0d, 0x0d, 0x0a,
	0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

static LSym *rsrcsym;

static char* strtbl;
static int strtblnextoff;
static int strtblsize;

int32 PESECTHEADR;
int32 PEFILEHEADR;

static int pe64;
static int nsect;
static int nextsectoff;
static int nextfileoff;
static int textsect;
static int datasect;

static IMAGE_FILE_HEADER fh;
static IMAGE_OPTIONAL_HEADER oh;
static PE64_IMAGE_OPTIONAL_HEADER oh64;
static IMAGE_SECTION_HEADER sh[16];
static IMAGE_DATA_DIRECTORY* dd;

#define	set(n, v)	(pe64 ? (oh64.n = v) : (oh.n = v))
#define	put(v)		(pe64 ? vputl(v) : lputl(v))

typedef struct Imp Imp;
struct Imp {
	LSym* s;
	uvlong off;
	Imp* next;
};

typedef struct Dll Dll;
struct Dll {
	char* name;
	uvlong nameoff;
	uvlong thunkoff;
	Imp* ms;
	Dll* next;
};

static Dll* dr;

static LSym *dexport[1024];
static int nexport;

typedef struct COFFSym COFFSym;
struct COFFSym
{
	LSym* sym;
	int strtbloff;
	int sect;
	vlong value;
};

static COFFSym* coffsym;
static int ncoffsym;

static IMAGE_SECTION_HEADER*
addpesection(char *name, int sectsize, int filesize)
{
	IMAGE_SECTION_HEADER *h;

	if(nsect == 16) {
		diag("too many sections");
		errorexit();
	}
	h = &sh[nsect++];
	strncpy((char*)h->Name, name, sizeof(h->Name));
	h->VirtualSize = sectsize;
	h->VirtualAddress = nextsectoff;
	nextsectoff = rnd(nextsectoff+sectsize, PESECTALIGN);
	h->PointerToRawData = nextfileoff;
	if(filesize > 0) {
		h->SizeOfRawData = rnd(filesize, PEFILEALIGN);
		nextfileoff += h->SizeOfRawData;
	}
	return h;
}

static void
chksectoff(IMAGE_SECTION_HEADER *h, vlong off)
{
	if(off != h->PointerToRawData) {
		diag("%s.PointerToRawData = %#llux, want %#llux", (char *)h->Name, (vlong)h->PointerToRawData, off);
		errorexit();
	}
}

static void
chksectseg(IMAGE_SECTION_HEADER *h, Segment *s)
{
	if(s->vaddr-PEBASE != h->VirtualAddress) {
		diag("%s.VirtualAddress = %#llux, want %#llux", (char *)h->Name, (vlong)h->VirtualAddress, (vlong)(s->vaddr-PEBASE));
		errorexit();
	}
	if(s->fileoff != h->PointerToRawData) {
		diag("%s.PointerToRawData = %#llux, want %#llux", (char *)h->Name, (vlong)h->PointerToRawData, (vlong)(s->fileoff));
		errorexit();
	}
}

void
peinit(void)
{
	int32 l;

	switch(thechar) {
	// 64-bit architectures
	case '6':
		pe64 = 1;
		l = sizeof(oh64);
		dd = oh64.DataDirectory;
		break;
	// 32-bit architectures
	default:
		l = sizeof(oh);
		dd = oh.DataDirectory;
		break;
	}
	
	PEFILEHEADR = rnd(sizeof(dosstub)+sizeof(fh)+l+sizeof(sh), PEFILEALIGN);
	PESECTHEADR = rnd(PEFILEHEADR, PESECTALIGN);
	nextsectoff = PESECTHEADR;
	nextfileoff = PEFILEHEADR;

	// some mingw libs depend on this symbol, for example, FindPESectionByName
	xdefine("__image_base__", SDATA, PEBASE);
	xdefine("_image_base__", SDATA, PEBASE);
}

static void
pewrite(void)
{
	cseek(0);
	cwrite(dosstub, sizeof dosstub);
	strnput("PE", 4);
	// TODO: This code should not assume that the
	// memory representation is little-endian or
	// that the structs are packed identically to
	// their file representation.
	cwrite(&fh, sizeof fh);
	if(pe64)
		cwrite(&oh64, sizeof oh64);
	else
		cwrite(&oh, sizeof oh);
	cwrite(sh, nsect * sizeof sh[0]);
}

static void
strput(char *s)
{
	int n;

	for(n=0; *s; n++)
		cput(*s++);
	cput('\0');
	n++;
	// string must be padded to even size
	if(n%2)
		cput('\0');
}

static Dll* 
initdynimport(void)
{
	Imp *m;
	Dll *d;
	LSym *s, *dynamic;

	dr = nil;
	m = nil;
	for(s = ctxt->allsym; s != S; s = s->allsym) {
		if(!s->reachable || s->type != SDYNIMPORT)
			continue;
		for(d = dr; d != nil; d = d->next) {
			if(strcmp(d->name,s->dynimplib) == 0) {
				m = mal(sizeof *m);
				break;
			}
		}
		if(d == nil) {
			d = mal(sizeof *d);
			d->name = s->dynimplib;
			d->next = dr;
			dr = d;
			m = mal(sizeof *m);
		}
		m->s = s;
		m->next = d->ms;
		d->ms = m;
	}
	
	dynamic = linklookup(ctxt, ".windynamic", 0);
	dynamic->reachable = 1;
	dynamic->type = SWINDOWS;
	for(d = dr; d != nil; d = d->next) {
		for(m = d->ms; m != nil; m = m->next) {
			m->s->type = SWINDOWS | SSUB;
			m->s->sub = dynamic->sub;
			dynamic->sub = m->s;
			m->s->value = dynamic->size;
			dynamic->size += PtrSize;
		}
		dynamic->size += PtrSize;
	}
		
	return dr;
}

static void
addimports(IMAGE_SECTION_HEADER *datsect)
{
	IMAGE_SECTION_HEADER *isect;
	uvlong n, oftbase, ftbase;
	vlong startoff, endoff;
	Imp *m;
	Dll *d;
	LSym* dynamic;
	
	startoff = cpos();
	dynamic = linklookup(ctxt, ".windynamic", 0);

	// skip import descriptor table (will write it later)
	n = 0;
	for(d = dr; d != nil; d = d->next)
		n++;
	cseek(startoff + sizeof(IMAGE_IMPORT_DESCRIPTOR) * (n + 1));

	// write dll names
	for(d = dr; d != nil; d = d->next) {
		d->nameoff = cpos() - startoff;
		strput(d->name);
	}

	// write function names
	for(d = dr; d != nil; d = d->next) {
		for(m = d->ms; m != nil; m = m->next) {
			m->off = nextsectoff + cpos() - startoff;
			wputl(0); // hint
			strput(m->s->extname);
		}
	}
	
	// write OriginalFirstThunks
	oftbase = cpos() - startoff;
	n = cpos();
	for(d = dr; d != nil; d = d->next) {
		d->thunkoff = cpos() - n;
		for(m = d->ms; m != nil; m = m->next)
			put(m->off);
		put(0);
	}

	// add pe section and pad it at the end
	n = cpos() - startoff;
	isect = addpesection(".idata", n, n);
	isect->Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA|
		IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE;
	chksectoff(isect, startoff);
	strnput("", isect->SizeOfRawData - n);
	endoff = cpos();

	// write FirstThunks (allocated in .data section)
	ftbase = dynamic->value - datsect->VirtualAddress - PEBASE;
	cseek(datsect->PointerToRawData + ftbase);
	for(d = dr; d != nil; d = d->next) {
		for(m = d->ms; m != nil; m = m->next)
			put(m->off);
		put(0);
	}
	
	// finally write import descriptor table
	cseek(startoff);
	for(d = dr; d != nil; d = d->next) {
		lputl(isect->VirtualAddress + oftbase + d->thunkoff);
		lputl(0);
		lputl(0);
		lputl(isect->VirtualAddress + d->nameoff);
		lputl(datsect->VirtualAddress + ftbase + d->thunkoff);
	}
	lputl(0); //end
	lputl(0);
	lputl(0);
	lputl(0);
	lputl(0);
	
	// update data directory
	dd[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress = isect->VirtualAddress;
	dd[IMAGE_DIRECTORY_ENTRY_IMPORT].Size = isect->VirtualSize;
	dd[IMAGE_DIRECTORY_ENTRY_IAT].VirtualAddress = dynamic->value - PEBASE;
	dd[IMAGE_DIRECTORY_ENTRY_IAT].Size = dynamic->size;

	cseek(endoff);
}

static int
scmp(const void *p1, const void *p2)
{
	LSym *s1, *s2;

	s1 = *(LSym**)p1;
	s2 = *(LSym**)p2;
	return strcmp(s1->extname, s2->extname);
}

static void
initdynexport(void)
{
	LSym *s;
	
	nexport = 0;
	for(s = ctxt->allsym; s != S; s = s->allsym) {
		if(!s->reachable || !(s->cgoexport & CgoExportDynamic))
			continue;
		if(nexport+1 > sizeof(dexport)/sizeof(dexport[0])) {
			diag("pe dynexport table is full");
			errorexit();
		}
		
		dexport[nexport] = s;
		nexport++;
	}
	
	qsort(dexport, nexport, sizeof dexport[0], scmp);
}

void
addexports(void)
{
	IMAGE_SECTION_HEADER *sect;
	IMAGE_EXPORT_DIRECTORY e;
	int size, i, va, va_name, va_addr, va_na, v;

	size = sizeof e + 10*nexport + strlen(outfile) + 1;
	for(i=0; i<nexport; i++)
		size += strlen(dexport[i]->extname) + 1;
	
	if (nexport == 0)
		return;
		
	sect = addpesection(".edata", size, size);
	sect->Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_MEM_READ;
	chksectoff(sect, cpos());
	va = sect->VirtualAddress;
	dd[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress = va;
	dd[IMAGE_DIRECTORY_ENTRY_EXPORT].Size = sect->VirtualSize;

	va_name = va + sizeof e + nexport*4;
	va_addr = va + sizeof e;
	va_na = va + sizeof e + nexport*8;

	e.Characteristics = 0;
	e.MajorVersion = 0;
	e.MinorVersion = 0;
	e.NumberOfFunctions = nexport;
	e.NumberOfNames = nexport;
	e.Name = va + sizeof e + nexport*10; // Program names.
	e.Base = 1;
	e.AddressOfFunctions = va_addr;
	e.AddressOfNames = va_name;
	e.AddressOfNameOrdinals = va_na;
	// put IMAGE_EXPORT_DIRECTORY
	for (i=0; i<sizeof(e); i++)
		cput(((char*)&e)[i]);
	// put EXPORT Address Table
	for(i=0; i<nexport; i++)
		lputl(dexport[i]->value - PEBASE);		
	// put EXPORT Name Pointer Table
	v = e.Name + strlen(outfile)+1;
	for(i=0; i<nexport; i++) {
		lputl(v);
		v += strlen(dexport[i]->extname)+1;
	}
	// put EXPORT Ordinal Table
	for(i=0; i<nexport; i++)
		wputl(i);
	// put Names
	strnput(outfile, strlen(outfile)+1);
	for(i=0; i<nexport; i++)
		strnput(dexport[i]->extname, strlen(dexport[i]->extname)+1);
	strnput("", sect->SizeOfRawData - size);
}

void
dope(void)
{
	LSym *rel;

	/* relocation table */
	rel = linklookup(ctxt, ".rel", 0);
	rel->reachable = 1;
	rel->type = SELFROSECT;

	initdynimport();
	initdynexport();
}

static int
strtbladd(char *name)
{
	int newsize, thisoff;

	newsize = strtblnextoff + strlen(name) + 1;
	if(newsize > strtblsize) {
		strtblsize = 2 * (newsize + (1<<18));
		strtbl = realloc(strtbl, strtblsize);
	}
	thisoff = strtblnextoff+4; // first string starts at offset=4
	strcpy(&strtbl[strtblnextoff], name);
	strtblnextoff += strlen(name);
	strtbl[strtblnextoff] = 0;
	strtblnextoff++;
	return thisoff;
}

/*
 * For more than 8 characters section names, name contains a slash (/) that is 
 * followed by an ASCII representation of a decimal number that is an offset into 
 * the string table. 
 * reference: pecoff_v8.docx Page 24.
 * <http://www.microsoft.com/whdc/system/platform/firmware/PECOFFdwn.mspx>
 */
IMAGE_SECTION_HEADER*
newPEDWARFSection(char *name, vlong size)
{
	IMAGE_SECTION_HEADER *h;
	char s[8];
	int off;

	if(size == 0)
		return nil;

	off = strtbladd(name);
	sprint(s, "/%d\0", off);
	h = addpesection(s, size, size);
	h->Characteristics = IMAGE_SCN_MEM_READ|
		IMAGE_SCN_MEM_DISCARDABLE;

	return h;
}

static void
addsym(LSym *s, char *name, int type, vlong addr, vlong size, int ver, LSym *gotype)
{
	COFFSym *cs;
	USED(name);
	USED(addr);
	USED(size);
	USED(ver);
	USED(gotype);

	if(s == nil)
		return;

	if(s->sect == nil)
		return;

	switch(type) {
	default:
		return;
	case 'D':
	case 'B':
	case 'T':
		break;
	}

	if(coffsym) {
		cs = &coffsym[ncoffsym];
		cs->sym = s;
		if(strlen(s->name) > 8)
			cs->strtbloff = strtbladd(s->name);
		if(s->value >= segdata.vaddr) {
			cs->value = s->value - segdata.vaddr;
			cs->sect = datasect;
		} else if(s->value >= segtext.vaddr) {
			cs->value = s->value - segtext.vaddr;
			cs->sect = textsect;
		} else {
			cs->value = 0;
			cs->sect = 0;
			diag("addsym %#llx", addr);
		}
	}
	ncoffsym++;
}

static void
addsymtable(void)
{
	IMAGE_SECTION_HEADER *h;
	int i, size;
	COFFSym *s;

	if(!debug['s']) {
		genasmsym(addsym);
		coffsym = mal(ncoffsym * sizeof coffsym[0]);
		ncoffsym = 0;
		genasmsym(addsym);
	}

	size = strtblnextoff + 4 + 18*ncoffsym;
	h = addpesection(".symtab", size, size);
	h->Characteristics = IMAGE_SCN_MEM_READ|
		IMAGE_SCN_MEM_DISCARDABLE;
	chksectoff(h, cpos());
	fh.PointerToSymbolTable = cpos();
	fh.NumberOfSymbols = ncoffsym;
	
	// put COFF symbol table
	for (i=0; i<ncoffsym; i++) {
		s = &coffsym[i];
		if(s->strtbloff == 0)
			strnput(s->sym->name, 8);
		else {
			lputl(0);
			lputl(s->strtbloff);
		}
		lputl(s->value);
		wputl(s->sect);
		wputl(0x0308);  // "array of structs"
		cput(2);        // storage class: external
		cput(0);        // no aux entries
	}

	// put COFF string table
	lputl(strtblnextoff + 4);
	for (i=0; i<strtblnextoff; i++)
		cput(strtbl[i]);
	strnput("", h->SizeOfRawData - size);
}

void
setpersrc(LSym *sym)
{
	if(rsrcsym != nil)
		diag("too many .rsrc sections");
	
	rsrcsym = sym;
}

void
addpersrc(void)
{
	IMAGE_SECTION_HEADER *h;
	uchar *p;
	uint32 val;
	Reloc *r;

	if(rsrcsym == nil)
		return;
	
	h = addpesection(".rsrc", rsrcsym->size, rsrcsym->size);
	h->Characteristics = IMAGE_SCN_MEM_READ|
		IMAGE_SCN_MEM_WRITE | IMAGE_SCN_CNT_INITIALIZED_DATA;
	chksectoff(h, cpos());
	// relocation
	for(r=rsrcsym->r; r<rsrcsym->r+rsrcsym->nr; r++) {
		p = rsrcsym->p + r->off;
		val = h->VirtualAddress + r->add;
		// 32-bit little-endian
		p[0] = val;
		p[1] = val>>8;
		p[2] = val>>16;
		p[3] = val>>24;
	}
	cwrite(rsrcsym->p, rsrcsym->size);
	strnput("", h->SizeOfRawData - rsrcsym->size);

	// update data directory
	dd[IMAGE_DIRECTORY_ENTRY_RESOURCE].VirtualAddress = h->VirtualAddress;
	dd[IMAGE_DIRECTORY_ENTRY_RESOURCE].Size = h->VirtualSize;
}

void
asmbpe(void)
{
	IMAGE_SECTION_HEADER *t, *d;

	switch(thechar) {
	default:
		diag("unknown PE architecture");
		errorexit();
	case '6':
		fh.Machine = IMAGE_FILE_MACHINE_AMD64;
		break;
	case '8':
		fh.Machine = IMAGE_FILE_MACHINE_I386;
		break;
	}

	t = addpesection(".text", segtext.len, segtext.len);
	t->Characteristics = IMAGE_SCN_CNT_CODE|
		IMAGE_SCN_CNT_INITIALIZED_DATA|
		IMAGE_SCN_MEM_EXECUTE|IMAGE_SCN_MEM_READ;
	chksectseg(t, &segtext);
	textsect = nsect;

	d = addpesection(".data", segdata.len, segdata.filelen);
	d->Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA|
		IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE;
	chksectseg(d, &segdata);
	datasect = nsect;

	if(!debug['s'])
		dwarfaddpeheaders();

	cseek(nextfileoff);
	addimports(d);
	addexports();
	addsymtable();
	addpersrc();

	fh.NumberOfSections = nsect;
	fh.TimeDateStamp = time(0);
	fh.Characteristics = IMAGE_FILE_RELOCS_STRIPPED|
		IMAGE_FILE_EXECUTABLE_IMAGE|IMAGE_FILE_DEBUG_STRIPPED;
	if (pe64) {
		fh.SizeOfOptionalHeader = sizeof(oh64);
		fh.Characteristics |= IMAGE_FILE_LARGE_ADDRESS_AWARE;
		set(Magic, 0x20b);	// PE32+
	} else {
		fh.SizeOfOptionalHeader = sizeof(oh);
		fh.Characteristics |= IMAGE_FILE_32BIT_MACHINE;
		set(Magic, 0x10b);	// PE32
		oh.BaseOfData = d->VirtualAddress;
	}
	set(MajorLinkerVersion, 3);
	set(MinorLinkerVersion, 0);
	set(SizeOfCode, t->SizeOfRawData);
	set(SizeOfInitializedData, d->SizeOfRawData);
	set(SizeOfUninitializedData, 0);
	set(AddressOfEntryPoint, entryvalue()-PEBASE);
	set(BaseOfCode, t->VirtualAddress);
	set(ImageBase, PEBASE);
	set(SectionAlignment, PESECTALIGN);
	set(FileAlignment, PEFILEALIGN);
	set(MajorOperatingSystemVersion, 4);
	set(MinorOperatingSystemVersion, 0);
	set(MajorImageVersion, 1);
	set(MinorImageVersion, 0);
	set(MajorSubsystemVersion, 4);
	set(MinorSubsystemVersion, 0);
	set(SizeOfImage, nextsectoff);
	set(SizeOfHeaders, PEFILEHEADR);
	if(strcmp(headstring, "windowsgui") == 0)
		set(Subsystem, IMAGE_SUBSYSTEM_WINDOWS_GUI);
	else
		set(Subsystem, IMAGE_SUBSYSTEM_WINDOWS_CUI);

	// Disable stack growth as we don't want Windows to
	// fiddle with the thread stack limits, which we set
	// ourselves to circumvent the stack checks in the
	// Windows exception dispatcher.
	// Commit size must be strictly less than reserve
	// size otherwise reserve will be rounded up to a
	// larger size, as verified with VMMap.

	// Go code would be OK with 64k stacks, but we need larger stacks for cgo.
	// That default stack reserve size affects only the main thread,
	// for other threads we specify stack size in runtime explicitly
	// (runtime knows whether cgo is enabled or not).
	// If you change stack reserve sizes here,
	// change STACKSIZE in runtime/cgo/gcc_windows_{386,amd64}.c as well.
	if(!iscgo) {
		set(SizeOfStackReserve, 0x00010000);
		set(SizeOfStackCommit, 0x0000ffff);
	} else {
		set(SizeOfStackReserve, pe64 ? 0x00200000 : 0x00100000);
		// account for 2 guard pages
		set(SizeOfStackCommit, (pe64 ? 0x00200000 : 0x00100000) - 0x2000);
	}
	set(SizeOfHeapReserve, 0x00100000);
	set(SizeOfHeapCommit, 0x00001000);
	set(NumberOfRvaAndSizes, 16);

	pewrite();
}
