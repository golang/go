// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PE (Portable Executable) file writing
// http://www.microsoft.com/whdc/system/platform/firmware/PECOFF.mspx

#include <time.h>

#include "l.h"
#include "../ld/lib.h"
#include "../ld/pe.h"

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

static int pe64;
static int nsect;
static int sect_virt_begin;
static int sect_raw_begin = PERESERVE;

static IMAGE_FILE_HEADER fh;
static IMAGE_OPTIONAL_HEADER oh;
static IMAGE_SECTION_HEADER sh[16];
static IMAGE_SECTION_HEADER *textsect, *datsect, *bsssect;

static IMAGE_SECTION_HEADER*
new_section(char *name, int size, int noraw)
{
	IMAGE_SECTION_HEADER *h;

	if(nsect == 16) {
		diag("too many sections");
		errorexit();
	}
	h = &sh[nsect++];
	strncpy((char*)h->Name, name, sizeof(h->Name));
	h->VirtualSize = size;
	if(!sect_virt_begin)
		sect_virt_begin = 0x1000;
	h->VirtualAddress = sect_virt_begin;
	sect_virt_begin = rnd(sect_virt_begin+size, 0x1000);
	if(!noraw) {
		h->SizeOfRawData = rnd(size, PEALIGN);
		h->PointerToRawData = sect_raw_begin;
		sect_raw_begin += h->SizeOfRawData;
	}
	return h;
}

void
peinit(void)
{
	switch(thechar) {
	// 64-bit architectures
	case '6':
		pe64 = 1;
		break;
	// 32-bit architectures
	default:
		break;
	}
}

static void
pewrite(void)
{
	int i, j;

	write(cout, dosstub, sizeof dosstub);
	strnput("PE", 4);

	for (i=0; i<sizeof(fh); i++)
		cput(((char*)&fh)[i]);
	for (i=0; i<sizeof(oh); i++)
		cput(((char*)&oh)[i]);
	for (i=0; i<nsect; i++)
		for (j=0; j<sizeof(sh[i]); j++)
			cput(((char*)&sh[i])[j]);
}

void
dope(void)
{
	textsect = new_section(".text", textsize, 0);
	textsect->Characteristics = IMAGE_SCN_CNT_CODE|
		IMAGE_SCN_CNT_INITIALIZED_DATA|
		IMAGE_SCN_MEM_EXECUTE|IMAGE_SCN_MEM_READ;

	datsect = new_section(".data", datsize, 0);
	datsect->Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA|
		IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE;
	INITDAT = PEBASE+datsect->VirtualAddress;

	bsssect = new_section(".bss", bsssize, 1);
	bsssect->Characteristics = IMAGE_SCN_CNT_UNINITIALIZED_DATA|
		IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE;
}

static void
strput(char *s)
{
	while(*s)
		cput(*s++);
	cput('\0');
}

static void
add_import_table(void)
{
	IMAGE_IMPORT_DESCRIPTOR ds[2], *d;
	char *dllname = "kernel32.dll";
	struct {
		char *name;
		uint32 thunk;
	} *f, fs[] = {
		{ "GetProcAddress", 0 },
		{ "LoadLibraryExA", 0 },
		{ 0, 0 }
	};

	uint32 size = 0;
	memset(ds, 0, sizeof(ds));
	size += sizeof(ds);
	ds[0].Name = size;
	size += strlen(dllname) + 1;
	for(f=fs; f->name; f++) {
		f->thunk = size;
		size += sizeof(uint16) + strlen(f->name) + 1;
	}
	ds[0].FirstThunk = size;
	for(f=fs; f->name; f++)
		size += sizeof(fs[0].thunk);

	IMAGE_SECTION_HEADER *isect;
	isect = new_section(".idata", size, 0);
	isect->Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA|
		IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE;
	
	uint32 va = isect->VirtualAddress;
	oh.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress = va;
	oh.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].Size = isect->VirtualSize;

	ds[0].Name += va;
	ds[0].FirstThunk += va;
	for(f=fs; f->name; f++)
		f->thunk += va;

	vlong off = seek(cout, 0, 1);
	seek(cout, 0, 2);
	for(d=ds; ; d++) {
		lputl(d->OriginalFirstThunk);
		lputl(d->TimeDateStamp);
		lputl(d->ForwarderChain);
		lputl(d->Name);
		lputl(d->FirstThunk);
		if(!d->Name) 
			break;
	}
	strput(dllname);
	for(f=fs; f->name; f++) {
		wputl(0);
		strput(f->name);
	}
	for(f=fs; f->name; f++)
		lputl(f->thunk);
	strnput("", isect->SizeOfRawData - size);
	cflush();
	seek(cout, off, 0);
}

void
asmbpe(void)
{
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

	if(!debug['s']) {
		IMAGE_SECTION_HEADER *symsect;
		symsect = new_section(".symdat", 8+symsize+lcsize, 0);
		symsect->Characteristics = IMAGE_SCN_MEM_READ|
			IMAGE_SCN_CNT_INITIALIZED_DATA;
	}

	add_import_table();

	fh.NumberOfSections = nsect;
	fh.TimeDateStamp = time(0);
	fh.SizeOfOptionalHeader = sizeof(oh);
	fh.Characteristics = IMAGE_FILE_RELOCS_STRIPPED|
		IMAGE_FILE_EXECUTABLE_IMAGE|IMAGE_FILE_DEBUG_STRIPPED;
	if(thechar == '8')
		fh.Characteristics |= IMAGE_FILE_32BIT_MACHINE;

	oh.Magic = 0x10b;	// PE32
	oh.MajorLinkerVersion = 1;
	oh.MinorLinkerVersion = 0;
	oh.SizeOfCode = textsect->SizeOfRawData;
	oh.SizeOfInitializedData = datsect->SizeOfRawData;
	oh.SizeOfUninitializedData = bsssect->SizeOfRawData;
	oh.AddressOfEntryPoint = entryvalue()-PEBASE;
	oh.BaseOfCode = textsect->VirtualAddress;
	oh.BaseOfData = datsect->VirtualAddress;

	oh.ImageBase = PEBASE;
	oh.SectionAlignment = 0x00001000;
	oh.FileAlignment = PEALIGN;
	oh.MajorOperatingSystemVersion = 4;
	oh.MinorOperatingSystemVersion = 0;
	oh.MajorImageVersion = 1;
	oh.MinorImageVersion = 0;
	oh.MajorSubsystemVersion = 4;
	oh.MinorSubsystemVersion = 0;
	oh.SizeOfImage = sect_virt_begin;
	oh.SizeOfHeaders = PERESERVE;
	oh.Subsystem = 3;	// WINDOWS_CUI
	oh.SizeOfStackReserve = 0x00200000;
	oh.SizeOfStackCommit = 0x00001000;
	oh.SizeOfHeapReserve = 0x00100000;
	oh.SizeOfHeapCommit = 0x00001000;
	oh.NumberOfRvaAndSizes = 16;

	pewrite();
}
