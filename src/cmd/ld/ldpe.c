// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"l.h"
#include	"lib.h"
#include	"../ld/pe.h"

#define IMAGE_SCN_MEM_DISCARDABLE 0x2000000

#define IMAGE_SYM_UNDEFINED	0
#define IMAGE_SYM_ABSOLUTE (-1)
#define IMAGE_SYM_DEBUG	(-2)
#define IMAGE_SYM_TYPE_NULL 0
#define IMAGE_SYM_TYPE_VOID 1
#define IMAGE_SYM_TYPE_CHAR 2
#define IMAGE_SYM_TYPE_SHORT 3
#define IMAGE_SYM_TYPE_INT 4
#define IMAGE_SYM_TYPE_LONG 5
#define IMAGE_SYM_TYPE_FLOAT 6
#define IMAGE_SYM_TYPE_DOUBLE 7
#define IMAGE_SYM_TYPE_STRUCT 8
#define IMAGE_SYM_TYPE_UNION 9
#define IMAGE_SYM_TYPE_ENUM 10
#define IMAGE_SYM_TYPE_MOE 11
#define IMAGE_SYM_TYPE_BYTE 12
#define IMAGE_SYM_TYPE_WORD 13
#define IMAGE_SYM_TYPE_UINT 14
#define IMAGE_SYM_TYPE_DWORD 15
#define IMAGE_SYM_TYPE_PCODE 32768
#define IMAGE_SYM_DTYPE_NULL 0
#define IMAGE_SYM_DTYPE_POINTER 0x10
#define IMAGE_SYM_DTYPE_FUNCTION 0x20
#define IMAGE_SYM_DTYPE_ARRAY 0x30
#define IMAGE_SYM_CLASS_END_OF_FUNCTION	(-1)
#define IMAGE_SYM_CLASS_NULL 0
#define IMAGE_SYM_CLASS_AUTOMATIC 1
#define IMAGE_SYM_CLASS_EXTERNAL 2
#define IMAGE_SYM_CLASS_STATIC 3
#define IMAGE_SYM_CLASS_REGISTER 4
#define IMAGE_SYM_CLASS_EXTERNAL_DEF 5
#define IMAGE_SYM_CLASS_LABEL 6
#define IMAGE_SYM_CLASS_UNDEFINED_LABEL 7
#define IMAGE_SYM_CLASS_MEMBER_OF_STRUCT 8
#define IMAGE_SYM_CLASS_ARGUMENT 9
#define IMAGE_SYM_CLASS_STRUCT_TAG 10
#define IMAGE_SYM_CLASS_MEMBER_OF_UNION 11
#define IMAGE_SYM_CLASS_UNION_TAG 12
#define IMAGE_SYM_CLASS_TYPE_DEFINITION 13
#define IMAGE_SYM_CLASS_UNDEFINED_STATIC 14
#define IMAGE_SYM_CLASS_ENUM_TAG 15
#define IMAGE_SYM_CLASS_MEMBER_OF_ENUM 16
#define IMAGE_SYM_CLASS_REGISTER_PARAM 17
#define IMAGE_SYM_CLASS_BIT_FIELD 18
#define IMAGE_SYM_CLASS_FAR_EXTERNAL 68 /* Not in PECOFF v8 spec */
#define IMAGE_SYM_CLASS_BLOCK 100
#define IMAGE_SYM_CLASS_FUNCTION 101
#define IMAGE_SYM_CLASS_END_OF_STRUCT 102
#define IMAGE_SYM_CLASS_FILE 103
#define IMAGE_SYM_CLASS_SECTION 104
#define IMAGE_SYM_CLASS_WEAK_EXTERNAL 105
#define IMAGE_SYM_CLASS_CLR_TOKEN 107

#define IMAGE_REL_I386_ABSOLUTE	0x0000
#define IMAGE_REL_I386_DIR16	0x0001
#define IMAGE_REL_I386_REL16	0x0002
#define IMAGE_REL_I386_DIR32	0x0006
#define IMAGE_REL_I386_DIR32NB	0x0007
#define IMAGE_REL_I386_SEG12	0x0009
#define IMAGE_REL_I386_SECTION	0x000A
#define IMAGE_REL_I386_SECREL	0x000B
#define IMAGE_REL_I386_TOKEN	0x000C
#define IMAGE_REL_I386_SECREL7	0x000D
#define IMAGE_REL_I386_REL32	0x0014

#define IMAGE_REL_AMD64_ABSOLUTE 0x0000
#define IMAGE_REL_AMD64_ADDR64 0x0001 // R_X86_64_64
#define IMAGE_REL_AMD64_ADDR32 0x0002 // R_X86_64_PC32
#define IMAGE_REL_AMD64_ADDR32NB 0x0003
#define IMAGE_REL_AMD64_REL32 0x0004 
#define IMAGE_REL_AMD64_REL32_1 0x0005
#define IMAGE_REL_AMD64_REL32_2 0x0006
#define IMAGE_REL_AMD64_REL32_3 0x0007
#define IMAGE_REL_AMD64_REL32_4 0x0008
#define IMAGE_REL_AMD64_REL32_5 0x0009
#define IMAGE_REL_AMD64_SECTION 0x000A
#define IMAGE_REL_AMD64_SECREL 0x000B
#define IMAGE_REL_AMD64_SECREL7 0x000C
#define IMAGE_REL_AMD64_TOKEN 0x000D
#define IMAGE_REL_AMD64_SREL32 0x000E
#define IMAGE_REL_AMD64_PAIR 0x000F
#define IMAGE_REL_AMD64_SSPAN32 0x0010

typedef struct PeSym PeSym;
typedef struct PeSect PeSect;
typedef struct PeObj PeObj;

struct PeSym {
	char* name;
	uint32 value;
	uint16 sectnum;
	uint16 type;
	uint8 sclass;
	uint8 aux;
	LSym* sym;
};

struct PeSect {
	char* name;
	uchar* base;
	uint64 size;
	LSym* sym;
	IMAGE_SECTION_HEADER sh;
};

struct PeObj {
	Biobuf	*f;
	char	*name;
	uint32 base;
	
	PeSect	*sect;
	uint	nsect;
	PeSym	*pesym;
	uint npesym;
	
	IMAGE_FILE_HEADER fh;
	char* snames;
};

static int map(PeObj *obj, PeSect *sect);
static int issect(PeSym *s);
static int readsym(PeObj *obj, int i, PeSym **sym);

void
ldpe(Biobuf *f, char *pkg, int64 len, char *pn)
{
	char *name;
	int32 base;
	uint32 l;
	int i, j, numaux;
	PeObj *obj;
	PeSect *sect, *rsect;
	IMAGE_SECTION_HEADER sh;
	uchar symbuf[18];
	LSym *s;
	Reloc *r, *rp;
	PeSym *sym;

	USED(len);
	if(debug['v'])
		Bprint(&bso, "%5.2f ldpe %s\n", cputime(), pn);
	
	sect = nil;
	ctxt->version++;
	base = Boffset(f);
	
	obj = mal(sizeof *obj);
	obj->f = f;
	obj->base = base;
	obj->name = pn;
	// read header
	if(Bread(f, &obj->fh, sizeof obj->fh) != sizeof obj->fh)
		goto bad;
	// load section list
	obj->sect = mal(obj->fh.NumberOfSections*sizeof obj->sect[0]);
	obj->nsect = obj->fh.NumberOfSections;
	for(i=0; i < obj->fh.NumberOfSections; i++) {
		if(Bread(f, &obj->sect[i].sh, sizeof sh) != sizeof sh)
			goto bad;
		obj->sect[i].size = obj->sect[i].sh.SizeOfRawData;
		obj->sect[i].name = (char*)obj->sect[i].sh.Name;
		// TODO return error if found .cormeta
	}
	// load string table
	Bseek(f, base+obj->fh.PointerToSymbolTable+sizeof(symbuf)*obj->fh.NumberOfSymbols, 0);
	if(Bread(f, symbuf, 4) != 4) 
		goto bad;
	l = le32(symbuf);
	obj->snames = mal(l);
	Bseek(f, base+obj->fh.PointerToSymbolTable+sizeof(symbuf)*obj->fh.NumberOfSymbols, 0);
	if(Bread(f, obj->snames, l) != l)
		goto bad;
	// rewrite section names if they start with /
	for(i=0; i < obj->fh.NumberOfSections; i++) {
		if(obj->sect[i].name == nil)
			continue;
		if(obj->sect[i].name[0] != '/')
			continue;
		l = atoi(obj->sect[i].name + 1);
		obj->sect[i].name = (char*)&obj->snames[l];
	}
	// read symbols
	obj->pesym = mal(obj->fh.NumberOfSymbols*sizeof obj->pesym[0]);
	obj->npesym = obj->fh.NumberOfSymbols;
	Bseek(f, base+obj->fh.PointerToSymbolTable, 0);
	for(i=0; i<obj->fh.NumberOfSymbols; i+=numaux+1) {
		Bseek(f, base+obj->fh.PointerToSymbolTable+sizeof(symbuf)*i, 0);
		if(Bread(f, symbuf, sizeof symbuf) != sizeof symbuf)
			goto bad;
		
		if((symbuf[0] == 0) && (symbuf[1] == 0) &&
			 (symbuf[2] == 0) && (symbuf[3] == 0)) {
			l = le32(&symbuf[4]);
			obj->pesym[i].name = (char*)&obj->snames[l];
		} else { // sym name length <= 8
			obj->pesym[i].name = mal(9);
			strncpy(obj->pesym[i].name, (char*)symbuf, 8);
			obj->pesym[i].name[8] = 0;
		}
		obj->pesym[i].value = le32(&symbuf[8]);
		obj->pesym[i].sectnum = le16(&symbuf[12]);
		obj->pesym[i].sclass = symbuf[16];
		obj->pesym[i].aux = symbuf[17];
		obj->pesym[i].type = le16(&symbuf[14]);
		numaux = obj->pesym[i].aux; 
		if (numaux < 0) 
			numaux = 0;
	}
	// create symbols for mapped sections
	for(i=0; i<obj->nsect; i++) {
		sect = &obj->sect[i];
		if(sect->sh.Characteristics&IMAGE_SCN_MEM_DISCARDABLE)
			continue;

		if((sect->sh.Characteristics&(IMAGE_SCN_CNT_CODE|IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_CNT_UNINITIALIZED_DATA)) == 0) {
			// This has been seen for .idata sections, which we
			// want to ignore.  See issues 5106 and 5273.
			continue;
		}

		if(map(obj, sect) < 0)
			goto bad;
		
		name = smprint("%s(%s)", pkg, sect->name);
		s = linklookup(ctxt, name, ctxt->version);
		free(name);
		switch(sect->sh.Characteristics&(IMAGE_SCN_CNT_UNINITIALIZED_DATA|IMAGE_SCN_CNT_INITIALIZED_DATA|
			IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE|IMAGE_SCN_CNT_CODE|IMAGE_SCN_MEM_EXECUTE)) {
			case IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_MEM_READ: //.rdata
				s->type = SRODATA;
				break;
			case IMAGE_SCN_CNT_UNINITIALIZED_DATA|IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE: //.bss
				s->type = SNOPTRBSS;
				break;
			case IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE: //.data
				s->type = SNOPTRDATA;
				break;
			case IMAGE_SCN_CNT_CODE|IMAGE_SCN_MEM_EXECUTE|IMAGE_SCN_MEM_READ: //.text
				s->type = STEXT;
				break;
			default:
				werrstr("unexpected flags %#08ux for PE section %s", sect->sh.Characteristics, sect->name);
				goto bad;
		}
		s->p = sect->base;
		s->np = sect->size;
		s->size = sect->size;
		sect->sym = s;
		if(strcmp(sect->name, ".rsrc") == 0)
			setpersrc(sect->sym);
	}
	
	// load relocations
	for(i=0; i<obj->nsect; i++) {
		rsect = &obj->sect[i];
		if(rsect->sym == 0 || rsect->sh.NumberOfRelocations == 0)
			continue;
		if(rsect->sh.Characteristics&IMAGE_SCN_MEM_DISCARDABLE)
			continue;
		if((sect->sh.Characteristics&(IMAGE_SCN_CNT_CODE|IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_CNT_UNINITIALIZED_DATA)) == 0) {
			// This has been seen for .idata sections, which we
			// want to ignore.  See issues 5106 and 5273.
			continue;
		}
		r = mal(rsect->sh.NumberOfRelocations*sizeof r[0]);
		Bseek(f, obj->base+rsect->sh.PointerToRelocations, 0);
		for(j=0; j<rsect->sh.NumberOfRelocations; j++) {
			rp = &r[j];
			if(Bread(f, symbuf, 10) != 10)
				goto bad;
			
			uint32 rva, symindex;
			uint16 type;
			rva = le32(&symbuf[0]);
			symindex = le32(&symbuf[4]);
			type = le16(&symbuf[8]);
			if(readsym(obj, symindex, &sym) < 0)
				goto bad;
			if(sym->sym == nil) {
				werrstr("reloc of invalid sym %s idx=%d type=%d", sym->name, symindex, sym->type);
				goto bad;
			}
			rp->sym = sym->sym;
			rp->siz = 4;
			rp->off = rva;
			switch(type) {
				default:
					diag("%s: unknown relocation type %d;", pn, type);
				case IMAGE_REL_I386_REL32:
				case IMAGE_REL_AMD64_REL32:
				case IMAGE_REL_AMD64_ADDR32: // R_X86_64_PC32
				case IMAGE_REL_AMD64_ADDR32NB:
					rp->type = R_PCREL;
					rp->add = (int32)le32(rsect->base+rp->off);
					break;
				case IMAGE_REL_I386_DIR32NB:
				case IMAGE_REL_I386_DIR32:
					rp->type = R_ADDR;
					// load addend from image
					rp->add = (int32)le32(rsect->base+rp->off);
					break;
				case IMAGE_REL_AMD64_ADDR64: // R_X86_64_64
					rp->siz = 8;
					rp->type = R_ADDR;
					// load addend from image
					rp->add = le64(rsect->base+rp->off);
					break;
			}
			// ld -r could generate multiple section symbols for the
			// same section but with different values, we have to take
			// that into account
			if(issect(&obj->pesym[symindex]))
				rp->add += obj->pesym[symindex].value;
		}
		qsort(r, rsect->sh.NumberOfRelocations, sizeof r[0], rbyoff);
		
		s = rsect->sym;
		s->r = r;
		s->nr = rsect->sh.NumberOfRelocations;
	}

	// enter sub-symbols into symbol table.
	for(i=0; i<obj->npesym; i++) {
		if(obj->pesym[i].name == 0)
			continue;
		if(issect(&obj->pesym[i]))
			continue;
		if(obj->pesym[i].sectnum > 0) {
			sect = &obj->sect[obj->pesym[i].sectnum-1];
			if(sect->sym == 0)
				continue;
		}
		if(readsym(obj, i, &sym) < 0)
			goto bad;
	
		s = sym->sym;
		if(sym->sectnum == 0) {// extern
			if(s->type == SDYNIMPORT)
				s->plt = -2; // flag for dynimport in PE object files.
			if (s->type == SXREF && sym->value > 0) {// global data
				s->type = SNOPTRDATA;
				s->size = sym->value;
			}
			continue;
		} else if (sym->sectnum > 0) {
			sect = &obj->sect[sym->sectnum-1];
			if(sect->sym == 0)
				diag("%s: %s sym == 0!", pn, s->name);
		} else {
			diag("%s: %s sectnum < 0!", pn, s->name);
		}

		if(sect == nil) 
			return;

		if(s->outer != S) {
			if(s->dupok)
				continue;
			diag("%s: duplicate symbol reference: %s in both %s and %s", pn, s->name, s->outer->name, sect->sym->name);
			errorexit();
		}
		s->sub = sect->sym->sub;
		sect->sym->sub = s;
		s->type = sect->sym->type | SSUB;
		s->value = sym->value;
		s->size = 4;
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

	return;
bad:
	diag("%s: malformed pe file: %r", pn);
}

static int
map(PeObj *obj, PeSect *sect)
{
	if(sect->base != nil)
		return 0;

	sect->base = mal(sect->sh.SizeOfRawData);
	if(sect->sh.PointerToRawData == 0) // .bss doesn't have data in object file
		return 0;
	werrstr("short read");
	if(Bseek(obj->f, obj->base+sect->sh.PointerToRawData, 0) < 0 || 
			Bread(obj->f, sect->base, sect->sh.SizeOfRawData) != sect->sh.SizeOfRawData)
		return -1;
	
	return 0;
}

static int
issect(PeSym *s)
{
	return s->sclass == IMAGE_SYM_CLASS_STATIC && s->type == 0 && s->name[0] == '.';
}

static int
readsym(PeObj *obj, int i, PeSym **y)
{
	LSym *s;
	PeSym *sym;
	char *name, *p;

	if(i >= obj->npesym || i < 0) {
		werrstr("invalid pe symbol index");
		return -1;
	}

	sym = &obj->pesym[i];
	*y = sym;
	
	if(issect(sym))
		name = obj->sect[sym->sectnum-1].sym->name;
	else {
		name = sym->name;
		if(strncmp(name, "__imp_", 6) == 0)
			name = &name[6]; // __imp_Name => Name
		if(thechar == '8' && name[0] == '_')
			name = &name[1]; // _Name => Name
	}
	// remove last @XXX
	p = strchr(name, '@');
	if(p)
		*p = 0;
	
	switch(sym->type) {
	default:
		werrstr("%s: invalid symbol type %d", sym->name, sym->type);
		return -1;
	case IMAGE_SYM_DTYPE_FUNCTION:
	case IMAGE_SYM_DTYPE_NULL:
		switch(sym->sclass) {
		case IMAGE_SYM_CLASS_EXTERNAL: //global
			s = linklookup(ctxt, name, 0);
			break;
		case IMAGE_SYM_CLASS_NULL:
		case IMAGE_SYM_CLASS_STATIC:
		case IMAGE_SYM_CLASS_LABEL:
			s = linklookup(ctxt, name, ctxt->version);
			s->dupok = 1;
			break;
		default:
			werrstr("%s: invalid symbol binding %d", sym->name, sym->sclass);
			return -1;
		}
		break;
	}

	if(s != nil && s->type == 0 && !(sym->sclass == IMAGE_SYM_CLASS_STATIC && sym->value == 0))
		s->type = SXREF;
	if(strncmp(sym->name, "__imp_", 6) == 0)
		s->got = -2; // flag for __imp_
	sym->sym = s;

	return 0;
}
