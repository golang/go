// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO/NICETOHAVE:
//   - eliminate DW_CLS_ if not used
//   - package info in compilation units
//   - assign global variables and types to their packages
//   - gdb uses c syntax, meaning clumsy quoting is needed for go identifiers. eg
//     ptype struct '[]uint8' and qualifiers need to be quoted away
//   - lexical scoping is lost, so gdb gets confused as to which 'main.i' you mean.
//   - file:line info for variables
//   - make strings a typedef so prettyprinters can see the underlying string type
//
#include	"l.h"
#include	"lib.h"
#include	"../ld/dwarf.h"
#include	"../ld/dwarf_defs.h"
#include	"../ld/elf.h"
#include	"../ld/macho.h"
#include	"../ld/pe.h"
#include	"../../runtime/typekind.h"

/*
 * Offsets and sizes of the debug_* sections in the cout file.
 */

static vlong abbrevo;
static vlong abbrevsize;
static LSym*  abbrevsym;
static vlong abbrevsympos;
static vlong lineo;
static vlong linesize;
static LSym*  linesym;
static vlong linesympos;
static vlong infoo;	// also the base for DWDie->offs and reference attributes.
static vlong infosize;
static LSym*  infosym;
static vlong infosympos;
static vlong frameo;
static vlong framesize;
static LSym*  framesym;
static vlong framesympos;
static vlong pubnameso;
static vlong pubnamessize;
static vlong pubtypeso;
static vlong pubtypessize;
static vlong arangeso;
static vlong arangessize;
static vlong gdbscripto;
static vlong gdbscriptsize;

static LSym *infosec;
static vlong inforeloco;
static vlong inforelocsize;

static LSym *arangessec;
static vlong arangesreloco;
static vlong arangesrelocsize;

static LSym *linesec;
static vlong linereloco;
static vlong linerelocsize;

static LSym *framesec;
static vlong framereloco;
static vlong framerelocsize;

static char  gdbscript[1024];

/*
 *  Basic I/O
 */

static void
addrput(vlong addr)
{
	switch(PtrSize) {
	case 4:
		LPUT(addr);
		break;
	case 8:
		VPUT(addr);
		break;
	}
}

static int
uleb128enc(uvlong v, char* dst)
{
	uint8 c, len;

	len = 0;
	do {
		c = v & 0x7f;
		v >>= 7;
		if (v)
			c |= 0x80;
		if (dst)
			*dst++ = c;
		len++;
	} while (c & 0x80);
	return len;
};

static int
sleb128enc(vlong v, char *dst)
{
	uint8 c, s, len;

	len = 0;
	do {
		c = v & 0x7f;
		s = v & 0x40;
		v >>= 7;
		if ((v != -1 || !s) && (v != 0 || s))
			c |= 0x80;
		if (dst)
			*dst++ = c;
		len++;
	} while(c & 0x80);
	return len;
}

static void
uleb128put(vlong v)
{
	char buf[10];
	strnput(buf, uleb128enc(v, buf));
}

static void
sleb128put(vlong v)
{
	char buf[10];
	strnput(buf, sleb128enc(v, buf));
}

/*
 * Defining Abbrevs.  This is hardcoded, and there will be
 * only a handful of them.  The DWARF spec places no restriction on
 * the ordering of attributes in the Abbrevs and DIEs, and we will
 * always write them out in the order of declaration in the abbrev.
 */
typedef struct DWAttrForm DWAttrForm;
struct DWAttrForm {
	uint16 attr;
	uint8 form;
};

// Go-specific type attributes.
enum {
	DW_AT_go_kind = 0x2900,
	DW_AT_go_key = 0x2901,
	DW_AT_go_elem = 0x2902,

	DW_AT_internal_location = 253,	 // params and locals; not emitted
};

// Index into the abbrevs table below.
// Keep in sync with ispubname() and ispubtype() below.
// ispubtype considers >= NULLTYPE public
enum
{
	DW_ABRV_NULL,
	DW_ABRV_COMPUNIT,
	DW_ABRV_FUNCTION,
	DW_ABRV_VARIABLE,
	DW_ABRV_AUTO,
	DW_ABRV_PARAM,
	DW_ABRV_STRUCTFIELD,
	DW_ABRV_FUNCTYPEPARAM,
	DW_ABRV_DOTDOTDOT,
	DW_ABRV_ARRAYRANGE,
	DW_ABRV_NULLTYPE,
	DW_ABRV_BASETYPE,
	DW_ABRV_ARRAYTYPE,
	DW_ABRV_CHANTYPE,
	DW_ABRV_FUNCTYPE,
	DW_ABRV_IFACETYPE,
	DW_ABRV_MAPTYPE,
	DW_ABRV_PTRTYPE,
	DW_ABRV_BARE_PTRTYPE, // only for void*, no DW_AT_type attr to please gdb 6.
	DW_ABRV_SLICETYPE,
	DW_ABRV_STRINGTYPE,
	DW_ABRV_STRUCTTYPE,
	DW_ABRV_TYPEDECL,
	DW_NABRV
};

typedef struct DWAbbrev DWAbbrev;
static struct DWAbbrev {
	uint8 tag;
	uint8 children;
	DWAttrForm attr[30];
} abbrevs[DW_NABRV] = {
	/* The mandatory DW_ABRV_NULL entry. */
	{ 0 },
	/* COMPUNIT */
	{
		DW_TAG_compile_unit, DW_CHILDREN_yes,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_language,	 DW_FORM_data1,
		DW_AT_low_pc,	 DW_FORM_addr,
		DW_AT_high_pc,	 DW_FORM_addr,
		DW_AT_stmt_list, DW_FORM_data4,
		0, 0
	},
	/* FUNCTION */
	{
		DW_TAG_subprogram, DW_CHILDREN_yes,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_low_pc,	 DW_FORM_addr,
		DW_AT_high_pc,	 DW_FORM_addr,
		DW_AT_external,	 DW_FORM_flag,
		0, 0
	},
	/* VARIABLE */
	{
		DW_TAG_variable, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_location,	 DW_FORM_block1,
		DW_AT_type,	 DW_FORM_ref_addr,
		DW_AT_external,	 DW_FORM_flag,
		0, 0
	},
	/* AUTO */
	{
		DW_TAG_variable, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_location,	 DW_FORM_block1,
		DW_AT_type,	 DW_FORM_ref_addr,
		0, 0
	},
	/* PARAM */
	{
		DW_TAG_formal_parameter, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_location,	 DW_FORM_block1,
		DW_AT_type,	 DW_FORM_ref_addr,
		0, 0
	},
	/* STRUCTFIELD */
	{
		DW_TAG_member,	DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		DW_AT_data_member_location, DW_FORM_block1,
		DW_AT_type,	 DW_FORM_ref_addr,
		0, 0
	},
	/* FUNCTYPEPARAM */
	{
		DW_TAG_formal_parameter, DW_CHILDREN_no,
		// No name!
		DW_AT_type,	 DW_FORM_ref_addr,
		0, 0
	},

	/* DOTDOTDOT */
	{
		DW_TAG_unspecified_parameters, DW_CHILDREN_no,
		0, 0
	},
	/* ARRAYRANGE */
	{
		DW_TAG_subrange_type, DW_CHILDREN_no,
		// No name!
		DW_AT_type,	 DW_FORM_ref_addr,
		DW_AT_count, DW_FORM_udata,
		0, 0
	},

	// Below here are the types considered public by ispubtype
	/* NULLTYPE */
	{
		DW_TAG_unspecified_type, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		0, 0
	},
	/* BASETYPE */
	{
		DW_TAG_base_type, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_encoding,	 DW_FORM_data1,
		DW_AT_byte_size, DW_FORM_data1,
		DW_AT_go_kind, DW_FORM_data1,
		0, 0
	},
	/* ARRAYTYPE */
	// child is subrange with upper bound
	{
		DW_TAG_array_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		DW_AT_byte_size, DW_FORM_udata,
		DW_AT_go_kind, DW_FORM_data1,
		0, 0
	},

	/* CHANTYPE */
	{
		DW_TAG_typedef, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		DW_AT_go_kind, DW_FORM_data1,
		DW_AT_go_elem, DW_FORM_ref_addr,
		0, 0
	},

	/* FUNCTYPE */
	{
		DW_TAG_subroutine_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
//		DW_AT_type,	DW_FORM_ref_addr,
		DW_AT_go_kind, DW_FORM_data1,
		0, 0
	},

	/* IFACETYPE */
	{
		DW_TAG_typedef, DW_CHILDREN_yes,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		DW_AT_go_kind, DW_FORM_data1,
		0, 0
	},

	/* MAPTYPE */
	{
		DW_TAG_typedef, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		DW_AT_go_kind, DW_FORM_data1,
		DW_AT_go_key, DW_FORM_ref_addr,
		DW_AT_go_elem, DW_FORM_ref_addr,
		0, 0
	},

	/* PTRTYPE */
	{
		DW_TAG_pointer_type, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		DW_AT_go_kind, DW_FORM_data1,
		0, 0
	},
	/* BARE_PTRTYPE */
	{
		DW_TAG_pointer_type, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		0, 0
	},

	/* SLICETYPE */
	{
		DW_TAG_structure_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
		DW_AT_byte_size, DW_FORM_udata,
		DW_AT_go_kind, DW_FORM_data1,
		DW_AT_go_elem, DW_FORM_ref_addr,
		0, 0
	},

	/* STRINGTYPE */
	{
		DW_TAG_structure_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
		DW_AT_byte_size, DW_FORM_udata,
		DW_AT_go_kind, DW_FORM_data1,
		0, 0
	},

	/* STRUCTTYPE */
	{
		DW_TAG_structure_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
		DW_AT_byte_size, DW_FORM_udata,
		DW_AT_go_kind, DW_FORM_data1,
		0, 0
	},

	/* TYPEDECL */
	{
		DW_TAG_typedef, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		0, 0
	},
};

static void
writeabbrev(void)
{
	int i, j;
	DWAttrForm *f;

	abbrevo = cpos();
	for (i = 1; i < DW_NABRV; i++) {
		// See section 7.5.3
		uleb128put(i);
		uleb128put(abbrevs[i].tag);
		cput(abbrevs[i].children);
		for(j=0; j<nelem(abbrevs[i].attr); j++) {
			f = &abbrevs[i].attr[j];
			uleb128put(f->attr);
			uleb128put(f->form);
			if(f->attr == 0)
				break;
		}
	}
	cput(0);
	abbrevsize = cpos() - abbrevo;
}

/*
 * Debugging Information Entries and their attributes.
 */

enum
{
	HASHSIZE = 107
};

static uint32
hashstr(char* s)
{
	uint32 h;

	h = 0;
	while (*s)
		h = h+h+h + *s++;
	return h % HASHSIZE;
}

// For DW_CLS_string and _block, value should contain the length, and
// data the data, for _reference, value is 0 and data is a DWDie* to
// the referenced instance, for all others, value is the whole thing
// and data is null.

typedef struct DWAttr DWAttr;
struct DWAttr {
	DWAttr *link;
	uint16 atr;  // DW_AT_
	uint8 cls;  // DW_CLS_
	vlong value;
	char *data;
};

typedef struct DWDie DWDie;
struct DWDie {
	int abbrev;
	DWDie *link;
	DWDie *child;
	DWAttr *attr;
	// offset into .debug_info section, i.e relative to
	// infoo. only valid after call to putdie()
	vlong offs;
	DWDie **hash;  // optional index of children by name, enabled by mkindex()
	DWDie *hlink;  // bucket chain in parent's index
};

/*
 * Root DIEs for compilation units, types and global variables.
 */

static DWDie dwroot;
static DWDie dwtypes;
static DWDie dwglobals;

static DWAttr*
newattr(DWDie *die, uint16 attr, int cls, vlong value, char *data)
{
	DWAttr *a;

	a = mal(sizeof *a);
	a->link = die->attr;
	die->attr = a;
	a->atr = attr;
	a->cls = cls;
	a->value = value;
	a->data = data;
	return a;
}

// Each DIE (except the root ones) has at least 1 attribute: its
// name. getattr moves the desired one to the front so
// frequently searched ones are found faster.
static DWAttr*
getattr(DWDie *die, uint16 attr)
{
	DWAttr *a, *b;

	if (die->attr->atr == attr)
		return die->attr;

	a = die->attr;
	b = a->link;
	while (b != nil) {
		if (b->atr == attr) {
			a->link = b->link;
			b->link = die->attr;
			die->attr = b;
			return b;
		}
		a = b;
		b = b->link;
	}
	return nil;
}

// Every DIE has at least a DW_AT_name attribute (but it will only be
// written out if it is listed in the abbrev).	If its parent is
// keeping an index, the new DIE will be inserted there.
static DWDie*
newdie(DWDie *parent, int abbrev, char *name)
{
	DWDie *die;
	int h;

	die = mal(sizeof *die);
	die->abbrev = abbrev;
	die->link = parent->child;
	parent->child = die;

	newattr(die, DW_AT_name, DW_CLS_STRING, strlen(name), name);

	if (parent->hash) {
		h = hashstr(name);
		die->hlink = parent->hash[h];
		parent->hash[h] = die;
	}

	return die;
}

static void
mkindex(DWDie *die)
{
	die->hash = mal(HASHSIZE * sizeof(DWDie*));
}

static DWDie*
walktypedef(DWDie *die)
{
	DWAttr *attr;

	// Resolve typedef if present.
	if (die->abbrev == DW_ABRV_TYPEDECL) {
		for (attr = die->attr; attr; attr = attr->link) {
			if (attr->atr == DW_AT_type && attr->cls == DW_CLS_REFERENCE && attr->data != nil) {
				return (DWDie*)attr->data;
			}
		}
	}
	return die;
}

// Find child by AT_name using hashtable if available or linear scan
// if not.
static DWDie*
find(DWDie *die, char* name)
{
	DWDie *a, *b, *die2;
	int h;

top:
	if (die->hash == nil) {
		for (a = die->child; a != nil; a = a->link)
			if (strcmp(name, getattr(a, DW_AT_name)->data) == 0)
				return a;
		goto notfound;
	}

	h = hashstr(name);
	a = die->hash[h];

	if (a == nil)
		goto notfound;


	if (strcmp(name, getattr(a, DW_AT_name)->data) == 0)
		return a;

	// Move found ones to head of the list.
	b = a->hlink;
	while (b != nil) {
		if (strcmp(name, getattr(b, DW_AT_name)->data) == 0) {
			a->hlink = b->hlink;
			b->hlink = die->hash[h];
			die->hash[h] = b;
			return b;
		}
		a = b;
		b = b->hlink;
	}

notfound:
	die2 = walktypedef(die);
	if(die2 != die) {
		die = die2;
		goto top;
	}

	return nil;
}

static DWDie*
find_or_diag(DWDie *die, char* name)
{
	DWDie *r;
	r = find(die, name);
	if (r == nil) {
		diag("dwarf find: %s %p has no %s", getattr(die, DW_AT_name)->data, die, name);
		errorexit();
	}
	return r;
}

static void
adddwarfrel(LSym* sec, LSym* sym, vlong offsetbase, int siz, vlong addend)
{
	Reloc *r;

	r = addrel(sec);
	r->sym = sym;
	r->xsym = sym;
	r->off = cpos() - offsetbase;
	r->siz = siz;
	r->type = R_ADDR;
	r->add = addend;
	r->xadd = addend;
	if(iself && thechar == '6')
		addend = 0;
	switch(siz) {
	case 4:
		LPUT(addend);
		break;
	case 8:
		VPUT(addend);
		break;
	default:
		diag("bad size in adddwarfrel");
		break;
	}
}

static DWAttr*
newrefattr(DWDie *die, uint16 attr, DWDie* ref)
{
	if (ref == nil)
		return nil;
	return newattr(die, attr, DW_CLS_REFERENCE, 0, (char*)ref);
}

static int fwdcount;

static void
putattr(int abbrev, int form, int cls, vlong value, char *data)
{
	vlong off;

	switch(form) {
	case DW_FORM_addr:	// address
		if(linkmode == LinkExternal) {
			value -= ((LSym*)data)->value;
			adddwarfrel(infosec, (LSym*)data, infoo, PtrSize, value);
			break;
		}
		addrput(value);
		break;

	case DW_FORM_block1:	// block
		if(cls == DW_CLS_ADDRESS) {
			cput(1+PtrSize);
			cput(DW_OP_addr);
			if(linkmode == LinkExternal) {
				value -= ((LSym*)data)->value;
				adddwarfrel(infosec, (LSym*)data, infoo, PtrSize, value);
				break;
			}
			addrput(value);
			break;
		}
		value &= 0xff;
		cput(value);
		while(value--)
			cput(*data++);
		break;

	case DW_FORM_block2:	// block
		value &= 0xffff;
		WPUT(value);
		while(value--)
			cput(*data++);
		break;

	case DW_FORM_block4:	// block
		value &= 0xffffffff;
		LPUT(value);
		while(value--)
			cput(*data++);
		break;

	case DW_FORM_block:	// block
		uleb128put(value);
		while(value--)
			cput(*data++);
		break;

	case DW_FORM_data1:	// constant
		cput(value);
		break;

	case DW_FORM_data2:	// constant
		WPUT(value);
		break;

	case DW_FORM_data4:	// constant, {line,loclist,mac,rangelist}ptr
		if(linkmode == LinkExternal && cls == DW_CLS_PTR) {
			adddwarfrel(infosec, linesym, infoo, 4, value);
			break;
		}
		LPUT(value);
		break;

	case DW_FORM_data8:	// constant, {line,loclist,mac,rangelist}ptr
		VPUT(value);
		break;

	case DW_FORM_sdata:	// constant
		sleb128put(value);
		break;

	case DW_FORM_udata:	// constant
		uleb128put(value);
		break;

	case DW_FORM_string:	// string
		strnput(data, value+1);
		break;

	case DW_FORM_flag:	// flag
		cput(value?1:0);
		break;

	case DW_FORM_ref_addr:	// reference to a DIE in the .info section
		// In DWARF 2 (which is what we claim to generate),
		// the ref_addr is the same size as a normal address.
		// In DWARF 3 it is always 32 bits, unless emitting a large
		// (> 4 GB of debug info aka "64-bit") unit, which we don't implement.
		if (data == nil) {
			diag("dwarf: null reference in %d", abbrev);
			if(PtrSize == 8)
				VPUT(0); // invalid dwarf, gdb will complain.
			else
				LPUT(0); // invalid dwarf, gdb will complain.
		} else {
			off = ((DWDie*)data)->offs;
			if (off == 0)
				fwdcount++;
			if(linkmode == LinkExternal) {
				adddwarfrel(infosec, infosym, infoo, PtrSize, off);
				break;
			}
			addrput(off);
		}
		break;

	case DW_FORM_ref1:	// reference within the compilation unit
	case DW_FORM_ref2:	// reference
	case DW_FORM_ref4:	// reference
	case DW_FORM_ref8:	// reference
	case DW_FORM_ref_udata:	// reference

	case DW_FORM_strp:	// string
	case DW_FORM_indirect:	// (see Section 7.5.3)
	default:
		diag("dwarf: unsupported attribute form %d / class %d", form, cls);
		errorexit();
	}
}

// Note that we can (and do) add arbitrary attributes to a DIE, but
// only the ones actually listed in the Abbrev will be written out.
static void
putattrs(int abbrev, DWAttr* attr)
{
	DWAttrForm* af;
	DWAttr *ap;

	for(af = abbrevs[abbrev].attr; af->attr; af++) {
		for(ap=attr; ap; ap=ap->link) {
			if(ap->atr == af->attr) {
				putattr(abbrev, af->form,
					ap->cls,
					ap->value,
					ap->data);
				goto done;
			}
		}
		putattr(abbrev, af->form, 0, 0, nil);
	done:;
	}
}

static void putdie(DWDie* die);

static void
putdies(DWDie* die)
{
	for(; die; die = die->link)
		putdie(die);
}

static void
putdie(DWDie* die)
{
	die->offs = cpos() - infoo;
	uleb128put(die->abbrev);
	putattrs(die->abbrev, die->attr);
	if (abbrevs[die->abbrev].children) {
		putdies(die->child);
		cput(0);
	}
}

static void
reverselist(DWDie** list)
{
	DWDie *curr, *prev;

	curr = *list;
	prev = nil;
	while(curr != nil) {
		DWDie* next = curr->link;
		curr->link = prev;
		prev = curr;
		curr = next;
	}
	*list = prev;
}

static void
reversetree(DWDie** list)
{
	 DWDie *die;

	 reverselist(list);
	 for (die = *list; die != nil; die = die->link)
		 if (abbrevs[die->abbrev].children)
			 reversetree(&die->child);
}

static void
newmemberoffsetattr(DWDie *die, int32 offs)
{
	char block[10];
	int i;

	i = 0;
	block[i++] = DW_OP_plus_uconst;
	i += uleb128enc(offs, block+i);
	newattr(die, DW_AT_data_member_location, DW_CLS_BLOCK, i, mal(i));
	memmove(die->attr->data, block, i);
}

// GDB doesn't like DW_FORM_addr for DW_AT_location, so emit a
// location expression that evals to a const.
static void
newabslocexprattr(DWDie *die, vlong addr, LSym *sym)
{
	newattr(die, DW_AT_location, DW_CLS_ADDRESS, addr, (char*)sym);
}

static DWDie* defptrto(DWDie *dwtype);	// below

// Lookup predefined types
static LSym*
lookup_or_diag(char *n)
{
	LSym *s;

	s = linkrlookup(ctxt, n, 0);
	if (s == nil || s->size == 0) {
		diag("dwarf: missing type: %s", n);
		errorexit();
	}
	return s;
}

static void
dotypedef(DWDie *parent, char *name, DWDie *def)
{
	DWDie *die;

	// Only emit typedefs for real names.
	if(strncmp(name, "map[", 4) == 0)
		return;
	if(strncmp(name, "struct {", 8) == 0)
		return;
	if(strncmp(name, "chan ", 5) == 0)
		return;
	if(*name == '[' || *name == '*')
		return;
	if(def == nil)
		diag("dwarf: bad def in dotypedef");

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die = newdie(parent, DW_ABRV_TYPEDECL, name);
	newrefattr(die, DW_AT_type, def);
}

// Define gotype, for composite ones recurse into constituents.
static DWDie*
defgotype(LSym *gotype)
{
	DWDie *die, *fld;
	LSym *s;
	char *name, *f;
	uint8 kind;
	vlong bytesize;
	int i, nfields;

	if (gotype == nil)
		return find_or_diag(&dwtypes, "<unspecified>");

	if (strncmp("type.", gotype->name, 5) != 0) {
		diag("dwarf: type name doesn't start with \".type\": %s", gotype->name);
		return find_or_diag(&dwtypes, "<unspecified>");
	}
	name = gotype->name + 5;  // could also decode from Type.string

	die = find(&dwtypes, name);
	if (die != nil)
		return die;

	if (0 && debug['v'] > 2)
		print("new type: %Y\n", gotype);

	kind = decodetype_kind(gotype);
	bytesize = decodetype_size(gotype);

	switch (kind) {
	case KindBool:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name);
		newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_boolean, 0);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindInt:
	case KindInt8:
	case KindInt16:
	case KindInt32:
	case KindInt64:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name);
		newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_signed, 0);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindUint:
	case KindUint8:
	case KindUint16:
	case KindUint32:
	case KindUint64:
	case KindUintptr:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name);
		newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_unsigned, 0);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindFloat32:
	case KindFloat64:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name);
		newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_float, 0);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindComplex64:
	case KindComplex128:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name);
		newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_complex_float, 0);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindArray:
		die = newdie(&dwtypes, DW_ABRV_ARRAYTYPE, name);
		dotypedef(&dwtypes, name, die);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		s = decodetype_arrayelem(gotype);
		newrefattr(die, DW_AT_type, defgotype(s));
		fld = newdie(die, DW_ABRV_ARRAYRANGE, "range");
		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, DW_AT_count, DW_CLS_CONSTANT, decodetype_arraylen(gotype), 0);
		newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"));
		break;

	case KindChan:
		die = newdie(&dwtypes, DW_ABRV_CHANTYPE, name);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		s = decodetype_chanelem(gotype);
		newrefattr(die, DW_AT_go_elem, defgotype(s));
		break;

	case KindFunc:
		die = newdie(&dwtypes, DW_ABRV_FUNCTYPE, name);
		dotypedef(&dwtypes, name, die);
		newrefattr(die, DW_AT_type, find_or_diag(&dwtypes, "void"));
		nfields = decodetype_funcincount(gotype);
		for (i = 0; i < nfields; i++) {
			s = decodetype_funcintype(gotype, i);
			fld = newdie(die, DW_ABRV_FUNCTYPEPARAM, s->name+5);
			newrefattr(fld, DW_AT_type, defgotype(s));
		}
		if (decodetype_funcdotdotdot(gotype))
			newdie(die, DW_ABRV_DOTDOTDOT, "...");
		nfields = decodetype_funcoutcount(gotype);
		for (i = 0; i < nfields; i++) {
			s = decodetype_funcouttype(gotype, i);
			fld = newdie(die, DW_ABRV_FUNCTYPEPARAM, s->name+5);
			newrefattr(fld, DW_AT_type, defptrto(defgotype(s)));
		}
		break;

	case KindInterface:
		die = newdie(&dwtypes, DW_ABRV_IFACETYPE, name);
		dotypedef(&dwtypes, name, die);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		nfields = decodetype_ifacemethodcount(gotype);
		if (nfields == 0)
			s = lookup_or_diag("type.runtime.eface");
		else
			s = lookup_or_diag("type.runtime.iface");
		newrefattr(die, DW_AT_type, defgotype(s));
		break;

	case KindMap:
		die = newdie(&dwtypes, DW_ABRV_MAPTYPE, name);
		s = decodetype_mapkey(gotype);
		newrefattr(die, DW_AT_go_key, defgotype(s));
		s = decodetype_mapvalue(gotype);
		newrefattr(die, DW_AT_go_elem, defgotype(s));
		break;

	case KindPtr:
		die = newdie(&dwtypes, DW_ABRV_PTRTYPE, name);
		dotypedef(&dwtypes, name, die);
		s = decodetype_ptrelem(gotype);
		newrefattr(die, DW_AT_type, defgotype(s));
		break;

	case KindSlice:
		die = newdie(&dwtypes, DW_ABRV_SLICETYPE, name);
		dotypedef(&dwtypes, name, die);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		s = decodetype_arrayelem(gotype);
		newrefattr(die, DW_AT_go_elem, defgotype(s));
		break;

	case KindString:
		die = newdie(&dwtypes, DW_ABRV_STRINGTYPE, name);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindStruct:
		die = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, name);
		dotypedef(&dwtypes, name, die);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		nfields = decodetype_structfieldcount(gotype);
		for (i = 0; i < nfields; i++) {
			f = decodetype_structfieldname(gotype, i);
			s = decodetype_structfieldtype(gotype, i);
			if (f == nil)
				f = s->name + 5;	 // skip "type."
			fld = newdie(die, DW_ABRV_STRUCTFIELD, f);
			newrefattr(fld, DW_AT_type, defgotype(s));
			newmemberoffsetattr(fld, decodetype_structfieldoffs(gotype, i));
		}
		break;

	case KindUnsafePointer:
		die = newdie(&dwtypes, DW_ABRV_BARE_PTRTYPE, name);
		break;

	default:
		diag("dwarf: definition of unknown kind %d: %s", kind, gotype->name);
		die = newdie(&dwtypes, DW_ABRV_TYPEDECL, name);
		newrefattr(die, DW_AT_type, find_or_diag(&dwtypes, "<unspecified>"));
	}

	newattr(die, DW_AT_go_kind, DW_CLS_CONSTANT, kind, 0);

	return die;
}

// Find or construct *T given T.
static DWDie*
defptrto(DWDie *dwtype)
{
	char ptrname[1024];
	DWDie *die;

	snprint(ptrname, sizeof ptrname, "*%s", getattr(dwtype, DW_AT_name)->data);
	die = find(&dwtypes, ptrname);
	if (die == nil) {
		die = newdie(&dwtypes, DW_ABRV_PTRTYPE,
			     strcpy(mal(strlen(ptrname)+1), ptrname));
		newrefattr(die, DW_AT_type, dwtype);
	}
	return die;
}

// Copies src's children into dst. Copies attributes by value.
// DWAttr.data is copied as pointer only.  If except is one of
// the top-level children, it will not be copied.
static void
copychildrenexcept(DWDie *dst, DWDie *src, DWDie *except)
{
	DWDie *c;
	DWAttr *a;

	for (src = src->child; src != nil; src = src->link) {
		if(src == except)
			continue;
		c = newdie(dst, src->abbrev, getattr(src, DW_AT_name)->data);
		for (a = src->attr; a != nil; a = a->link)
			newattr(c, a->atr, a->cls, a->value, a->data);
		copychildrenexcept(c, src, nil);
	}
	reverselist(&dst->child);
}
static void
copychildren(DWDie *dst, DWDie *src)
{
	copychildrenexcept(dst, src, nil);
}

// Search children (assumed to have DW_TAG_member) for the one named
// field and set its DW_AT_type to dwtype
static void
substitutetype(DWDie *structdie, char *field, DWDie* dwtype)
{
	DWDie *child;
	DWAttr *a;

	child = find_or_diag(structdie, field);
	if (child == nil)
		return;

	a = getattr(child, DW_AT_type);
	if (a != nil)
		a->data = (char*) dwtype;
	else
		newrefattr(child, DW_AT_type, dwtype);
}

static void
synthesizestringtypes(DWDie* die)
{
	DWDie *prototype;

	prototype = walktypedef(defgotype(lookup_or_diag("type.runtime._string")));
	if (prototype == nil)
		return;

	for (; die != nil; die = die->link) {
		if (die->abbrev != DW_ABRV_STRINGTYPE)
			continue;
		copychildren(die, prototype);
	}
}

static void
synthesizeslicetypes(DWDie *die)
{
	DWDie *prototype, *elem;

	prototype = walktypedef(defgotype(lookup_or_diag("type.runtime.slice")));
	if (prototype == nil)
		return;

	for (; die != nil; die = die->link) {
		if (die->abbrev != DW_ABRV_SLICETYPE)
			continue;
		copychildren(die, prototype);
		elem = (DWDie*) getattr(die, DW_AT_go_elem)->data;
		substitutetype(die, "array", defptrto(elem));
	}
}

static char*
mkinternaltypename(char *base, char *arg1, char *arg2)
{
	char buf[1024];
	char *n;

	if (arg2 == nil)
		snprint(buf, sizeof buf, "%s<%s>", base, arg1);
	else
		snprint(buf, sizeof buf, "%s<%s,%s>", base, arg1, arg2);
	n = mal(strlen(buf) + 1);
	memmove(n, buf, strlen(buf));
	return n;
}

// synthesizemaptypes is way too closely married to runtime/hashmap.c
enum {
	MaxKeySize = 128,
	MaxValSize = 128,
	BucketSize = 8,
};

static void
synthesizemaptypes(DWDie *die)
{

	DWDie *hash, *bucket, *dwh, *dwhk, *dwhv, *dwhb, *keytype, *valtype, *fld;
	int indirect_key, indirect_val;
	int keysize, valsize;
	DWAttr *a;

	hash		= walktypedef(defgotype(lookup_or_diag("type.runtime.hmap")));
	bucket		= walktypedef(defgotype(lookup_or_diag("type.runtime.bmap")));

	if (hash == nil)
		return;

	for (; die != nil; die = die->link) {
		if (die->abbrev != DW_ABRV_MAPTYPE)
			continue;

		keytype = walktypedef((DWDie*) getattr(die, DW_AT_go_key)->data);
		valtype = walktypedef((DWDie*) getattr(die, DW_AT_go_elem)->data);

		// compute size info like hashmap.c does.
		a = getattr(keytype, DW_AT_byte_size);
		keysize = a ? a->value : PtrSize;  // We don't store size with Pointers
		a = getattr(valtype, DW_AT_byte_size);
		valsize = a ? a->value : PtrSize;
		indirect_key = 0;
		indirect_val = 0;
		if(keysize > MaxKeySize) {
			keysize = PtrSize;
			indirect_key = 1;
		}
		if(valsize > MaxValSize) {
			valsize = PtrSize;
			indirect_val = 1;
		}

		// Construct type to represent an array of BucketSize keys
		dwhk = newdie(&dwtypes, DW_ABRV_ARRAYTYPE,
			      mkinternaltypename("[]key",
						 getattr(keytype, DW_AT_name)->data, nil));
		newattr(dwhk, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize * keysize, 0);
		newrefattr(dwhk, DW_AT_type, indirect_key ? defptrto(keytype) : keytype);
		fld = newdie(dwhk, DW_ABRV_ARRAYRANGE, "size");
		newattr(fld, DW_AT_count, DW_CLS_CONSTANT, BucketSize, 0);
		newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"));
		
		// Construct type to represent an array of BucketSize values
		dwhv = newdie(&dwtypes, DW_ABRV_ARRAYTYPE, 
			      mkinternaltypename("[]val",
						 getattr(valtype, DW_AT_name)->data, nil));
		newattr(dwhv, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize * valsize, 0);
		newrefattr(dwhv, DW_AT_type, indirect_val ? defptrto(valtype) : valtype);
		fld = newdie(dwhv, DW_ABRV_ARRAYRANGE, "size");
		newattr(fld, DW_AT_count, DW_CLS_CONSTANT, BucketSize, 0);
		newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"));

		// Construct bucket<K,V>
		dwhb = newdie(&dwtypes, DW_ABRV_STRUCTTYPE,
			      mkinternaltypename("bucket",
						 getattr(keytype, DW_AT_name)->data,
						 getattr(valtype, DW_AT_name)->data));
		// Copy over all fields except the field "data" from the generic bucket.
		// "data" will be replaced with keys/values below.
		copychildrenexcept(dwhb, bucket, find(bucket, "data"));
		
		fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "keys");
		newrefattr(fld, DW_AT_type, dwhk);
		newmemberoffsetattr(fld, BucketSize);
		fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "values");
		newrefattr(fld, DW_AT_type, dwhv);
		newmemberoffsetattr(fld, BucketSize + BucketSize * keysize);
		fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "overflow");
		newrefattr(fld, DW_AT_type, defptrto(dwhb));
		newmemberoffsetattr(fld, BucketSize + BucketSize * (keysize + valsize));
		if(RegSize > PtrSize) {
			fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "pad");
			newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"));
			newmemberoffsetattr(fld, BucketSize + BucketSize * (keysize + valsize) + PtrSize);
		}
		newattr(dwhb, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize + BucketSize * keysize + BucketSize * valsize + RegSize, 0);

		// Construct hash<K,V>
		dwh = newdie(&dwtypes, DW_ABRV_STRUCTTYPE,
			mkinternaltypename("hash",
				getattr(keytype, DW_AT_name)->data,
				getattr(valtype, DW_AT_name)->data));
		copychildren(dwh, hash);
		substitutetype(dwh, "buckets", defptrto(dwhb));
		substitutetype(dwh, "oldbuckets", defptrto(dwhb));
		newattr(dwh, DW_AT_byte_size, DW_CLS_CONSTANT,
			getattr(hash, DW_AT_byte_size)->value, nil);

		// make map type a pointer to hash<K,V>
		newrefattr(die, DW_AT_type, defptrto(dwh));
	}
}

static void
synthesizechantypes(DWDie *die)
{
	DWDie *sudog, *waitq, *hchan,
		*dws, *dww, *dwh, *elemtype;
	DWAttr *a;
	int elemsize, sudogsize;

	sudog = walktypedef(defgotype(lookup_or_diag("type.runtime.sudog")));
	waitq = walktypedef(defgotype(lookup_or_diag("type.runtime.waitq")));
	hchan = walktypedef(defgotype(lookup_or_diag("type.runtime.hchan")));
	if (sudog == nil || waitq == nil || hchan == nil)
		return;

	sudogsize = getattr(sudog, DW_AT_byte_size)->value;

	for (; die != nil; die = die->link) {
		if (die->abbrev != DW_ABRV_CHANTYPE)
			continue;
		elemtype = (DWDie*) getattr(die, DW_AT_go_elem)->data;
		a = getattr(elemtype, DW_AT_byte_size);
		elemsize = a ? a->value : PtrSize;

		// sudog<T>
		dws = newdie(&dwtypes, DW_ABRV_STRUCTTYPE,
			mkinternaltypename("sudog",
				getattr(elemtype, DW_AT_name)->data, nil));
		copychildren(dws, sudog);
		substitutetype(dws, "elem", elemtype);
		newattr(dws, DW_AT_byte_size, DW_CLS_CONSTANT,
			sudogsize + (elemsize > 8 ? elemsize - 8 : 0), nil);

		// waitq<T>
		dww = newdie(&dwtypes, DW_ABRV_STRUCTTYPE,
			mkinternaltypename("waitq", getattr(elemtype, DW_AT_name)->data, nil));
		copychildren(dww, waitq);
		substitutetype(dww, "first", defptrto(dws));
		substitutetype(dww, "last",  defptrto(dws));
		newattr(dww, DW_AT_byte_size, DW_CLS_CONSTANT,
			getattr(waitq, DW_AT_byte_size)->value, nil);

		// hchan<T>
		dwh = newdie(&dwtypes, DW_ABRV_STRUCTTYPE,
			mkinternaltypename("hchan", getattr(elemtype, DW_AT_name)->data, nil));
		copychildren(dwh, hchan);
		substitutetype(dwh, "recvq", dww);
		substitutetype(dwh, "sendq", dww);
		newattr(dwh, DW_AT_byte_size, DW_CLS_CONSTANT,
			getattr(hchan, DW_AT_byte_size)->value, nil);

		newrefattr(die, DW_AT_type, defptrto(dwh));
	}
}

// For use with pass.c::genasmsym
static void
defdwsymb(LSym* sym, char *s, int t, vlong v, vlong size, int ver, LSym *gotype)
{
	DWDie *dv, *dt;

	USED(size);
	if (strncmp(s, "go.string.", 10) == 0)
		return;

	if (strncmp(s, "type.", 5) == 0 && strcmp(s, "type.*") != 0 && strncmp(s, "type..", 6) != 0) {
		defgotype(sym);
		return;
	}

	dv = nil;

	switch (t) {
	default:
		return;
	case 'd':
	case 'b':
	case 'D':
	case 'B':
		dv = newdie(&dwglobals, DW_ABRV_VARIABLE, s);
		newabslocexprattr(dv, v, sym);
		if (ver == 0)
			newattr(dv, DW_AT_external, DW_CLS_FLAG, 1, 0);
		// fallthrough
	case 'a':
	case 'p':
		dt = defgotype(gotype);
	}

	if (dv != nil)
		newrefattr(dv, DW_AT_type, dt);
}

static void
movetomodule(DWDie *parent)
{
	DWDie *die;

	die = dwroot.child->child;
	while(die->link != nil)
		die = die->link;
	die->link = parent->child;
}

// If the pcln table contains runtime/string.goc, use that to set gdbscript path.
static void
finddebugruntimepath(LSym *s)
{
	int i;
	char *p;
	LSym *f;
	
	if(gdbscript[0] != '\0')
		return;

	for(i=0; i<s->pcln->nfile; i++) {
		f = s->pcln->file[i];
		if((p = strstr(f->name, "runtime/string.goc")) != nil) {
			*p = '\0';
			snprint(gdbscript, sizeof gdbscript, "%sruntime/runtime-gdb.py", f->name);
			*p = 'r';
			break;
		}
	}
}

/*
 * Generate short opcodes when possible, long ones when necessary.
 * See section 6.2.5
 */

enum {
	LINE_BASE = -1,
	LINE_RANGE = 4,
	OPCODE_BASE = 10
};

static void
putpclcdelta(vlong delta_pc, vlong delta_lc)
{
	if (LINE_BASE <= delta_lc && delta_lc < LINE_BASE+LINE_RANGE) {
		vlong opcode = OPCODE_BASE + (delta_lc - LINE_BASE) + (LINE_RANGE * delta_pc);
		if (OPCODE_BASE <= opcode && opcode < 256) {
			cput(opcode);
			return;
		}
	}

	if (delta_pc) {
		cput(DW_LNS_advance_pc);
		sleb128put(delta_pc);
	}

	cput(DW_LNS_advance_line);
	sleb128put(delta_lc);
	cput(DW_LNS_copy);
}

static void
newcfaoffsetattr(DWDie *die, int32 offs)
{
	char block[10];
	int i;

	i = 0;

	block[i++] = DW_OP_call_frame_cfa;
	if (offs != 0) {
		block[i++] = DW_OP_consts;
		i += sleb128enc(offs, block+i);
		block[i++] = DW_OP_plus;
	}
	newattr(die, DW_AT_location, DW_CLS_BLOCK, i, mal(i));
	memmove(die->attr->data, block, i);
}

static char*
mkvarname(char* name, int da)
{
	char buf[1024];
	char *n;

	snprint(buf, sizeof buf, "%s#%d", name, da);
	n = mal(strlen(buf) + 1);
	memmove(n, buf, strlen(buf));
	return n;
}

/*
 * Walk prog table, emit line program and build DIE tree.
 */

// flush previous compilation unit.
static void
flushunit(DWDie *dwinfo, vlong pc, LSym *pcsym, vlong unitstart, int32 header_length)
{
	vlong here;

	if (dwinfo != nil && pc != 0) {
		newattr(dwinfo, DW_AT_high_pc, DW_CLS_ADDRESS, pc+1, (char*)pcsym);
	}

	if (unitstart >= 0) {
		cput(0);  // start extended opcode
		uleb128put(1);
		cput(DW_LNE_end_sequence);

		here = cpos();
		cseek(unitstart);
		LPUT(here - unitstart - sizeof(int32));	 // unit_length
		WPUT(2);  // dwarf version
		LPUT(header_length); // header length starting here
		cseek(here);
	}
}

static void
writelines(void)
{
	LSym *s, *epcs;
	Auto *a;
	vlong unitstart, headerend, offs;
	vlong pc, epc;
	int i, lang, da, dt, line, file;
	DWDie *dwinfo, *dwfunc, *dwvar, **dws;
	DWDie *varhash[HASHSIZE];
	char *n, *nn;
	Pciter pcfile, pcline;
	LSym **files, *f;

	if(linesec == S)
		linesec = linklookup(ctxt, ".dwarfline", 0);
	linesec->nr = 0;

	unitstart = -1;
	headerend = -1;
	epc = 0;
	epcs = S;
	lineo = cpos();
	dwinfo = nil;
	
	flushunit(dwinfo, epc, epcs, unitstart, headerend - unitstart - 10);
	unitstart = cpos();
	
	lang = DW_LANG_Go;
	
	s = ctxt->textp;

	dwinfo = newdie(&dwroot, DW_ABRV_COMPUNIT, estrdup("go"));
	newattr(dwinfo, DW_AT_language, DW_CLS_CONSTANT,lang, 0);
	newattr(dwinfo, DW_AT_stmt_list, DW_CLS_PTR, unitstart - lineo, 0);
	newattr(dwinfo, DW_AT_low_pc, DW_CLS_ADDRESS, s->value, (char*)s);

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	LPUT(0);   // unit_length (*), will be filled in by flushunit.
	WPUT(2);   // dwarf version (appendix F)
	LPUT(0);   // header_length (*), filled in by flushunit.
	// cpos == unitstart + 4 + 2 + 4
	cput(1);   // minimum_instruction_length
	cput(1);   // default_is_stmt
	cput(LINE_BASE);     // line_base
	cput(LINE_RANGE);    // line_range
	cput(OPCODE_BASE);   // opcode_base
	cput(0);   // standard_opcode_lengths[1]
	cput(1);   // standard_opcode_lengths[2]
	cput(1);   // standard_opcode_lengths[3]
	cput(1);   // standard_opcode_lengths[4]
	cput(1);   // standard_opcode_lengths[5]
	cput(0);   // standard_opcode_lengths[6]
	cput(0);   // standard_opcode_lengths[7]
	cput(0);   // standard_opcode_lengths[8]
	cput(1);   // standard_opcode_lengths[9]
	cput(0);   // include_directories  (empty)

	files = emallocz(ctxt->nhistfile*sizeof files[0]);
	for(f = ctxt->filesyms; f != nil; f = f->next)
		files[f->value-1] = f;

	for(i=0; i<ctxt->nhistfile; i++) {
		strnput(files[i]->name, strlen(files[i]->name) + 4);
		// 4 zeros: the string termination + 3 fields.
	}

	cput(0);   // terminate file_names.
	headerend = cpos();

	cput(0);  // start extended opcode
	uleb128put(1 + PtrSize);
	cput(DW_LNE_set_address);

	pc = s->value;
	line = 1;
	file = 1;
	if(linkmode == LinkExternal)
		adddwarfrel(linesec, s, lineo, PtrSize, 0);
	else
		addrput(pc);

	for(ctxt->cursym = ctxt->textp; ctxt->cursym != nil; ctxt->cursym = ctxt->cursym->next) {
		s = ctxt->cursym;

		dwfunc = newdie(dwinfo, DW_ABRV_FUNCTION, s->name);
		newattr(dwfunc, DW_AT_low_pc, DW_CLS_ADDRESS, s->value, (char*)s);
		epc = s->value + s->size;
		epcs = s;
		newattr(dwfunc, DW_AT_high_pc, DW_CLS_ADDRESS, epc, (char*)s);
		if (s->version == 0)
			newattr(dwfunc, DW_AT_external, DW_CLS_FLAG, 1, 0);

		if(s->pcln == nil)
			continue;

		finddebugruntimepath(s);

		pciterinit(ctxt, &pcfile, &s->pcln->pcfile);
		pciterinit(ctxt, &pcline, &s->pcln->pcline);
		epc = pc;
		while(!pcfile.done && !pcline.done) {
			if(epc - s->value >= pcfile.nextpc) {
				pciternext(&pcfile);
				continue;
			}
			if(epc - s->value >= pcline.nextpc) {
				pciternext(&pcline);
				continue;
			}

			if(file != pcfile.value) {
				cput(DW_LNS_set_file);
				uleb128put(pcfile.value);
				file = pcfile.value;
			}
			putpclcdelta(s->value + pcline.pc - pc, pcline.value - line);

			pc = s->value + pcline.pc;
			line = pcline.value;
			if(pcfile.nextpc < pcline.nextpc)
				epc = pcfile.nextpc;
			else
				epc = pcline.nextpc;
			epc += s->value;
		}

		da = 0;
		dwfunc->hash = varhash;	 // enable indexing of children by name
		memset(varhash, 0, sizeof varhash);
		for(a = s->autom; a; a = a->link) {
			switch (a->type) {
			case A_AUTO:
				dt = DW_ABRV_AUTO;
				offs = a->aoffset - PtrSize;
				break;
			case A_PARAM:
				dt = DW_ABRV_PARAM;
				offs = a->aoffset;
				break;
			default:
				continue;
			}
			if (strstr(a->asym->name, ".autotmp_"))
				continue;
			if (find(dwfunc, a->asym->name) != nil)
				n = mkvarname(a->asym->name, da);
			else
				n = a->asym->name;
			// Drop the package prefix from locals and arguments.
			nn = strrchr(n, '.');
			if (nn)
				n = nn + 1;

			dwvar = newdie(dwfunc, dt, n);
			newcfaoffsetattr(dwvar, offs);
			newrefattr(dwvar, DW_AT_type, defgotype(a->gotype));

			// push dwvar down dwfunc->child to preserve order
			newattr(dwvar, DW_AT_internal_location, DW_CLS_CONSTANT, offs, nil);
			dwfunc->child = dwvar->link;  // take dwvar out from the top of the list
			for (dws = &dwfunc->child; *dws != nil; dws = &(*dws)->link)
				if (offs > getattr(*dws, DW_AT_internal_location)->value)
					break;
			dwvar->link = *dws;
			*dws = dwvar;

			da++;
		}

		dwfunc->hash = nil;
	}

	flushunit(dwinfo, epc, epcs, unitstart, headerend - unitstart - 10);
	linesize = cpos() - lineo;
}

/*
 *  Emit .debug_frame
 */
enum
{
	CIERESERVE = 16,
	DATAALIGNMENTFACTOR = -4,	// TODO -PtrSize?
	FAKERETURNCOLUMN = 16		// TODO gdb6 doesn't like > 15?
};

static void
putpccfadelta(vlong deltapc, vlong cfa)
{
	cput(DW_CFA_def_cfa_offset_sf);
	sleb128put(cfa / DATAALIGNMENTFACTOR);

	if (deltapc < 0x40) {
		cput(DW_CFA_advance_loc + deltapc);
	} else if (deltapc < 0x100) {
		cput(DW_CFA_advance_loc1);
		cput(deltapc);
	} else if (deltapc < 0x10000) {
		cput(DW_CFA_advance_loc2);
		WPUT(deltapc);
	} else {
		cput(DW_CFA_advance_loc4);
		LPUT(deltapc);
	}
}

static void
writeframes(void)
{
	LSym *s;
	vlong fdeo, fdesize, pad;
	Pciter pcsp;
	uint32 nextpc;

	if(framesec == S)
		framesec = linklookup(ctxt, ".dwarfframe", 0);
	framesec->nr = 0;
	frameo = cpos();

	// Emit the CIE, Section 6.4.1
	LPUT(CIERESERVE);	// initial length, must be multiple of PtrSize
	LPUT(0xffffffff);	// cid.
	cput(3);		// dwarf version (appendix F)
	cput(0);		// augmentation ""
	uleb128put(1);		// code_alignment_factor
	sleb128put(DATAALIGNMENTFACTOR); // guess
	uleb128put(FAKERETURNCOLUMN);	// return_address_register

	cput(DW_CFA_def_cfa);
	uleb128put(DWARFREGSP);	// register SP (**ABI-dependent, defined in l.h)
	uleb128put(PtrSize);	// offset

	cput(DW_CFA_offset + FAKERETURNCOLUMN);	 // return address
	uleb128put(-PtrSize / DATAALIGNMENTFACTOR);  // at cfa - x*4

	// 4 is to exclude the length field.
	pad = CIERESERVE + frameo + 4 - cpos();
	if (pad < 0) {
		diag("dwarf: CIERESERVE too small by %lld bytes.", -pad);
		errorexit();
	}
	strnput("", pad);

	for(ctxt->cursym = ctxt->textp; ctxt->cursym != nil; ctxt->cursym = ctxt->cursym->next) {
		s = ctxt->cursym;
		if(s->pcln == nil)
			continue;

		fdeo = cpos();
		// Emit a FDE, Section 6.4.1, starting wit a placeholder.
		LPUT(0);	// length, must be multiple of PtrSize
		LPUT(0);	// Pointer to the CIE above, at offset 0
		addrput(0);	// initial location
		addrput(0);	// address range

		for(pciterinit(ctxt, &pcsp, &s->pcln->pcsp); !pcsp.done; pciternext(&pcsp)) {
			nextpc = pcsp.nextpc;
			// pciterinit goes up to the end of the function,
			// but DWARF expects us to stop just before the end.
			if(nextpc == s->size) {
				nextpc--;
				if(nextpc < pcsp.pc)
					continue;
			}
			putpccfadelta(nextpc - pcsp.pc, PtrSize + pcsp.value);
		}

		fdesize = cpos() - fdeo - 4;	// exclude the length field.
		pad = rnd(fdesize, PtrSize) - fdesize;
		strnput("", pad);
		fdesize += pad;

		// Emit the FDE header for real, Section 6.4.1.
		cseek(fdeo);
		LPUT(fdesize);
		if(linkmode == LinkExternal) {
			adddwarfrel(framesec, framesym, frameo, 4, 0);
			adddwarfrel(framesec, s, frameo, PtrSize, 0);
		}
		else {
			LPUT(0);
			addrput(s->value);
		}
		addrput(s->size);
		cseek(fdeo + 4 + fdesize);
	}

	cflush();
	framesize = cpos() - frameo;
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */
enum
{
	COMPUNITHEADERSIZE = 4+2+4+1
};

static void
writeinfo(void)
{
	DWDie *compunit;
	vlong unitstart, here;

	fwdcount = 0;
	if (infosec == S)
		infosec = linklookup(ctxt, ".dwarfinfo", 0);
	infosec->nr = 0;

	if(arangessec == S)
		arangessec = linklookup(ctxt, ".dwarfaranges", 0);
	arangessec->nr = 0;

	for (compunit = dwroot.child; compunit; compunit = compunit->link) {
		unitstart = cpos();

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		LPUT(0);	// unit_length (*), will be filled in later.
		WPUT(2);	// dwarf version (appendix F)

		// debug_abbrev_offset (*)
		if(linkmode == LinkExternal)
			adddwarfrel(infosec, abbrevsym, infoo, 4, 0);
		else
			LPUT(0);

		cput(PtrSize);	// address_size

		putdie(compunit);

		here = cpos();
		cseek(unitstart);
		LPUT(here - unitstart - 4);	// exclude the length field.
		cseek(here);
	}
	cflush();
}

/*
 *  Emit .debug_pubnames/_types.  _info must have been written before,
 *  because we need die->offs and infoo/infosize;
 */
static int
ispubname(DWDie *die)
{
	DWAttr *a;

	switch(die->abbrev) {
	case DW_ABRV_FUNCTION:
	case DW_ABRV_VARIABLE:
		a = getattr(die, DW_AT_external);
		return a && a->value;
	}
	return 0;
}

static int
ispubtype(DWDie *die)
{
	return die->abbrev >= DW_ABRV_NULLTYPE;
}

static vlong
writepub(int (*ispub)(DWDie*))
{
	DWDie *compunit, *die;
	DWAttr *dwa;
	vlong unitstart, unitend, sectionstart, here;

	sectionstart = cpos();

	for (compunit = dwroot.child; compunit != nil; compunit = compunit->link) {
		unitstart = compunit->offs - COMPUNITHEADERSIZE;
		if (compunit->link != nil)
			unitend = compunit->link->offs - COMPUNITHEADERSIZE;
		else
			unitend = infoo + infosize;

		// Write .debug_pubnames/types	Header (sec 6.1.1)
		LPUT(0);			// unit_length (*), will be filled in later.
		WPUT(2);			// dwarf version (appendix F)
		LPUT(unitstart);		// debug_info_offset (of the Comp unit Header)
		LPUT(unitend - unitstart);	// debug_info_length

		for (die = compunit->child; die != nil; die = die->link) {
			if (!ispub(die)) continue;
			LPUT(die->offs - unitstart);
			dwa = getattr(die, DW_AT_name);
			strnput(dwa->data, dwa->value + 1);
		}
		LPUT(0);

		here = cpos();
		cseek(sectionstart);
		LPUT(here - sectionstart - 4);	// exclude the length field.
		cseek(here);

	}

	return sectionstart;
}

/*
 *  emit .debug_aranges.  _info must have been written before,
 *  because we need die->offs of dw_globals.
 */
static vlong
writearanges(void)
{
	DWDie *compunit;
	DWAttr *b, *e;
	int headersize;
	vlong sectionstart;
	vlong value;

	sectionstart = cpos();
	headersize = rnd(4+2+4+1+1, PtrSize);  // don't count unit_length field itself

	for (compunit = dwroot.child; compunit != nil; compunit = compunit->link) {
		b = getattr(compunit,  DW_AT_low_pc);
		if (b == nil)
			continue;
		e = getattr(compunit,  DW_AT_high_pc);
		if (e == nil)
			continue;

		// Write .debug_aranges	 Header + entry	 (sec 6.1.2)
		LPUT(headersize + 4*PtrSize - 4);	// unit_length (*)
		WPUT(2);	// dwarf version (appendix F)

		value = compunit->offs - COMPUNITHEADERSIZE;	// debug_info_offset
		if(linkmode == LinkExternal)
			adddwarfrel(arangessec, infosym, sectionstart, 4, value);
		else
			LPUT(value);

		cput(PtrSize);	// address_size
		cput(0);	// segment_size
		strnput("", headersize - (4+2+4+1+1));	// align to PtrSize

		if(linkmode == LinkExternal)
			adddwarfrel(arangessec, (LSym*)b->data, sectionstart, PtrSize, b->value-((LSym*)b->data)->value);
		else
			addrput(b->value);

		addrput(e->value - b->value);
		addrput(0);
		addrput(0);
	}
	cflush();
	return sectionstart;
}

static vlong
writegdbscript(void)
{
	vlong sectionstart;

	sectionstart = cpos();

	if (gdbscript[0]) {
		cput(1);  // magic 1 byte?
		strnput(gdbscript, strlen(gdbscript)+1);
		cflush();
	}
	return sectionstart;
}

static void
align(vlong size)
{
	if(HEADTYPE == Hwindows) // Only Windows PE need section align.
		strnput("", rnd(size, PEFILEALIGN) - size);
}

static vlong
writedwarfreloc(LSym* s)
{
	int i;
	vlong start;
	Reloc *r;
	
	start = cpos();
	for(r = s->r; r < s->r+s->nr; r++) {
		if(iself)
			i = elfreloc1(r, r->off);
		else if(HEADTYPE == Hdarwin)
			i = machoreloc1(r, r->off);
		else
			i = -1;
		if(i < 0)
			diag("unsupported obj reloc %d/%d to %s", r->type, r->siz, r->sym->name);
	}
	return start;
}

/*
 * This is the main entry point for generating dwarf.  After emitting
 * the mandatory debug_abbrev section, it calls writelines() to set up
 * the per-compilation unit part of the DIE tree, while simultaneously
 * emitting the debug_line section.  When the final tree contains
 * forward references, it will write the debug_info section in 2
 * passes.
 *
 */
void
dwarfemitdebugsections(void)
{
	vlong infoe;
	DWDie* die;

	if(debug['w'])  // disable dwarf
		return;

	if(linkmode == LinkExternal && !iself)
		return;

	// For diagnostic messages.
	newattr(&dwtypes, DW_AT_name, DW_CLS_STRING, strlen("dwtypes"), "dwtypes");

	mkindex(&dwroot);
	mkindex(&dwtypes);
	mkindex(&dwglobals);

	// Some types that must exist to define other ones.
	newdie(&dwtypes, DW_ABRV_NULLTYPE, "<unspecified>");
	newdie(&dwtypes, DW_ABRV_NULLTYPE, "void");
	newdie(&dwtypes, DW_ABRV_BARE_PTRTYPE, "unsafe.Pointer");

	die = newdie(&dwtypes, DW_ABRV_BASETYPE, "uintptr");  // needed for array size
	newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_unsigned, 0);
	newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, PtrSize, 0);
	newattr(die, DW_AT_go_kind, DW_CLS_CONSTANT, KindUintptr, 0);

	// Needed by the prettyprinter code for interface inspection.
	defgotype(lookup_or_diag("type.runtime._type"));
	defgotype(lookup_or_diag("type.runtime.interfacetype"));
	defgotype(lookup_or_diag("type.runtime.itab"));

	genasmsym(defdwsymb);

	writeabbrev();
	align(abbrevsize);
	writelines();
	align(linesize);
	writeframes();
	align(framesize);

	synthesizestringtypes(dwtypes.child);
	synthesizeslicetypes(dwtypes.child);
	synthesizemaptypes(dwtypes.child);
	synthesizechantypes(dwtypes.child);

	reversetree(&dwroot.child);
	reversetree(&dwtypes.child);
	reversetree(&dwglobals.child);

	movetomodule(&dwtypes);
	movetomodule(&dwglobals);

	infoo = cpos();
	writeinfo();
	infoe = cpos();
	pubnameso = infoe;
	pubtypeso = infoe;
	arangeso = infoe;
	gdbscripto = infoe;

	if (fwdcount > 0) {
		if (debug['v'])
			Bprint(&bso, "%5.2f dwarf pass 2.\n", cputime());
		cseek(infoo);
		writeinfo();
		if (fwdcount > 0) {
			diag("dwarf: unresolved references after first dwarf info pass");
			errorexit();
		}
		if (infoe != cpos()) {
			diag("dwarf: inconsistent second dwarf info pass");
			errorexit();
		}
	}
	infosize = infoe - infoo;
	align(infosize);

	pubnameso  = writepub(ispubname);
	pubnamessize  = cpos() - pubnameso;
	align(pubnamessize);

	pubtypeso  = writepub(ispubtype);
	pubtypessize  = cpos() - pubtypeso;
	align(pubtypessize);

	arangeso   = writearanges();
	arangessize   = cpos() - arangeso;
	align(arangessize);

	gdbscripto = writegdbscript();
	gdbscriptsize = cpos() - gdbscripto;
	align(gdbscriptsize);

	while(cpos()&7)
		cput(0);
	inforeloco = writedwarfreloc(infosec);
	inforelocsize = cpos() - inforeloco;
	align(inforelocsize);

	arangesreloco = writedwarfreloc(arangessec);
	arangesrelocsize = cpos() - arangesreloco;
	align(arangesrelocsize);

	linereloco = writedwarfreloc(linesec);
	linerelocsize = cpos() - linereloco;
	align(linerelocsize);

	framereloco = writedwarfreloc(framesec);
	framerelocsize = cpos() - framereloco;
	align(framerelocsize);
}

/*
 *  Elf.
 */
enum
{
	ElfStrDebugAbbrev,
	ElfStrDebugAranges,
	ElfStrDebugFrame,
	ElfStrDebugInfo,
	ElfStrDebugLine,
	ElfStrDebugLoc,
	ElfStrDebugMacinfo,
	ElfStrDebugPubNames,
	ElfStrDebugPubTypes,
	ElfStrDebugRanges,
	ElfStrDebugStr,
	ElfStrGDBScripts,
	ElfStrRelDebugInfo,
	ElfStrRelDebugAranges,
	ElfStrRelDebugLine,
	ElfStrRelDebugFrame,
	NElfStrDbg
};

vlong elfstrdbg[NElfStrDbg];

void
dwarfaddshstrings(LSym *shstrtab)
{
	if(debug['w'])  // disable dwarf
		return;

	elfstrdbg[ElfStrDebugAbbrev]   = addstring(shstrtab, ".debug_abbrev");
	elfstrdbg[ElfStrDebugAranges]  = addstring(shstrtab, ".debug_aranges");
	elfstrdbg[ElfStrDebugFrame]    = addstring(shstrtab, ".debug_frame");
	elfstrdbg[ElfStrDebugInfo]     = addstring(shstrtab, ".debug_info");
	elfstrdbg[ElfStrDebugLine]     = addstring(shstrtab, ".debug_line");
	elfstrdbg[ElfStrDebugLoc]      = addstring(shstrtab, ".debug_loc");
	elfstrdbg[ElfStrDebugMacinfo]  = addstring(shstrtab, ".debug_macinfo");
	elfstrdbg[ElfStrDebugPubNames] = addstring(shstrtab, ".debug_pubnames");
	elfstrdbg[ElfStrDebugPubTypes] = addstring(shstrtab, ".debug_pubtypes");
	elfstrdbg[ElfStrDebugRanges]   = addstring(shstrtab, ".debug_ranges");
	elfstrdbg[ElfStrDebugStr]      = addstring(shstrtab, ".debug_str");
	elfstrdbg[ElfStrGDBScripts]    = addstring(shstrtab, ".debug_gdb_scripts");
	if(linkmode == LinkExternal) {
		if(thechar == '6' || thechar == '9') {
			elfstrdbg[ElfStrRelDebugInfo] = addstring(shstrtab, ".rela.debug_info");
			elfstrdbg[ElfStrRelDebugAranges] = addstring(shstrtab, ".rela.debug_aranges");
			elfstrdbg[ElfStrRelDebugLine] = addstring(shstrtab, ".rela.debug_line");
			elfstrdbg[ElfStrRelDebugFrame] = addstring(shstrtab, ".rela.debug_frame");
		} else {
			elfstrdbg[ElfStrRelDebugInfo] = addstring(shstrtab, ".rel.debug_info");
			elfstrdbg[ElfStrRelDebugAranges] = addstring(shstrtab, ".rel.debug_aranges");
			elfstrdbg[ElfStrRelDebugLine] = addstring(shstrtab, ".rel.debug_line");
			elfstrdbg[ElfStrRelDebugFrame] = addstring(shstrtab, ".rel.debug_frame");
		}

		infosym = linklookup(ctxt, ".debug_info", 0);
		infosym->hide = 1;

		abbrevsym = linklookup(ctxt, ".debug_abbrev", 0);
		abbrevsym->hide = 1;

		linesym = linklookup(ctxt, ".debug_line", 0);
		linesym->hide = 1;

		framesym = linklookup(ctxt, ".debug_frame", 0);
		framesym->hide = 1;
	}
}

// Add section symbols for DWARF debug info.  This is called before
// dwarfaddelfheaders.
void
dwarfaddelfsectionsyms()
{
	if(infosym != nil) {
		infosympos = cpos();
		putelfsectionsym(infosym, 0);
	}
	if(abbrevsym != nil) {
		abbrevsympos = cpos();
		putelfsectionsym(abbrevsym, 0);
	}
	if(linesym != nil) {
		linesympos = cpos();
		putelfsectionsym(linesym, 0);
	}
	if(framesym != nil) {
		framesympos = cpos();
		putelfsectionsym(framesym, 0);
	}
}

static void
dwarfaddelfrelocheader(int elfstr, ElfShdr *shdata, vlong off, vlong size)
{
	ElfShdr *sh;

	sh = newElfShdr(elfstrdbg[elfstr]);
	if(thechar == '6' || thechar == '9') {
		sh->type = SHT_RELA;
	} else {
		sh->type = SHT_REL;
	}
	sh->entsize = PtrSize*(2+(sh->type==SHT_RELA));
	sh->link = elfshname(".symtab")->shnum;
	sh->info = shdata->shnum;
	sh->off = off;
	sh->size = size;
	sh->addralign = PtrSize;
	
}

void
dwarfaddelfheaders(void)
{
	ElfShdr *sh, *shinfo, *sharanges, *shline, *shframe;

	if(debug['w'])  // disable dwarf
		return;

	sh = newElfShdr(elfstrdbg[ElfStrDebugAbbrev]);
	sh->type = SHT_PROGBITS;
	sh->off = abbrevo;
	sh->size = abbrevsize;
	sh->addralign = 1;
	if(abbrevsympos > 0)
		putelfsymshndx(abbrevsympos, sh->shnum);

	sh = newElfShdr(elfstrdbg[ElfStrDebugLine]);
	sh->type = SHT_PROGBITS;
	sh->off = lineo;
	sh->size = linesize;
	sh->addralign = 1;
	if(linesympos > 0)
		putelfsymshndx(linesympos, sh->shnum);
	shline = sh;

	sh = newElfShdr(elfstrdbg[ElfStrDebugFrame]);
	sh->type = SHT_PROGBITS;
	sh->off = frameo;
	sh->size = framesize;
	sh->addralign = 1;
	if(framesympos > 0)
		putelfsymshndx(framesympos, sh->shnum);
	shframe = sh;

	sh = newElfShdr(elfstrdbg[ElfStrDebugInfo]);
	sh->type = SHT_PROGBITS;
	sh->off = infoo;
	sh->size = infosize;
	sh->addralign = 1;
	if(infosympos > 0)
		putelfsymshndx(infosympos, sh->shnum);
	shinfo = sh;

	if (pubnamessize > 0) {
		sh = newElfShdr(elfstrdbg[ElfStrDebugPubNames]);
		sh->type = SHT_PROGBITS;
		sh->off = pubnameso;
		sh->size = pubnamessize;
		sh->addralign = 1;
	}

	if (pubtypessize > 0) {
		sh = newElfShdr(elfstrdbg[ElfStrDebugPubTypes]);
		sh->type = SHT_PROGBITS;
		sh->off = pubtypeso;
		sh->size = pubtypessize;
		sh->addralign = 1;
	}

	sharanges = nil;
	if (arangessize) {
		sh = newElfShdr(elfstrdbg[ElfStrDebugAranges]);
		sh->type = SHT_PROGBITS;
		sh->off = arangeso;
		sh->size = arangessize;
		sh->addralign = 1;
		sharanges = sh;
	}

	if (gdbscriptsize) {
		sh = newElfShdr(elfstrdbg[ElfStrGDBScripts]);
		sh->type = SHT_PROGBITS;
		sh->off = gdbscripto;
		sh->size = gdbscriptsize;
		sh->addralign = 1;
	}

	if(inforelocsize)
		dwarfaddelfrelocheader(ElfStrRelDebugInfo, shinfo, inforeloco, inforelocsize);

	if(arangesrelocsize)
		dwarfaddelfrelocheader(ElfStrRelDebugAranges, sharanges, arangesreloco, arangesrelocsize);

	if(linerelocsize)
		dwarfaddelfrelocheader(ElfStrRelDebugLine, shline, linereloco, linerelocsize);

	if(framerelocsize)
		dwarfaddelfrelocheader(ElfStrRelDebugFrame, shframe, framereloco, framerelocsize);
}

/*
 * Macho
 */
void
dwarfaddmachoheaders(void)
{
	MachoSect *msect;
	MachoSeg *ms;
	vlong fakestart;
	int nsect;

	if(debug['w'])  // disable dwarf
		return;

	// Zero vsize segments won't be loaded in memory, even so they
	// have to be page aligned in the file.
	fakestart = abbrevo & ~0xfff;

	nsect = 4;
	if (pubnamessize  > 0)
		nsect++;
	if (pubtypessize  > 0)
		nsect++;
	if (arangessize	  > 0)
		nsect++;
	if (gdbscriptsize > 0)
		nsect++;

	ms = newMachoSeg("__DWARF", nsect);
	ms->fileoffset = fakestart;
	ms->filesize = abbrevo-fakestart;
	ms->vaddr = ms->fileoffset + segdata.vaddr - segdata.fileoff;

	msect = newMachoSect(ms, "__debug_abbrev", "__DWARF");
	msect->off = abbrevo;
	msect->size = abbrevsize;
	msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
	ms->filesize += msect->size;

	msect = newMachoSect(ms, "__debug_line", "__DWARF");
	msect->off = lineo;
	msect->size = linesize;
	msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
	ms->filesize += msect->size;

	msect = newMachoSect(ms, "__debug_frame", "__DWARF");
	msect->off = frameo;
	msect->size = framesize;
	msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
	ms->filesize += msect->size;

	msect = newMachoSect(ms, "__debug_info", "__DWARF");
	msect->off = infoo;
	msect->size = infosize;
	msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
	ms->filesize += msect->size;

	if (pubnamessize > 0) {
		msect = newMachoSect(ms, "__debug_pubnames", "__DWARF");
		msect->off = pubnameso;
		msect->size = pubnamessize;
		msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
		ms->filesize += msect->size;
	}

	if (pubtypessize > 0) {
		msect = newMachoSect(ms, "__debug_pubtypes", "__DWARF");
		msect->off = pubtypeso;
		msect->size = pubtypessize;
		msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
		ms->filesize += msect->size;
	}

	if (arangessize > 0) {
		msect = newMachoSect(ms, "__debug_aranges", "__DWARF");
		msect->off = arangeso;
		msect->size = arangessize;
		msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
		ms->filesize += msect->size;
	}

	// TODO(lvd) fix gdb/python to load MachO (16 char section name limit)
	if (gdbscriptsize > 0) {
		msect = newMachoSect(ms, "__debug_gdb_scripts", "__DWARF");
		msect->off = gdbscripto;
		msect->size = gdbscriptsize;
		msect->addr = msect->off + segdata.vaddr - segdata.fileoff;
		ms->filesize += msect->size;
	}
}

/*
 * Windows PE
 */
void
dwarfaddpeheaders(void)
{
	if(debug['w'])  // disable dwarf
		return;

	newPEDWARFSection(".debug_abbrev", abbrevsize);
	newPEDWARFSection(".debug_line", linesize);
	newPEDWARFSection(".debug_frame", framesize);
	newPEDWARFSection(".debug_info", infosize);
	newPEDWARFSection(".debug_pubnames", pubnamessize);
	newPEDWARFSection(".debug_pubtypes", pubtypessize);
	newPEDWARFSection(".debug_aranges", arangessize);
	newPEDWARFSection(".debug_gdb_scripts", gdbscriptsize);
}
