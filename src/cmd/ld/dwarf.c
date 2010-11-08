// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO:
//   - eliminate DW_CLS_ if not used
//   - package info in compilation units
//   - assign global variables and types to their packages
//   - (upstream) type info for C parts of runtime
//   - gdb uses c syntax, meaning clumsy quoting is needed for go identifiers. eg
//     ptype struct '[]uint8' and qualifiers need to be quoted away
//   - lexical scoping is lost, so gdb gets confused as to which 'main.i' you mean.
//
#include	"l.h"
#include	"lib.h"
#include	"../ld/dwarf.h"
#include	"../ld/dwarf_defs.h"
#include	"../ld/elf.h"
#include	"../ld/macho.h"

/*
 * Offsets and sizes of the debug_* sections in the cout file.
 */

static vlong abbrevo;
static vlong abbrevsize;
static vlong lineo;
static vlong linesize;
static vlong infoo;	 // also the base for DWDie->offs and reference attributes.
static vlong infosize;
static vlong frameo;
static vlong framesize;
static vlong pubnameso;
static vlong pubnamessize;
static vlong pubtypeso;
static vlong pubtypessize;
static vlong arangeso;
static vlong arangessize;

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
 * the ordering of atributes in the Abbrevs and DIEs, and we will
 * always write them out in the order of declaration in the abbrev.
 * This implementation relies on tag, attr < 127, so they serialize as
 * a char, hence we do not support user-defined tags or attributes.
 */
typedef struct DWAttrForm DWAttrForm;
struct DWAttrForm {
	uint8 attr;
	uint8 form;
};

// Index into the abbrevs table below.
// Keep in sync with ispubname() and ispubtype() below.
enum
{
	DW_ABRV_NULL,
	DW_ABRV_COMPUNIT,
	DW_ABRV_FUNCTION,
	DW_ABRV_VARIABLE,
	DW_ABRV_AUTO,
	DW_ABRV_PARAM,
	DW_ABRV_STRUCTFIELD,
	DW_ABRV_NULLTYPE,
	DW_ABRV_BASETYPE,
	DW_ABRV_ARRAYTYPE,
	DW_ABRV_CHANTYPE,
	DW_ABRV_FUNCTYPE,
	DW_ABRV_IFACETYPE,
	DW_ABRV_MAPTYPE,
	DW_ABRV_PTRTYPE,
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
		DW_AT_location,	 DW_FORM_addr,
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
		DW_TAG_member, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		DW_AT_data_member_location,	 DW_FORM_block1,
		DW_AT_type,	 DW_FORM_ref_addr,
		0, 0
	},

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
		0, 0
	},
	/* ARRAYTYPE */
	{
		DW_TAG_array_type, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		DW_AT_byte_size, DW_FORM_udata,
		0, 0
	},

	/* CHANTYPE */
	{
		DW_TAG_typedef, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		0, 0
	},

	/* FUNCTYPE */
	{
		DW_TAG_typedef, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		0, 0
	},

	/* IFACETYPE */
	{
		DW_TAG_interface_type, DW_CHILDREN_no,
		DW_AT_name,	 DW_FORM_string,
		0, 0
	},

	/* MAPTYPE */
	{
		DW_TAG_typedef, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		0, 0
	},

	/* PTRTYPE */
	{
		DW_TAG_pointer_type, DW_CHILDREN_no,
		DW_AT_name,	DW_FORM_string,
		DW_AT_type,	DW_FORM_ref_addr,
		0, 0
	},

	/* SLICETYPE */
	// Children are data, len and cap of runtime::struct Slice.
	{
		DW_TAG_structure_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
		DW_AT_byte_size, DW_FORM_udata,
		0, 0
	},

	/* STRINGTYPE */
	// Children are str and len of runtime::struct String.
	{
		DW_TAG_structure_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
		DW_AT_byte_size, DW_FORM_udata,
		0, 0
	},

	/* STRUCTTYPE */
	{
		DW_TAG_structure_type, DW_CHILDREN_yes,
		DW_AT_name,	DW_FORM_string,
		DW_AT_byte_size, DW_FORM_udata,
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
	int i, n;

	abbrevo = cpos();
	for (i = 1; i < DW_NABRV; i++) {
		// See section 7.5.3
		uleb128put(i);
		uleb128put(abbrevs[i].tag);
		cput(abbrevs[i].children);
		// 0 is not a valid attr or form, and DWAbbrev.attr is
		// 0-terminated, so we can treat it as a string
		n = strlen((char*)abbrevs[i].attr) / 2;
		strnput((char*)abbrevs[i].attr,
			(n+1) * sizeof(DWAttrForm));
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
	uint8 atr;  // DW_AT_
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
newattr(DWDie *die, uint8 attr, int cls, vlong value, char *data)
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
getattr(DWDie *die, uint8 attr)
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
find(DWDie *die, char* name)
{
	DWDie *a, *b;
	int h;

	if (die->hash == nil) {
		diag("lookup of %s in non-indexed DIE", name);
		errorexit();
	}

	h = hashstr(name);
	a = die->hash[h];

	if (a == nil)
		return nil;

	// AT_name always exists.
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
	return nil;
}

static DWAttr*
newrefattr(DWDie *die, uint8 attr, DWDie* ref)
{
	if (ref == nil)
		return nil;
	return newattr(die, attr, DW_CLS_REFERENCE, 0, (char*)ref);
}

static int fwdcount;

static void
putattr(int form, int cls, vlong value, char *data)
{
	switch(form) {
	case DW_FORM_addr:	// address
		addrput(value);
		break;

	case DW_FORM_block1:	// block
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
		if (data == nil) {
			diag("null dwarf reference");
			LPUT(0);  // invalid dwarf, gdb will complain.
		} else {
			if (((DWDie*)data)->offs == 0)
				fwdcount++;
			LPUT(((DWDie*)data)->offs);
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
		diag("Unsupported atribute form %d / class %d", form, cls);
		errorexit();
	}
}

static void
putattrs(int abbrev, DWAttr* attr)
{
	DWAttr *attrs[DW_AT_recursive + 1];
	DWAttrForm* af;

	memset(attrs, 0, sizeof attrs);
	for( ; attr; attr = attr->link)
		attrs[attr->atr] = attr;
	for(af = abbrevs[abbrev].attr; af->attr; af++)
		if (attrs[af->attr])
			putattr(af->form,
				attrs[af->attr]->cls,
				attrs[af->attr]->value,
				attrs[af->attr]->data);
		else
			putattr(af->form, 0, 0, 0);
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
	DWDie *curr, * prev;

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
	if (offs != 0) {
		block[i++] = DW_OP_consts;
		i += sleb128enc(offs, block+i);
		block[i++] = DW_OP_plus;
	}
	newattr(die, DW_AT_data_member_location, DW_CLS_BLOCK, i, mal(i));
	memmove(die->attr->data, block, i);
}

// Decoding the type.* symbols.	 This has to be in sync with
// ../../pkg/runtime/type.go, or more specificaly, with what
// ../gc/reflect.c stuffs in these.

enum {
	KindBool = 1,
	KindInt,
	KindInt8,
	KindInt16,
	KindInt32,
	KindInt64,
	KindUint,
	KindUint8,
	KindUint16,
	KindUint32,
	KindUint64,
	KindUintptr,
	KindFloat,
	KindFloat32,
	KindFloat64,
	KindComplex,
	KindComplex64,
	KindComplex128,
	KindArray,
	KindChan,
	KindFunc,
	KindInterface,
	KindMap,
	KindPtr,
	KindSlice,
	KindString,
	KindStruct,
	KindUnsafePointer,

	KindNoPointers = 1<<7,
};

static Sym*
decode_reloc(Sym *s, int32 off)
{
	int i;

	for (i = 0; i < s->nr; i++)
		if (s->r[i].off == off)
			return s->r[i].sym;
	return nil;
}

static uvlong
decode_inuxi(uchar* p, int sz)
{
	uvlong r;
	uchar *inuxi;
	int i;

	r = 0;
	inuxi = nil;
	switch (sz) {
	case 2:
		inuxi = inuxi2;
		break;
	case 4:
		inuxi = inuxi4;
		break;
	case 8:
		inuxi = inuxi8;
		break;
	default:
		diag("decode inuxi %d", sz);
		errorexit();
	}
	for (i = 0; i < sz; i++)
		r += p[i] << (8*inuxi[i]);

	return r;
}

// Type.commonType.kind
static uint8
decodetype_kind(Sym *s)
{
	return s->p[3*PtrSize + 7] & ~KindNoPointers;	//  0x13 / 0x1f
}

// Type.commonType.size
static vlong
decodetype_size(Sym *s)
{
	return decode_inuxi(s->p + 2*PtrSize, PtrSize);	 // 0x8 / 0x10
}

// Type.ArrayType.elem
static Sym*
decodetype_arrayelem(Sym *s)
{
	return decode_reloc(s, 5*PtrSize + 8);	// 0x1c / 0x30
}

// Type.PtrType.elem
static Sym*
decodetype_ptrelem(Sym *s)
{
	return decode_reloc(s, 5*PtrSize + 8);	// 0x1c / 0x30
}

// Type.StructType.fields.Slice::len
static int
decodetype_structfieldcount(Sym *s)
{
	return decode_inuxi(s->p + 6*PtrSize + 8, 4);  //  0x20 / 0x38
}

// Type.StructType.fields[]-> name, typ and offset. sizeof(structField) =  5*PtrSize
static char*
decodetype_structfieldname(Sym *s, int i)
{
	Sym *p;
	p = decode_reloc(s, 6*PtrSize + 0x10 + i*5*PtrSize);  // go.string."foo"  0x28 / 0x40
	if (p == nil)				// embedded structs have a nil name.
		return nil;
	p = decode_reloc(p, 0);			// string."foo"
	if (p == nil)				// shouldn't happen.
		return nil;
	return (char*)p->p;    			// the c-string
}

static Sym*
decodetype_structfieldtype(Sym *s, int i)
{
	return decode_reloc(s, 8*PtrSize + 0x10 + i*5*PtrSize);	 //   0x30 / 0x50
}

static vlong
decodetype_structfieldoffs(Sym *s, int i)
{
	return decode_inuxi(s->p + 10*PtrSize + 0x10 + i*5*PtrSize, 4);	 // 0x38  / 0x60
}

// Define gotype, for composite ones recurse into constituents.
static DWDie*
defgotype(Sym *gotype)
{
	DWDie *die, *fld, *elem, *ptrelem;
	Sym *s;
	char *name, *ptrname, *f;
	uint8 kind;
	vlong bytesize;
	int i, nfields;

	if (gotype == nil)
		return find(&dwtypes, "<unspecified>");	 // must be defined before

	if (strncmp("type.", gotype->name, 5) != 0) {
		diag("Type name doesn't start with \".type\": %s", gotype->name);
		return find(&dwtypes, "<unspecified>");
	}
	name = gotype->name + 5;  // Altenatively decode from Type.string

	die = find(&dwtypes, name);
	if (die != nil)
		return die;

	if (0 && debug['v'] > 2) {
		print("new type: %s @0x%08x [%d]", gotype->name, gotype->value, gotype->size);
		for (i = 0; i < gotype->size; ++i) {
			if (!(i%8)) print("\n\t%04x ", i);
			print("%02x ", gotype->p[i]);
		}
		print("\n");
		for (i = 0; i < gotype->nr; ++i) {
			print("\t%02x %d %d %lld %s\n",
			      gotype->r[i].off,
			      gotype->r[i].siz,
			      gotype->r[i].type,
			      gotype->r[i].add,
			      gotype->r[i].sym->name);
		}
	}

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

	case KindFloat:
	case KindFloat32:
	case KindFloat64:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name);
		newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_float, 0);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindComplex:
	case KindComplex64:
	case KindComplex128:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name);
		newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_complex_float, 0);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		break;

	case KindArray:
		die = newdie(&dwtypes, DW_ABRV_ARRAYTYPE, name);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		s = decodetype_arrayelem(gotype);
		newrefattr(die, DW_AT_type, defgotype(s));
		break;

	case KindChan:
		die = newdie(&dwtypes, DW_ABRV_CHANTYPE, name);
		// TODO: describe ../../pkg/runtime/chan.c::struct Hchan
		break;

	case KindFunc:
		die = newdie(&dwtypes, DW_ABRV_FUNCTYPE, name);
		break;

	case KindInterface:
		die = newdie(&dwtypes, DW_ABRV_IFACETYPE, name);
		break;

	case KindMap:
		die = newdie(&dwtypes, DW_ABRV_MAPTYPE, name);
		// TODO: describe ../../pkg/runtime/hashmap.c::struct hash
		break;

	case KindPtr:
		die = newdie(&dwtypes, DW_ABRV_PTRTYPE, name);
		s = decodetype_ptrelem(gotype);
		newrefattr(die, DW_AT_type, defgotype(s));
		break;

	case KindSlice:
		die = newdie(&dwtypes, DW_ABRV_SLICETYPE, name);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		fld = newdie(die, DW_ABRV_STRUCTFIELD, "data");
		// Synthesize *elemtype if not already exists.	Maybe
		// this should be named '<*T>' to not stand in the way
		// of the real definition of *T.
		s = decodetype_arrayelem(gotype);
		elem = defgotype(s);
		ptrname = strdup(s->name + 4);	// skip "type" but leave the '.'
		ptrname[0] = '*';		//  .. to stuff in the '*'
		ptrelem = find(&dwtypes, ptrname);
		if (ptrelem == nil) {
			ptrelem = newdie(&dwtypes, DW_ABRV_PTRTYPE, ptrname);
			newrefattr(ptrelem, DW_AT_type, elem);
		} else {
			free(ptrname);
		}
		newrefattr(fld, DW_AT_type, ptrelem);
		newmemberoffsetattr(fld, 0);
		fld = newdie(die, DW_ABRV_STRUCTFIELD, "len");
		newrefattr(fld, DW_AT_type, find(&dwtypes, "<int32>"));
		newmemberoffsetattr(fld, PtrSize);
		fld = newdie(die, DW_ABRV_STRUCTFIELD, "cap");
		newrefattr(fld, DW_AT_type, find(&dwtypes, "<int32>"));
		newmemberoffsetattr(fld, PtrSize + 4);

		break;

	case KindString:
		die = newdie(&dwtypes, DW_ABRV_STRINGTYPE, name);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		fld = newdie(die, DW_ABRV_STRUCTFIELD, "str");
		newrefattr(fld, DW_AT_type, find(&dwtypes, "<byte*>"));
		newmemberoffsetattr(fld, 0);
		fld = newdie(die, DW_ABRV_STRUCTFIELD, "len");
		newrefattr(fld, DW_AT_type, find(&dwtypes, "<int32>"));
		newmemberoffsetattr(fld, PtrSize);
		break;

	case KindStruct:
		die = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, name);
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0);
		nfields = decodetype_structfieldcount(gotype);
		for (i = 0; i < nfields; ++i) {
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
		die = newdie(&dwtypes, DW_ABRV_PTRTYPE, name);
		newrefattr(die, DW_AT_type, find(&dwtypes, "void"));
		break;

	default:
		diag("definition of unknown kind %d: %s", kind, gotype->name);
		die = newdie(&dwtypes, DW_ABRV_TYPEDECL, name);
		newrefattr(die, DW_AT_type, find(&dwtypes, "<unspecified>"));
	 }

	return die;
 }

// For use with pass.c::genasmsym
static void
defdwsymb(Sym* sym, char *s, int t, vlong v, vlong size, int ver, Sym *gotype)
{
	DWDie *dv, *dt;

	if (gotype == nil) {
		return;
	}

	dv = nil;

	switch (t) {
	default:
		return;
	case 'D':
	case 'B':
		dv = newdie(&dwglobals, DW_ABRV_VARIABLE, s);
		newattr(dv, DW_AT_location, DW_CLS_ADDRESS, v, 0);
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

// TODO(lvd) For now, just append them all to the first compilation
// unit (that should be main), in the future distribute them to the
// appropriate compilation units.
static void
movetomodule(DWDie *parent)
{
	DWDie *die;

	for (die = dwroot.child->child; die->link != nil; die = die->link) /* nix */;
	die->link = parent->child;
}


/*
 * Filename fragments for the line history stack.
 */

static char **ftab;
static int ftabsize;

void
dwarfaddfrag(int n, char *frag)
{
	int s;

	if (n >= ftabsize) {
		s = ftabsize;
		ftabsize = 1 + n + (n >> 2);
		ftab = realloc(ftab, ftabsize * sizeof(ftab[0]));
		memset(ftab + s, 0, (ftabsize - s) * sizeof(ftab[0]));
	}

	if (*frag == '<')
		frag++;
	ftab[n] = frag;
}

// Returns a malloc'ed string, piecewise copied from the ftab.
static char *
decodez(char *s)
{
	int len, o;
	char *ss, *f;
	char *r, *rb, *re;

	len = 0;
	ss = s + 1;	// first is 0
	while((o = ((uint8)ss[0] << 8) | (uint8)ss[1]) != 0) {
		if (o < 0 || o >= ftabsize) {
			diag("corrupt z entry");
			return 0;
		}
		f = ftab[o];
		if (f == nil) {
			diag("corrupt z entry");
			return 0;
		}
		len += strlen(f) + 1;	// for the '/'
		ss += 2;
	}

	if (len == 0)
		return 0;

	r = malloc(len + 1);
	rb = r;
	re = rb + len + 1;

	s++;
	while((o = ((uint8)s[0] << 8) | (uint8)s[1]) != 0) {
		f = ftab[o];
		if (rb == r || rb[-1] == '/')
			rb = seprint(rb, re, "%s", f);
		else
			rb = seprint(rb, re, "/%s", f);
		s += 2;
	}
	return r;
}

/*
 * The line history itself
 */

static char **histfile;	   // [0] holds "<eof>", DW_LNS_set_file arguments must be > 0.
static int  histfilesize;
static int  histfilecap;

static void
clearhistfile(void)
{
	int i;

	// [0] holds "<eof>"
	for (i = 1; i < histfilesize; i++)
		free(histfile[i]);
	histfilesize = 0;
}

static int
addhistfile(char *zentry)
{
	char *fname;

	if (histfilesize == histfilecap) {
		histfilecap = 2 * histfilecap + 2;
		histfile = realloc(histfile, histfilecap * sizeof(char*));
	}
	if (histfilesize == 0)
		histfile[histfilesize++] = "<eof>";

	fname = decodez(zentry);
	if (fname == 0)
		return -1;
	// Don't fill with duplicates (check only top one).
	if (strcmp(fname, histfile[histfilesize-1]) == 0) {
		free(fname);
		return histfilesize - 1;
	}
	histfile[histfilesize++] = fname;
	return histfilesize - 1;
}

// Go's runtime C sources are sane, and Go sources nest only 1 level,
// so 16 should be plenty.
static struct {
	int file;
	vlong line;
} includestack[16];
static int includetop;
static vlong absline;

typedef struct Linehist Linehist;
struct Linehist {
	Linehist *link;
	vlong absline;
	vlong line;
	int file;
};

static Linehist *linehist;

static void
checknesting(void)
{
	int i;

	if (includetop < 0) {
		diag("corrupt z stack");
		errorexit();
	}
	if (includetop >= nelem(includestack)) {
		diag("nesting too deep");
		for (i = 0; i < nelem(includestack); i++)
			diag("\t%s", histfile[includestack[i].file]);
		errorexit();
	}
}

/*
 * Return false if the a->link chain contains no history, otherwise
 * returns true and finds z and Z entries in the Auto list (of a
 * Prog), and resets the history stack
 */
static int
inithist(Auto *a)
{
	Linehist *lh;

	for (; a; a = a->link)
		if (a->type == D_FILE)
			break;
	if (a==nil)
		return 0;

	// We have a new history.  They are guaranteed to come completely
	// at the beginning of the compilation unit.
	if (a->aoffset != 1) {
		diag("stray 'z' with offset %d", a->aoffset);
		return 0;
	}

	// Clear the history.
	clearhistfile();
	includetop = 0;
	includestack[includetop].file = 0;
	includestack[includetop].line = -1;
	absline = 0;
	while (linehist != nil) {
		lh = linehist->link;
		free(linehist);
		linehist = lh;
	}

	// Construct the new one.
	for (; a; a = a->link) {
		if (a->type == D_FILE) {  // 'z'
			int f = addhistfile(a->asym->name);
			if (f < 0) {	   // pop file
				includetop--;
				checknesting();
			} else if(f != includestack[includetop].file) { // pushed a new file
				includestack[includetop].line += a->aoffset - absline;
				includetop++;
				checknesting();
				includestack[includetop].file = f;
				includestack[includetop].line = 1;
			}
			absline = a->aoffset;
		} else if (a->type == D_FILE1) {  // 'Z'
			// We could just fixup the current
			// linehist->line, but there doesn't appear to
			// be a guarantee that every 'Z' is preceded
			// by it's own 'z', so do the safe thing and
			// update the stack and push a new Linehist
			// entry
			includestack[includetop].line =	 a->aoffset;
		} else
			continue;
		if (linehist == 0 || linehist->absline != absline) {
			Linehist* lh = malloc(sizeof *lh);
			lh->link = linehist;
			lh->absline = absline;
			linehist = lh;
		}
		linehist->file = includestack[includetop].file;
		linehist->line = includestack[includetop].line;
	}
	return 1;
}

static Linehist *
searchhist(vlong absline)
{
	Linehist *lh;

	for (lh = linehist; lh; lh = lh->link)
		if (lh->absline <= absline)
			break;
	return lh;
}

static int
guesslang(char *s)
{
	if(strlen(s) >= 3 && strcmp(s+strlen(s)-3, ".go") == 0)
		return DW_LANG_Go;

	return DW_LANG_C;
}

/*
 * Generate short opcodes when possible, long ones when neccesary.
 * See section 6.2.5
 */

enum {
	LINE_BASE = -1,
	LINE_RANGE = 4,
	OPCODE_BASE = 5
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
flushunit(DWDie *dwinfo, vlong pc, vlong unitstart)
{
	vlong here;

	if (dwinfo != nil && pc != 0) {
		newattr(dwinfo, DW_AT_high_pc, DW_CLS_ADDRESS, pc+1, 0);
	}

	if (unitstart >= 0) {
		cput(0);  // start extended opcode
		uleb128put(1);
		cput(DW_LNE_end_sequence);
		cflush();

		here = cpos();
		seek(cout, unitstart, 0);
		LPUT(here - unitstart - sizeof(int32));
		cflush();
		seek(cout, here, 0);
	}
}

static void
writelines(void)
{
	Prog *q;
	Sym *s;
	Auto *a;
	vlong unitstart;
	vlong pc, epc, lc, llc, lline;
	int currfile;
	int i, lang, da, dt;
	Linehist *lh;
	DWDie *dwinfo, *dwfunc, *dwvar;
	DWDie *varhash[HASHSIZE];
	char *n;

	unitstart = -1;
	epc = pc = 0;
	lc = 1;
	llc = 1;
	currfile = -1;
	lineo = cpos();
	dwinfo = nil;

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		s = cursym;

		// Look for history stack.  If we find one,
		// we're entering a new compilation unit

		if (inithist(s->autom)) {
			flushunit(dwinfo, epc, unitstart);
			unitstart = cpos();

			if(debug['v'] > 1) {
				print("dwarf writelines found %s\n", histfile[1]);
				Linehist* lh;
				for (lh = linehist; lh; lh = lh->link)
					print("\t%8lld: [%4lld]%s\n",
					      lh->absline, lh->line, histfile[lh->file]);
			}

			lang = guesslang(histfile[1]);

			dwinfo = newdie(&dwroot, DW_ABRV_COMPUNIT, strdup(histfile[1]));
			newattr(dwinfo, DW_AT_language, DW_CLS_CONSTANT,lang, 0);
			newattr(dwinfo, DW_AT_stmt_list, DW_CLS_PTR, unitstart - lineo, 0);
			newattr(dwinfo, DW_AT_low_pc, DW_CLS_ADDRESS, s->text->pc, 0);

			// Write .debug_line Line Number Program Header (sec 6.2.4)
			// Fields marked with (*) must be changed for 64-bit dwarf
			LPUT(0);   // unit_length (*), will be filled in later.
			WPUT(3);   // dwarf version (appendix F)
			LPUT(11);  // header_length (*), starting here.

			cput(1);   // minimum_instruction_length
			cput(1);   // default_is_stmt
			cput(LINE_BASE);     // line_base
			cput(LINE_RANGE);    // line_range
			cput(OPCODE_BASE);   // opcode_base (we only use 1..4)
			cput(0);   // standard_opcode_lengths[1]
			cput(1);   // standard_opcode_lengths[2]
			cput(1);   // standard_opcode_lengths[3]
			cput(1);   // standard_opcode_lengths[4]
			cput(0);   // include_directories  (empty)
			cput(0);   // file_names (empty) (emitted by DW_LNE's below)
			// header_length ends here.

			for (i=1; i < histfilesize; i++) {
				cput(0);  // start extended opcode
				uleb128put(1 + strlen(histfile[i]) + 4);
				cput(DW_LNE_define_file);
				strnput(histfile[i], strlen(histfile[i]) + 4);
				// 4 zeros: the string termination + 3 fields.
			}

			epc = pc = s->text->pc;
			currfile = 1;
			lc = 1;
			llc = 1;

			cput(0);  // start extended opcode
			uleb128put(1 + PtrSize);
			cput(DW_LNE_set_address);
			addrput(pc);
		}
		if (!s->reachable)
			continue;

		if (unitstart < 0) {
			diag("reachable code before seeing any history: %P", s->text);
			continue;
		}

		dwfunc = newdie(dwinfo, DW_ABRV_FUNCTION, s->name);
		newattr(dwfunc, DW_AT_low_pc, DW_CLS_ADDRESS, s->value, 0);
		epc = s->value + s->size;
		newattr(dwfunc, DW_AT_high_pc, DW_CLS_ADDRESS, epc, 0);
		if (s->version == 0)
			newattr(dwfunc, DW_AT_external, DW_CLS_FLAG, 1, 0);

		for(q = s->text; q != P; q = q->link) {
			lh = searchhist(q->line);
			if (lh == nil) {
				diag("corrupt history or bad absolute line: %P", q);
				continue;
			}

			if (lh->file < 1) {  // 0 is the past-EOF entry.
				// diag("instruction with line number past EOF in %s: %P", histfile[1], q);
				continue;
			}

			lline = lh->line + q->line - lh->absline;
			if (debug['v'] > 1)
				print("%6llux %s[%lld] %P\n", q->pc, histfile[lh->file], lline, q);

			if (q->line == lc)
				continue;
			if (currfile != lh->file) {
				currfile = lh->file;
				cput(DW_LNS_set_file);
				uleb128put(currfile);
			}
			putpclcdelta(q->pc - pc, lline - llc);
			pc  = q->pc;
			lc  = q->line;
			llc = lline;
		}

		da = 0;
		dwfunc->hash = varhash;	 // enable indexing of children by name
		memset(varhash, 0, sizeof varhash);

		for(a = s->autom; a; a = a->link) {
			switch (a->type) {
			case D_AUTO:
				dt = DW_ABRV_AUTO;
				break;
			case D_PARAM:
				dt = DW_ABRV_PARAM;
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
			dwvar = newdie(dwfunc, dt, n);
			newcfaoffsetattr(dwvar, a->aoffset);
			newrefattr(dwvar, DW_AT_type, defgotype(a->gotype));
			da++;
		}

		dwfunc->hash = nil;
	}

	flushunit(dwinfo, epc, unitstart);
	linesize = cpos() - lineo;
}

/*
 *  Emit .debug_frame
 */
enum
{
	CIERESERVE = 16,
	DATAALIGNMENTFACTOR = -4,	// TODO -PtrSize?
	FAKERETURNCOLUMN = 16		// TODO gdb6 doesnt like > 15?
};

static void
putpccfadelta(vlong deltapc, vlong cfa)
{
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

	cput(DW_CFA_def_cfa_offset_sf);
	sleb128put(cfa / DATAALIGNMENTFACTOR);
}

static void
writeframes(void)
{
	Prog *p, *q;
	Sym *s;
	vlong fdeo, fdesize, pad, cfa, pc;

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
		diag("CIERESERVE too small by %lld bytes.", -pad);
		errorexit();
	}
	strnput("", pad);

	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		s = cursym;
		if (!s->reachable)
			continue;

		fdeo = cpos();
		// Emit a FDE, Section 6.4.1, starting wit a placeholder.
		LPUT(0);	// length, must be multiple of PtrSize
		LPUT(0);	// Pointer to the CIE above, at offset 0
		addrput(0);	// initial location
		addrput(0);	// address range

		cfa = PtrSize;	// CFA starts at sp+PtrSize
		p = s->text;
		pc = p->pc;

		for(q = p; q->link != P; q = q->link) {
			if (q->spadj == 0)
				continue;
			cfa += q->spadj;
			putpccfadelta(q->link->pc - pc, cfa);
			pc = q->link->pc;
		}

		fdesize = cpos() - fdeo - 4;	// exclude the length field.
		pad = rnd(fdesize, PtrSize) - fdesize;
		strnput("", pad);
		fdesize += pad;
		cflush();

		// Emit the FDE header for real, Section 6.4.1.
		seek(cout, fdeo, 0);
		LPUT(fdesize);
		LPUT(0);
		addrput(p->pc);
		addrput(s->size);

		cflush();
		seek(cout, fdeo + 4 + fdesize, 0);
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

	for (compunit = dwroot.child; compunit; compunit = compunit->link) {
		unitstart = cpos();

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		LPUT(0);	// unit_length (*), will be filled in later.
		WPUT(3);	// dwarf version (appendix F)
		LPUT(0);	// debug_abbrev_offset (*)
		cput(PtrSize);	// address_size

		putdie(compunit);

		cflush();
		here = cpos();
		seek(cout, unitstart, 0);
		LPUT(here - unitstart - 4);	// exclude the length field.
		cflush();
		seek(cout, here, 0);
	}

}

/*
 *  Emit .debug_pubnames/_types.  _info must have been written before,
 *  because we need die->offs and infoo/infosize;
 */
static int
ispubname(DWDie *die) {
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
ispubtype(DWDie *die) {
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

		cflush();
		here = cpos();
		seek(cout, sectionstart, 0);
		LPUT(here - sectionstart - 4);	// exclude the length field.
		cflush();
		seek(cout, here, 0);

	}

	return sectionstart;
}

/*
 *  emit .debug_aranges.  _info must have been written before,
 *  because we need die->offs of dw_globals.
 */
static vlong
writearanges()
{
	DWDie *compunit;
	DWAttr *b, *e;
	int headersize;
	vlong sectionstart;

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
		LPUT(compunit->offs - COMPUNITHEADERSIZE);	// debug_info_offset
		cput(PtrSize);	// address_size
		cput(0);	// segment_size
		strnput("", headersize - (4+2+4+1+1));	// align to PtrSize

		addrput(b->value);
		addrput(e->value - b->value);
		addrput(0);
		addrput(0);
	}
	cflush();
	return sectionstart;
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

	mkindex(&dwroot);
	mkindex(&dwtypes);
	mkindex(&dwglobals);

	// Some types that must exist to define other ones.
	newdie(&dwtypes, DW_ABRV_NULLTYPE, "<unspecified>");
	newdie(&dwtypes, DW_ABRV_NULLTYPE, "void");
	die = newdie(&dwtypes, DW_ABRV_PTRTYPE, "unsafe.Pointer");
	newrefattr(die, DW_AT_type, find(&dwtypes, "void"));

	die = newdie(&dwtypes, DW_ABRV_BASETYPE, "<int32>");
	newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_signed, 0);
	newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, 4, 0);

	die = newdie(&dwtypes, DW_ABRV_BASETYPE, "<byte>");
	newattr(die, DW_AT_encoding,  DW_CLS_CONSTANT, DW_ATE_unsigned, 0);
	newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, 1, 0);

	die = newdie(&dwtypes, DW_ABRV_PTRTYPE, "<byte*>");
	newrefattr(die, DW_AT_type, find(&dwtypes, "<byte>"));

	genasmsym(defdwsymb);
	reversetree(&dwtypes.child);
	reversetree(&dwglobals.child);

	writeabbrev();
	writelines();
	writeframes();

	reversetree(&dwroot.child);
	movetomodule(&dwtypes);	 // TODO: put before functions
	movetomodule(&dwglobals);


	infoo = cpos();
	writeinfo();
	infoe = cpos();

	if (fwdcount > 0) {
		if (debug['v'])
			Bprint(&bso, "%5.2f dwarf pass 2.\n", cputime());
		seek(cout, infoo, 0);
		writeinfo();
		if (fwdcount > 0) {
			diag("unresolved references after first dwarf info pass");
			errorexit();
		}
		if (infoe != cpos()) {
			diag("inconsistent second dwarf info pass");
			errorexit();
		}
	}
	infosize = infoe - infoo;

	pubnameso = writepub(ispubname);
	pubtypeso = writepub(ispubtype);
	arangeso  = writearanges();

	pubnamessize = pubtypeso - pubnameso;
	pubtypessize = arangeso - pubtypeso;
	arangessize  = cpos() - arangeso;
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
	NElfStrDbg
};

vlong elfstrdbg[NElfStrDbg];

void
dwarfaddshstrings(Sym *shstrtab)
{
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
}

void
dwarfaddelfheaders(void)
{
	ElfShdr *sh;

	sh = newElfShdr(elfstrdbg[ElfStrDebugAbbrev]);
	sh->type = SHT_PROGBITS;
	sh->off = abbrevo;
	sh->size = abbrevsize;
	sh->addralign = 1;

	sh = newElfShdr(elfstrdbg[ElfStrDebugLine]);
	sh->type = SHT_PROGBITS;
	sh->off = lineo;
	sh->size = linesize;
	sh->addralign = 1;

	sh = newElfShdr(elfstrdbg[ElfStrDebugFrame]);
	sh->type = SHT_PROGBITS;
	sh->off = frameo;
	sh->size = framesize;
	sh->addralign = 1;

	sh = newElfShdr(elfstrdbg[ElfStrDebugInfo]);
	sh->type = SHT_PROGBITS;
	sh->off = infoo;
	sh->size = infosize;
	sh->addralign = 1;

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

	if (arangessize) {
		sh = newElfShdr(elfstrdbg[ElfStrDebugAranges]);
		sh->type = SHT_PROGBITS;
		sh->off = arangeso;
		sh->size = arangessize;
		sh->addralign = 1;
	}
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

	// Zero vsize segments won't be loaded in memory, even so they
	// have to be page aligned in the file.
	fakestart = abbrevo & ~0xfff;

	ms = newMachoSeg("__DWARF", 7);
	ms->fileoffset = fakestart;
	ms->filesize = abbrevo-fakestart;

	msect = newMachoSect(ms, "__debug_abbrev");
	msect->off = abbrevo;
	msect->size = abbrevsize;
	ms->filesize += msect->size;

	msect = newMachoSect(ms, "__debug_line");
	msect->off = lineo;
	msect->size = linesize;
	ms->filesize += msect->size;

	msect = newMachoSect(ms, "__debug_frame");
	msect->off = frameo;
	msect->size = framesize;
	ms->filesize += msect->size;

	msect = newMachoSect(ms, "__debug_info");
	msect->off = infoo;
	msect->size = infosize;
	ms->filesize += msect->size;

	if (pubnamessize > 0) {
		msect = newMachoSect(ms, "__debug_pubnames");
		msect->off = pubnameso;
		msect->size = pubnamessize;
		ms->filesize += msect->size;
	}

	if (pubtypessize > 0) {
		msect = newMachoSect(ms, "__debug_pubtypes");
		msect->off = pubtypeso;
		msect->size = pubtypessize;
		ms->filesize += msect->size;
	}

	if (arangessize > 0) {
		msect = newMachoSect(ms, "__debug_aranges");
		msect->off = arangeso;
		msect->size = arangessize;
		ms->filesize += msect->size;
	}
}
