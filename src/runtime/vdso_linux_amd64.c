// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "textflag.h"

// Look up symbols in the Linux vDSO.

// This code was originally based on the sample Linux vDSO parser at
// https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/Documentation/vDSO/parse_vdso.c

// This implements the ELF dynamic linking spec at
// http://sco.com/developers/gabi/latest/ch5.dynamic.html

// The version section is documented at
// http://refspecs.linuxfoundation.org/LSB_3.2.0/LSB-Core-generic/LSB-Core-generic/symversion.html

#define AT_RANDOM 25
#define AT_SYSINFO_EHDR 33
#define AT_NULL	0    /* End of vector */
#define PT_LOAD	1    /* Loadable program segment */
#define PT_DYNAMIC 2 /* Dynamic linking information */
#define DT_NULL 0    /* Marks end of dynamic section */
#define DT_HASH 4    /* Dynamic symbol hash table */
#define DT_STRTAB 5  /* Address of string table */
#define DT_SYMTAB 6  /* Address of symbol table */
#define DT_VERSYM 0x6ffffff0
#define	DT_VERDEF 0x6ffffffc

#define VER_FLG_BASE 0x1 /* Version definition of file itself */
#define SHN_UNDEF 0      /* Undefined section */
#define SHT_DYNSYM 11    /* Dynamic linker symbol table */
#define STT_FUNC 2       /* Symbol is a code object */
#define STB_GLOBAL 1     /* Global symbol */
#define STB_WEAK 2       /* Weak symbol */

/* How to extract and insert information held in the st_info field.  */
#define ELF64_ST_BIND(val) (((byte) (val)) >> 4)
#define ELF64_ST_TYPE(val) ((val) & 0xf)

#define EI_NIDENT (16)

typedef uint16 Elf64_Half;
typedef uint32 Elf64_Word;
typedef	int32  Elf64_Sword;
typedef uint64 Elf64_Xword;
typedef	int64  Elf64_Sxword;
typedef uint64 Elf64_Addr;
typedef uint64 Elf64_Off;
typedef uint16 Elf64_Section;
typedef Elf64_Half Elf64_Versym;


typedef struct Elf64_Sym
{
	Elf64_Word st_name;
	byte st_info;
	byte st_other;
	Elf64_Section st_shndx;
	Elf64_Addr st_value;
	Elf64_Xword st_size;
} Elf64_Sym;

typedef struct Elf64_Verdef
{
	Elf64_Half vd_version; /* Version revision */
	Elf64_Half vd_flags;   /* Version information */
	Elf64_Half vd_ndx;     /* Version Index */
	Elf64_Half vd_cnt;     /* Number of associated aux entries */
	Elf64_Word vd_hash;    /* Version name hash value */
	Elf64_Word vd_aux;     /* Offset in bytes to verdaux array */
	Elf64_Word vd_next;    /* Offset in bytes to next verdef entry */
} Elf64_Verdef;

typedef struct Elf64_Ehdr
{
	byte e_ident[EI_NIDENT]; /* Magic number and other info */
	Elf64_Half e_type;       /* Object file type */
	Elf64_Half e_machine;    /* Architecture */
	Elf64_Word e_version;    /* Object file version */
	Elf64_Addr e_entry;      /* Entry point virtual address */
	Elf64_Off e_phoff;       /* Program header table file offset */
	Elf64_Off e_shoff;       /* Section header table file offset */
	Elf64_Word e_flags;      /* Processor-specific flags */
	Elf64_Half e_ehsize;     /* ELF header size in bytes */
	Elf64_Half e_phentsize;  /* Program header table entry size */
	Elf64_Half e_phnum;      /* Program header table entry count */
	Elf64_Half e_shentsize;  /* Section header table entry size */
	Elf64_Half e_shnum;      /* Section header table entry count */
	Elf64_Half e_shstrndx;   /* Section header string table index */
} Elf64_Ehdr;

typedef struct Elf64_Phdr
{
	Elf64_Word p_type;    /* Segment type */
	Elf64_Word p_flags;   /* Segment flags */
	Elf64_Off p_offset;   /* Segment file offset */
	Elf64_Addr p_vaddr;   /* Segment virtual address */
	Elf64_Addr p_paddr;   /* Segment physical address */
	Elf64_Xword p_filesz; /* Segment size in file */
	Elf64_Xword p_memsz;  /* Segment size in memory */
	Elf64_Xword p_align;  /* Segment alignment */
} Elf64_Phdr;

typedef struct Elf64_Shdr
{
	Elf64_Word sh_name;       /* Section name (string tbl index) */
	Elf64_Word sh_type;       /* Section type */
	Elf64_Xword sh_flags;     /* Section flags */
	Elf64_Addr sh_addr;       /* Section virtual addr at execution */
	Elf64_Off sh_offset;      /* Section file offset */
	Elf64_Xword sh_size;      /* Section size in bytes */
	Elf64_Word sh_link;       /* Link to another section */
	Elf64_Word sh_info;       /* Additional section information */
	Elf64_Xword sh_addralign; /* Section alignment */
	Elf64_Xword sh_entsize;   /* Entry size if section holds table */
} Elf64_Shdr;

typedef struct Elf64_Dyn
{
	Elf64_Sxword d_tag; /* Dynamic entry type */
	union
	{
		Elf64_Xword d_val;  /* Integer value */
		Elf64_Addr d_ptr;   /* Address value */
	} d_un;
} Elf64_Dyn;

typedef struct Elf64_Verdaux
{
	Elf64_Word vda_name; /* Version or dependency names */
	Elf64_Word vda_next; /* Offset in bytes to next verdaux entry */
} Elf64_Verdaux;

typedef struct Elf64_auxv_t
{
	uint64 a_type;        /* Entry type */
	union
	{
		uint64 a_val; /* Integer value */
	} a_un;
} Elf64_auxv_t;


typedef struct symbol_key {
	byte* name;
	int32 sym_hash;
	void** var_ptr;
} symbol_key;

typedef struct version_key {
	byte* version;
	int32 ver_hash;
} version_key;

struct vdso_info {
	bool valid;

	/* Load information */
	uintptr load_addr;
	uintptr load_offset;  /* load_addr - recorded vaddr */

	/* Symbol table */
	Elf64_Sym *symtab;
	const byte *symstrings;
	Elf64_Word *bucket, *chain;
	Elf64_Word nbucket, nchain;

	/* Version table */
	Elf64_Versym *versym;
	Elf64_Verdef *verdef;
};

#pragma dataflag NOPTR
static version_key linux26 = { (byte*)"LINUX_2.6", 0x3ae75f6 };

// initialize with vsyscall fallbacks
#pragma dataflag NOPTR
void* runtime·__vdso_time_sym = (void*)0xffffffffff600400ULL;
#pragma dataflag NOPTR
void* runtime·__vdso_gettimeofday_sym = (void*)0xffffffffff600000ULL;
#pragma dataflag NOPTR
void* runtime·__vdso_clock_gettime_sym = (void*)0;

#pragma dataflag NOPTR
static symbol_key sym_keys[] = {
	{ (byte*)"__vdso_time", 0xa33c485, &runtime·__vdso_time_sym },
	{ (byte*)"__vdso_gettimeofday", 0x315ca59, &runtime·__vdso_gettimeofday_sym },
	{ (byte*)"__vdso_clock_gettime", 0xd35ec75, &runtime·__vdso_clock_gettime_sym },
};

static void
vdso_init_from_sysinfo_ehdr(struct vdso_info *vdso_info, Elf64_Ehdr* hdr)
{
	uint64 i;
	bool found_vaddr = false;
	Elf64_Phdr *pt;
	Elf64_Dyn *dyn;
	Elf64_Word *hash;

	vdso_info->valid = false;
	vdso_info->load_addr = (uintptr) hdr;

	pt = (Elf64_Phdr*)(vdso_info->load_addr + hdr->e_phoff);
	dyn = nil;

	// We need two things from the segment table: the load offset
	// and the dynamic table.
	for(i=0; i<hdr->e_phnum; i++) {
		if(pt[i].p_type == PT_LOAD && found_vaddr == false) {
			found_vaddr = true;
			vdso_info->load_offset =	(uintptr)hdr
				+ (uintptr)pt[i].p_offset
				- (uintptr)pt[i].p_vaddr;
		} else if(pt[i].p_type == PT_DYNAMIC) {
			dyn = (Elf64_Dyn*)((uintptr)hdr + pt[i].p_offset);
		}
	}

	if(found_vaddr == false || dyn == nil)
		return;  // Failed

	// Fish out the useful bits of the dynamic table.
	hash = nil;
	vdso_info->symstrings = nil;
	vdso_info->symtab = nil;
	vdso_info->versym = nil;
	vdso_info->verdef = nil;
	for(i=0; dyn[i].d_tag!=DT_NULL; i++) {
		switch(dyn[i].d_tag) {
		case DT_STRTAB:
			vdso_info->symstrings = (const byte *)
				((uintptr)dyn[i].d_un.d_ptr
				 + vdso_info->load_offset);
			break;
		case DT_SYMTAB:
			vdso_info->symtab = (Elf64_Sym *)
				((uintptr)dyn[i].d_un.d_ptr
				 + vdso_info->load_offset);
			break;
		case DT_HASH:
			hash = (Elf64_Word *)
			  ((uintptr)dyn[i].d_un.d_ptr
			   + vdso_info->load_offset);
			break;
		case DT_VERSYM:
			vdso_info->versym = (Elf64_Versym *)
				((uintptr)dyn[i].d_un.d_ptr
				 + vdso_info->load_offset);
			break;
		case DT_VERDEF:
			vdso_info->verdef = (Elf64_Verdef *)
				((uintptr)dyn[i].d_un.d_ptr
				 + vdso_info->load_offset);
			break;
		}
	}
	if(vdso_info->symstrings == nil || vdso_info->symtab == nil || hash == nil)
		return;  // Failed

	if(vdso_info->verdef == nil)
		vdso_info->versym = 0;

	// Parse the hash table header.
	vdso_info->nbucket = hash[0];
	vdso_info->nchain = hash[1];
	vdso_info->bucket = &hash[2];
	vdso_info->chain = &hash[vdso_info->nbucket + 2];

	// That's all we need.
	vdso_info->valid = true;
}

static int32
vdso_find_version(struct vdso_info *vdso_info, version_key* ver)
{
	if(vdso_info->valid == false) {
		return 0;
	}
	Elf64_Verdef *def = vdso_info->verdef;
	while(true) {
		if((def->vd_flags & VER_FLG_BASE) == 0) {
			Elf64_Verdaux *aux = (Elf64_Verdaux*)((byte *)def + def->vd_aux);
			if(def->vd_hash == ver->ver_hash &&
				runtime·strcmp(ver->version, vdso_info->symstrings + aux->vda_name) == 0) {
				return def->vd_ndx & 0x7fff;
			}
		}

		if(def->vd_next == 0) {
			break;
		}
		def = (Elf64_Verdef *)((byte *)def + def->vd_next);
	}
	return -1; // can not match any version
}

static void
vdso_parse_symbols(struct vdso_info *vdso_info, int32 version)
{
	int32 i;
	Elf64_Word chain;
	Elf64_Sym *sym;

	if(vdso_info->valid == false)
		return;

	for(i=0; i<nelem(sym_keys); i++) {
		for(chain = vdso_info->bucket[sym_keys[i].sym_hash % vdso_info->nbucket];
			chain != 0; chain = vdso_info->chain[chain]) {

			sym = &vdso_info->symtab[chain];
			if(ELF64_ST_TYPE(sym->st_info) != STT_FUNC)
				continue;
			if(ELF64_ST_BIND(sym->st_info) != STB_GLOBAL &&
				 ELF64_ST_BIND(sym->st_info) != STB_WEAK)
				continue;
			if(sym->st_shndx == SHN_UNDEF)
				continue;
			if(runtime·strcmp(sym_keys[i].name, vdso_info->symstrings + sym->st_name) != 0)
				continue;

			// Check symbol version.
			if(vdso_info->versym != nil && version != 0
				&& vdso_info->versym[chain] & 0x7fff != version)
				continue;

			*sym_keys[i].var_ptr = (void *)(vdso_info->load_offset + sym->st_value);
			break;
		}
	}
}

static void
runtime·linux_setup_vdso(int32 argc, uint8** argv)
{
	struct vdso_info vdso_info;

	// skip argvc
	byte **p = argv;
	p = &p[argc+1];

	// skip envp to get to ELF auxiliary vector.
	for(; *p!=0; p++) {}

	// skip NULL separator
	p++;

	// now, p points to auxv
	Elf64_auxv_t *elf_auxv = (Elf64_auxv_t*) p;

	for(int32 i=0; elf_auxv[i].a_type!=AT_NULL; i++) {
		if(elf_auxv[i].a_type == AT_SYSINFO_EHDR) {
			if(elf_auxv[i].a_un.a_val == 0) {
				// Something went wrong
				continue;
			}
			vdso_init_from_sysinfo_ehdr(&vdso_info, (Elf64_Ehdr*)elf_auxv[i].a_un.a_val);
			vdso_parse_symbols(&vdso_info, vdso_find_version(&vdso_info, &linux26));
			continue;
		}
		if(elf_auxv[i].a_type == AT_RANDOM) {
		        runtime·startup_random_data = (byte*)elf_auxv[i].a_un.a_val;
		        runtime·startup_random_data_len = 16;
			continue;
		}
	}
}

void (*runtime·sysargs)(int32, uint8**) = runtime·linux_setup_vdso;
