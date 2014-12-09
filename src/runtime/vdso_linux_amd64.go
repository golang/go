// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Look up symbols in the Linux vDSO.

// This code was originally based on the sample Linux vDSO parser at
// https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/Documentation/vDSO/parse_vdso.c

// This implements the ELF dynamic linking spec at
// http://sco.com/developers/gabi/latest/ch5.dynamic.html

// The version section is documented at
// http://refspecs.linuxfoundation.org/LSB_3.2.0/LSB-Core-generic/LSB-Core-generic/symversion.html

const (
	_AT_RANDOM       = 25
	_AT_SYSINFO_EHDR = 33
	_AT_NULL         = 0 /* End of vector */

	_PT_LOAD    = 1 /* Loadable program segment */
	_PT_DYNAMIC = 2 /* Dynamic linking information */

	_DT_NULL   = 0 /* Marks end of dynamic section */
	_DT_HASH   = 4 /* Dynamic symbol hash table */
	_DT_STRTAB = 5 /* Address of string table */
	_DT_SYMTAB = 6 /* Address of symbol table */
	_DT_VERSYM = 0x6ffffff0
	_DT_VERDEF = 0x6ffffffc

	_VER_FLG_BASE = 0x1 /* Version definition of file itself */

	_SHN_UNDEF = 0 /* Undefined section */

	_SHT_DYNSYM = 11 /* Dynamic linker symbol table */

	_STT_FUNC = 2 /* Symbol is a code object */

	_STB_GLOBAL = 1 /* Global symbol */
	_STB_WEAK   = 2 /* Weak symbol */

	_EI_NIDENT = 16
)

/* How to extract and insert information held in the st_info field.  */
func _ELF64_ST_BIND(val byte) byte { return val >> 4 }
func _ELF64_ST_TYPE(val byte) byte { return val & 0xf }

type elf64Sym struct {
	st_name  uint32
	st_info  byte
	st_other byte
	st_shndx uint16
	st_value uint64
	st_size  uint64
}

type elf64Verdef struct {
	vd_version uint16 /* Version revision */
	vd_flags   uint16 /* Version information */
	vd_ndx     uint16 /* Version Index */
	vd_cnt     uint16 /* Number of associated aux entries */
	vd_hash    uint32 /* Version name hash value */
	vd_aux     uint32 /* Offset in bytes to verdaux array */
	vd_next    uint32 /* Offset in bytes to next verdef entry */
}

type elf64Ehdr struct {
	e_ident     [_EI_NIDENT]byte /* Magic number and other info */
	e_type      uint16           /* Object file type */
	e_machine   uint16           /* Architecture */
	e_version   uint32           /* Object file version */
	e_entry     uint64           /* Entry point virtual address */
	e_phoff     uint64           /* Program header table file offset */
	e_shoff     uint64           /* Section header table file offset */
	e_flags     uint32           /* Processor-specific flags */
	e_ehsize    uint16           /* ELF header size in bytes */
	e_phentsize uint16           /* Program header table entry size */
	e_phnum     uint16           /* Program header table entry count */
	e_shentsize uint16           /* Section header table entry size */
	e_shnum     uint16           /* Section header table entry count */
	e_shstrndx  uint16           /* Section header string table index */
}

type elf64Phdr struct {
	p_type   uint32 /* Segment type */
	p_flags  uint32 /* Segment flags */
	p_offset uint64 /* Segment file offset */
	p_vaddr  uint64 /* Segment virtual address */
	p_paddr  uint64 /* Segment physical address */
	p_filesz uint64 /* Segment size in file */
	p_memsz  uint64 /* Segment size in memory */
	p_align  uint64 /* Segment alignment */
}

type elf64Shdr struct {
	sh_name      uint32 /* Section name (string tbl index) */
	sh_type      uint32 /* Section type */
	sh_flags     uint64 /* Section flags */
	sh_addr      uint64 /* Section virtual addr at execution */
	sh_offset    uint64 /* Section file offset */
	sh_size      uint64 /* Section size in bytes */
	sh_link      uint32 /* Link to another section */
	sh_info      uint32 /* Additional section information */
	sh_addralign uint64 /* Section alignment */
	sh_entsize   uint64 /* Entry size if section holds table */
}

type elf64Dyn struct {
	d_tag int64  /* Dynamic entry type */
	d_val uint64 /* Integer value */
}

type elf64Verdaux struct {
	vda_name uint32 /* Version or dependency names */
	vda_next uint32 /* Offset in bytes to next verdaux entry */
}

type elf64Auxv struct {
	a_type uint64 /* Entry type */
	a_val  uint64 /* Integer value */
}

type symbol_key struct {
	name     string
	sym_hash uint32
	ptr      *uintptr
}

type version_key struct {
	version  string
	ver_hash uint32
}

type vdso_info struct {
	valid bool

	/* Load information */
	load_addr   uintptr
	load_offset uintptr /* load_addr - recorded vaddr */

	/* Symbol table */
	symtab     *[1 << 32]elf64Sym
	symstrings *[1 << 32]byte
	chain      []uint32
	bucket     []uint32

	/* Version table */
	versym *[1 << 32]uint16
	verdef *elf64Verdef
}

var linux26 = version_key{"LINUX_2.6", 0x3ae75f6}

var sym_keys = []symbol_key{
	{"__vdso_time", 0xa33c485, &__vdso_time_sym},
	{"__vdso_gettimeofday", 0x315ca59, &__vdso_gettimeofday_sym},
	{"__vdso_clock_gettime", 0xd35ec75, &__vdso_clock_gettime_sym},
}

// initialize with vsyscall fallbacks
var (
	__vdso_time_sym          uintptr = 0xffffffffff600400
	__vdso_gettimeofday_sym  uintptr = 0xffffffffff600000
	__vdso_clock_gettime_sym uintptr = 0
)

func vdso_init_from_sysinfo_ehdr(info *vdso_info, hdr *elf64Ehdr) {
	info.valid = false
	info.load_addr = uintptr(unsafe.Pointer(hdr))

	pt := unsafe.Pointer(info.load_addr + uintptr(hdr.e_phoff))

	// We need two things from the segment table: the load offset
	// and the dynamic table.
	var found_vaddr bool
	var dyn *[1 << 20]elf64Dyn
	for i := uint16(0); i < hdr.e_phnum; i++ {
		pt := (*elf64Phdr)(add(pt, uintptr(i)*unsafe.Sizeof(elf64Phdr{})))
		switch pt.p_type {
		case _PT_LOAD:
			if !found_vaddr {
				found_vaddr = true
				info.load_offset = info.load_addr + uintptr(pt.p_offset-pt.p_vaddr)
			}

		case _PT_DYNAMIC:
			dyn = (*[1 << 20]elf64Dyn)(unsafe.Pointer(info.load_addr + uintptr(pt.p_offset)))
		}
	}

	if !found_vaddr || dyn == nil {
		return // Failed
	}

	// Fish out the useful bits of the dynamic table.

	var hash *[1 << 30]uint32
	hash = nil
	info.symstrings = nil
	info.symtab = nil
	info.versym = nil
	info.verdef = nil
	for i := 0; dyn[i].d_tag != _DT_NULL; i++ {
		dt := &dyn[i]
		p := info.load_offset + uintptr(dt.d_val)
		switch dt.d_tag {
		case _DT_STRTAB:
			info.symstrings = (*[1 << 32]byte)(unsafe.Pointer(p))
		case _DT_SYMTAB:
			info.symtab = (*[1 << 32]elf64Sym)(unsafe.Pointer(p))
		case _DT_HASH:
			hash = (*[1 << 30]uint32)(unsafe.Pointer(p))
		case _DT_VERSYM:
			info.versym = (*[1 << 32]uint16)(unsafe.Pointer(p))
		case _DT_VERDEF:
			info.verdef = (*elf64Verdef)(unsafe.Pointer(p))
		}
	}

	if info.symstrings == nil || info.symtab == nil || hash == nil {
		return // Failed
	}

	if info.verdef == nil {
		info.versym = nil
	}

	// Parse the hash table header.
	nbucket := hash[0]
	nchain := hash[1]
	info.bucket = hash[2 : 2+nbucket]
	info.chain = hash[2+nbucket : 2+nbucket+nchain]

	// That's all we need.
	info.valid = true
}

func vdso_find_version(info *vdso_info, ver *version_key) int32 {
	if !info.valid {
		return 0
	}

	def := info.verdef
	for {
		if def.vd_flags&_VER_FLG_BASE == 0 {
			aux := (*elf64Verdaux)(add(unsafe.Pointer(def), uintptr(def.vd_aux)))
			if def.vd_hash == ver.ver_hash && ver.version == gostringnocopy(&info.symstrings[aux.vda_name]) {
				return int32(def.vd_ndx & 0x7fff)
			}
		}

		if def.vd_next == 0 {
			break
		}
		def = (*elf64Verdef)(add(unsafe.Pointer(def), uintptr(def.vd_next)))
	}

	return -1 // can not match any version
}

func vdso_parse_symbols(info *vdso_info, version int32) {
	if !info.valid {
		return
	}

	for _, k := range sym_keys {
		for chain := info.bucket[k.sym_hash%uint32(len(info.bucket))]; chain != 0; chain = info.chain[chain] {
			sym := &info.symtab[chain]
			typ := _ELF64_ST_TYPE(sym.st_info)
			bind := _ELF64_ST_BIND(sym.st_info)
			if typ != _STT_FUNC || bind != _STB_GLOBAL && bind != _STB_WEAK || sym.st_shndx == _SHN_UNDEF {
				continue
			}
			if k.name != gostringnocopy(&info.symstrings[sym.st_name]) {
				continue
			}

			// Check symbol version.
			if info.versym != nil && version != 0 && int32(info.versym[chain]&0x7fff) != version {
				continue
			}

			*k.ptr = info.load_offset + uintptr(sym.st_value)
			break
		}
	}
}

func sysargs(argc int32, argv **byte) {
	n := argc + 1

	// skip envp to get to ELF auxiliary vector.
	for argv_index(argv, n) != nil {
		n++
	}

	// skip NULL separator
	n++

	// now argv+n is auxv
	auxv := (*[1 << 32]elf64Auxv)(add(unsafe.Pointer(argv), uintptr(n)*ptrSize))

	for i := 0; auxv[i].a_type != _AT_NULL; i++ {
		av := &auxv[i]
		switch av.a_type {
		case _AT_SYSINFO_EHDR:
			if av.a_val == 0 {
				// Something went wrong
				continue
			}
			var info vdso_info
			// TODO(rsc): I don't understand why the compiler thinks info escapes
			// when passed to the three functions below.
			info1 := (*vdso_info)(noescape(unsafe.Pointer(&info)))
			vdso_init_from_sysinfo_ehdr(info1, (*elf64Ehdr)(unsafe.Pointer(uintptr(av.a_val))))
			vdso_parse_symbols(info1, vdso_find_version(info1, &linux26))

		case _AT_RANDOM:
			startupRandomData = (*[16]byte)(unsafe.Pointer(uintptr(av.a_val)))[:]
		}
	}
}
