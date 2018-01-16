// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build 386 amd64

package runtime

import "unsafe"

// Look up symbols in the Linux vDSO.

// This code was originally based on the sample Linux vDSO parser at
// https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/tools/testing/selftests/vDSO/parse_vdso.c

// This implements the ELF dynamic linking spec at
// http://sco.com/developers/gabi/latest/ch5.dynamic.html

// The version section is documented at
// http://refspecs.linuxfoundation.org/LSB_3.2.0/LSB-Core-generic/LSB-Core-generic/symversion.html

const (
	_AT_SYSINFO_EHDR = 33

	_PT_LOAD    = 1 /* Loadable program segment */
	_PT_DYNAMIC = 2 /* Dynamic linking information */

	_DT_NULL     = 0          /* Marks end of dynamic section */
	_DT_HASH     = 4          /* Dynamic symbol hash table */
	_DT_STRTAB   = 5          /* Address of string table */
	_DT_SYMTAB   = 6          /* Address of symbol table */
	_DT_GNU_HASH = 0x6ffffef5 /* GNU-style dynamic symbol hash table */
	_DT_VERSYM   = 0x6ffffff0
	_DT_VERDEF   = 0x6ffffffc

	_VER_FLG_BASE = 0x1 /* Version definition of file itself */

	_SHN_UNDEF = 0 /* Undefined section */

	_SHT_DYNSYM = 11 /* Dynamic linker symbol table */

	_STT_FUNC = 2 /* Symbol is a code object */

	_STB_GLOBAL = 1 /* Global symbol */
	_STB_WEAK   = 2 /* Weak symbol */

	_EI_NIDENT = 16

	// Maximum indices for the array types used when traversing the vDSO ELF structures.
	// Computed from architecture-specific max provided by vdso_linux_*.go
	vdsoSymTabSize     = vdsoArrayMax / unsafe.Sizeof(elfSym{})
	vdsoDynSize        = vdsoArrayMax / unsafe.Sizeof(elfDyn{})
	vdsoSymStringsSize = vdsoArrayMax     // byte
	vdsoVerSymSize     = vdsoArrayMax / 2 // uint16
	vdsoHashSize       = vdsoArrayMax / 4 // uint32

	// vdsoBloomSizeScale is a scaling factor for gnuhash tables which are uint32 indexed,
	// but contain uintptrs
	vdsoBloomSizeScale = unsafe.Sizeof(uintptr(0)) / 4 // uint32
)

/* How to extract and insert information held in the st_info field.  */
func _ELF_ST_BIND(val byte) byte { return val >> 4 }
func _ELF_ST_TYPE(val byte) byte { return val & 0xf }

type symbol_key struct {
	name     string
	sym_hash uint32
	gnu_hash uint32
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
	symtab     *[vdsoSymTabSize]elfSym
	symstrings *[vdsoSymStringsSize]byte
	chain      []uint32
	bucket     []uint32
	symOff     uint32
	isGNUHash  bool

	/* Version table */
	versym *[vdsoVerSymSize]uint16
	verdef *elfVerdef
}

var linux26 = version_key{"LINUX_2.6", 0x3ae75f6}

// see vdso_linux_*.go for sym_keys[] and __vdso_* vars

func vdso_init_from_sysinfo_ehdr(info *vdso_info, hdr *elfEhdr) {
	info.valid = false
	info.load_addr = uintptr(unsafe.Pointer(hdr))

	pt := unsafe.Pointer(info.load_addr + uintptr(hdr.e_phoff))

	// We need two things from the segment table: the load offset
	// and the dynamic table.
	var found_vaddr bool
	var dyn *[vdsoDynSize]elfDyn
	for i := uint16(0); i < hdr.e_phnum; i++ {
		pt := (*elfPhdr)(add(pt, uintptr(i)*unsafe.Sizeof(elfPhdr{})))
		switch pt.p_type {
		case _PT_LOAD:
			if !found_vaddr {
				found_vaddr = true
				info.load_offset = info.load_addr + uintptr(pt.p_offset-pt.p_vaddr)
			}

		case _PT_DYNAMIC:
			dyn = (*[vdsoDynSize]elfDyn)(unsafe.Pointer(info.load_addr + uintptr(pt.p_offset)))
		}
	}

	if !found_vaddr || dyn == nil {
		return // Failed
	}

	// Fish out the useful bits of the dynamic table.

	var hash, gnuhash *[vdsoHashSize]uint32
	info.symstrings = nil
	info.symtab = nil
	info.versym = nil
	info.verdef = nil
	for i := 0; dyn[i].d_tag != _DT_NULL; i++ {
		dt := &dyn[i]
		p := info.load_offset + uintptr(dt.d_val)
		switch dt.d_tag {
		case _DT_STRTAB:
			info.symstrings = (*[vdsoSymStringsSize]byte)(unsafe.Pointer(p))
		case _DT_SYMTAB:
			info.symtab = (*[vdsoSymTabSize]elfSym)(unsafe.Pointer(p))
		case _DT_HASH:
			hash = (*[vdsoHashSize]uint32)(unsafe.Pointer(p))
		case _DT_GNU_HASH:
			gnuhash = (*[vdsoHashSize]uint32)(unsafe.Pointer(p))
		case _DT_VERSYM:
			info.versym = (*[vdsoVerSymSize]uint16)(unsafe.Pointer(p))
		case _DT_VERDEF:
			info.verdef = (*elfVerdef)(unsafe.Pointer(p))
		}
	}

	if info.symstrings == nil || info.symtab == nil || (hash == nil && gnuhash == nil) {
		return // Failed
	}

	if info.verdef == nil {
		info.versym = nil
	}

	if gnuhash != nil {
		// Parse the GNU hash table header.
		nbucket := gnuhash[0]
		info.symOff = gnuhash[1]
		bloomSize := gnuhash[2]
		info.bucket = gnuhash[4+bloomSize*uint32(vdsoBloomSizeScale):][:nbucket]
		info.chain = gnuhash[4+bloomSize*uint32(vdsoBloomSizeScale)+nbucket:]
		info.isGNUHash = true
	} else {
		// Parse the hash table header.
		nbucket := hash[0]
		nchain := hash[1]
		info.bucket = hash[2 : 2+nbucket]
		info.chain = hash[2+nbucket : 2+nbucket+nchain]
	}

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
			aux := (*elfVerdaux)(add(unsafe.Pointer(def), uintptr(def.vd_aux)))
			if def.vd_hash == ver.ver_hash && ver.version == gostringnocopy(&info.symstrings[aux.vda_name]) {
				return int32(def.vd_ndx & 0x7fff)
			}
		}

		if def.vd_next == 0 {
			break
		}
		def = (*elfVerdef)(add(unsafe.Pointer(def), uintptr(def.vd_next)))
	}

	return -1 // cannot match any version
}

func vdso_parse_symbols(info *vdso_info, version int32) {
	if !info.valid {
		return
	}

	apply := func(symIndex uint32, k symbol_key) bool {
		sym := &info.symtab[symIndex]
		typ := _ELF_ST_TYPE(sym.st_info)
		bind := _ELF_ST_BIND(sym.st_info)
		if typ != _STT_FUNC || bind != _STB_GLOBAL && bind != _STB_WEAK || sym.st_shndx == _SHN_UNDEF {
			return false
		}
		if k.name != gostringnocopy(&info.symstrings[sym.st_name]) {
			return false
		}

		// Check symbol version.
		if info.versym != nil && version != 0 && int32(info.versym[symIndex]&0x7fff) != version {
			return false
		}

		*k.ptr = info.load_offset + uintptr(sym.st_value)
		return true
	}

	if !info.isGNUHash {
		// Old-style DT_HASH table.
		for _, k := range sym_keys {
			for chain := info.bucket[k.sym_hash%uint32(len(info.bucket))]; chain != 0; chain = info.chain[chain] {
				if apply(chain, k) {
					break
				}
			}
		}
		return
	}

	// New-style DT_GNU_HASH table.
	for _, k := range sym_keys {
		symIndex := info.bucket[k.gnu_hash%uint32(len(info.bucket))]
		if symIndex < info.symOff {
			continue
		}
		for ; ; symIndex++ {
			hash := info.chain[symIndex-info.symOff]
			if hash|1 == k.gnu_hash|1 {
				// Found a hash match.
				if apply(symIndex, k) {
					break
				}
			}
			if hash&1 != 0 {
				// End of chain.
				break
			}
		}
	}
}

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_SYSINFO_EHDR:
		if val == 0 {
			// Something went wrong
			return
		}
		var info vdso_info
		// TODO(rsc): I don't understand why the compiler thinks info escapes
		// when passed to the three functions below.
		info1 := (*vdso_info)(noescape(unsafe.Pointer(&info)))
		vdso_init_from_sysinfo_ehdr(info1, (*elfEhdr)(unsafe.Pointer(val)))
		vdso_parse_symbols(info1, vdso_find_version(info1, &linux26))
	}
}
