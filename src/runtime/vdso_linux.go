// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || amd64 || arm || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x)

package runtime

import "unsafe"

// Look up symbols in the Linux vDSO.

// This code was originally based on the sample Linux vDSO parser at
// https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/tools/testing/selftests/vDSO/parse_vdso.c

// This implements the ELF dynamic linking spec at
// http://sco.com/developers/gabi/latest/ch5.dynamic.html

// The version section is documented at
// https://refspecs.linuxfoundation.org/LSB_3.2.0/LSB-Core-generic/LSB-Core-generic/symversion.html

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

	_STT_NOTYPE = 0 /* Symbol type is not specified */

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

type vdsoSymbolKey struct {
	name    string
	symHash uint32
	gnuHash uint32
	ptr     *uintptr
}

type vdsoVersionKey struct {
	version string
	verHash uint32
}

type vdsoInfo struct {
	valid bool

	/* Load information */
	loadAddr   uintptr
	loadOffset uintptr /* loadAddr - recorded vaddr */

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

// see vdso_linux_*.go for vdsoSymbolKeys[] and vdso*Sym vars

func vdsoInitFromSysinfoEhdr(info *vdsoInfo, hdr *elfEhdr) {
	info.valid = false
	info.loadAddr = uintptr(unsafe.Pointer(hdr))

	pt := unsafe.Pointer(info.loadAddr + uintptr(hdr.e_phoff))

	// We need two things from the segment table: the load offset
	// and the dynamic table.
	var foundVaddr bool
	var dyn *[vdsoDynSize]elfDyn
	for i := uint16(0); i < hdr.e_phnum; i++ {
		pt := (*elfPhdr)(add(pt, uintptr(i)*unsafe.Sizeof(elfPhdr{})))
		switch pt.p_type {
		case _PT_LOAD:
			if !foundVaddr {
				foundVaddr = true
				info.loadOffset = info.loadAddr + uintptr(pt.p_offset-pt.p_vaddr)
			}

		case _PT_DYNAMIC:
			dyn = (*[vdsoDynSize]elfDyn)(unsafe.Pointer(info.loadAddr + uintptr(pt.p_offset)))
		}
	}

	if !foundVaddr || dyn == nil {
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
		p := info.loadOffset + uintptr(dt.d_val)
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

func vdsoFindVersion(info *vdsoInfo, ver *vdsoVersionKey) int32 {
	if !info.valid {
		return 0
	}

	def := info.verdef
	for {
		if def.vd_flags&_VER_FLG_BASE == 0 {
			aux := (*elfVerdaux)(add(unsafe.Pointer(def), uintptr(def.vd_aux)))
			if def.vd_hash == ver.verHash && ver.version == gostringnocopy(&info.symstrings[aux.vda_name]) {
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

func vdsoParseSymbols(info *vdsoInfo, version int32) {
	if !info.valid {
		return
	}

	apply := func(symIndex uint32, k vdsoSymbolKey) bool {
		sym := &info.symtab[symIndex]
		typ := _ELF_ST_TYPE(sym.st_info)
		bind := _ELF_ST_BIND(sym.st_info)
		// On ppc64x, VDSO functions are of type _STT_NOTYPE.
		if typ != _STT_FUNC && typ != _STT_NOTYPE || bind != _STB_GLOBAL && bind != _STB_WEAK || sym.st_shndx == _SHN_UNDEF {
			return false
		}
		if k.name != gostringnocopy(&info.symstrings[sym.st_name]) {
			return false
		}
		// Check symbol version.
		if info.versym != nil && version != 0 && int32(info.versym[symIndex]&0x7fff) != version {
			return false
		}

		*k.ptr = info.loadOffset + uintptr(sym.st_value)
		return true
	}

	if !info.isGNUHash {
		// Old-style DT_HASH table.
		for _, k := range vdsoSymbolKeys {
			if len(info.bucket) > 0 {
				for chain := info.bucket[k.symHash%uint32(len(info.bucket))]; chain != 0; chain = info.chain[chain] {
					if apply(chain, k) {
						break
					}
				}
			}
		}
		return
	}

	// New-style DT_GNU_HASH table.
	for _, k := range vdsoSymbolKeys {
		symIndex := info.bucket[k.gnuHash%uint32(len(info.bucket))]
		if symIndex < info.symOff {
			continue
		}
		for ; ; symIndex++ {
			hash := info.chain[symIndex-info.symOff]
			if hash|1 == k.gnuHash|1 {
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

func vdsoauxv(tag, val uintptr) {
	switch tag {
	case _AT_SYSINFO_EHDR:
		if val == 0 {
			// Something went wrong
			return
		}
		var info vdsoInfo
		// TODO(rsc): I don't understand why the compiler thinks info escapes
		// when passed to the three functions below.
		info1 := (*vdsoInfo)(noescape(unsafe.Pointer(&info)))
		vdsoInitFromSysinfoEhdr(info1, (*elfEhdr)(unsafe.Pointer(val)))
		vdsoParseSymbols(info1, vdsoFindVersion(info1, &vdsoLinuxVersion))
	}
}

// vdsoMarker reports whether PC is on the VDSO page.
//
//go:nosplit
func inVDSOPage(pc uintptr) bool {
	for _, k := range vdsoSymbolKeys {
		if *k.ptr != 0 {
			page := *k.ptr &^ (physPageSize - 1)
			return pc >= page && pc < page+physPageSize
		}
	}
	return false
}
