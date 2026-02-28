// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (amd64 || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x)

package runtime

// ELF64 structure definitions for use by the vDSO loader

type elfSym struct {
	st_name  uint32
	st_info  byte
	st_other byte
	st_shndx uint16
	st_value uint64
	st_size  uint64
}

type elfVerdef struct {
	vd_version uint16 /* Version revision */
	vd_flags   uint16 /* Version information */
	vd_ndx     uint16 /* Version Index */
	vd_cnt     uint16 /* Number of associated aux entries */
	vd_hash    uint32 /* Version name hash value */
	vd_aux     uint32 /* Offset in bytes to verdaux array */
	vd_next    uint32 /* Offset in bytes to next verdef entry */
}

type elfEhdr struct {
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

type elfPhdr struct {
	p_type   uint32 /* Segment type */
	p_flags  uint32 /* Segment flags */
	p_offset uint64 /* Segment file offset */
	p_vaddr  uint64 /* Segment virtual address */
	p_paddr  uint64 /* Segment physical address */
	p_filesz uint64 /* Segment size in file */
	p_memsz  uint64 /* Segment size in memory */
	p_align  uint64 /* Segment alignment */
}

type elfShdr struct {
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

type elfDyn struct {
	d_tag int64  /* Dynamic entry type */
	d_val uint64 /* Integer value */
}

type elfVerdaux struct {
	vda_name uint32 /* Version or dependency names */
	vda_next uint32 /* Offset in bytes to next verdaux entry */
}
