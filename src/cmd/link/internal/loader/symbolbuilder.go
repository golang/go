// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"cmp"
	"slices"
)

// SymbolBuilder is a helper designed to help with the construction
// of new symbol contents.
type SymbolBuilder struct {
	*extSymPayload         // points to payload being updated
	symIdx         Sym     // index of symbol being updated/constructed
	l              *Loader // loader
}

// MakeSymbolBuilder creates a symbol builder for use in constructing
// an entirely new symbol.
func (l *Loader) MakeSymbolBuilder(name string) *SymbolBuilder {
	// for now assume that any new sym is intended to be static
	symIdx := l.CreateStaticSym(name)
	sb := &SymbolBuilder{l: l, symIdx: symIdx}
	sb.extSymPayload = l.getPayload(symIdx)
	return sb
}

// MakeSymbolUpdater creates a symbol builder helper for an existing
// symbol 'symIdx'. If 'symIdx' is not an external symbol, then create
// a clone of it (copy name, properties, etc) fix things up so that
// the lookup tables and caches point to the new version, not the old
// version.
func (l *Loader) MakeSymbolUpdater(symIdx Sym) *SymbolBuilder {
	if symIdx == 0 {
		panic("can't update the null symbol")
	}
	if !l.IsExternal(symIdx) {
		// Create a clone with the same name/version/kind etc.
		l.cloneToExternal(symIdx)
	}

	// Construct updater and return.
	sb := &SymbolBuilder{l: l, symIdx: symIdx}
	sb.extSymPayload = l.getPayload(symIdx)
	return sb
}

// CreateSymForUpdate creates a symbol with given name and version,
// returns a CreateSymForUpdate for update. If the symbol already
// exists, it will update in-place.
func (l *Loader) CreateSymForUpdate(name string, version int) *SymbolBuilder {
	s := l.LookupOrCreateSym(name, version)
	l.SetAttrReachable(s, true)
	return l.MakeSymbolUpdater(s)
}

// Getters for properties of the symbol we're working on.

func (sb *SymbolBuilder) Sym() Sym               { return sb.symIdx }
func (sb *SymbolBuilder) Name() string           { return sb.name }
func (sb *SymbolBuilder) Version() int           { return sb.ver }
func (sb *SymbolBuilder) Type() sym.SymKind      { return sb.kind }
func (sb *SymbolBuilder) Size() int64            { return sb.size }
func (sb *SymbolBuilder) Data() []byte           { return sb.data }
func (sb *SymbolBuilder) Value() int64           { return sb.l.SymValue(sb.symIdx) }
func (sb *SymbolBuilder) Align() int32           { return sb.l.SymAlign(sb.symIdx) }
func (sb *SymbolBuilder) Localentry() uint8      { return sb.l.SymLocalentry(sb.symIdx) }
func (sb *SymbolBuilder) OnList() bool           { return sb.l.AttrOnList(sb.symIdx) }
func (sb *SymbolBuilder) External() bool         { return sb.l.AttrExternal(sb.symIdx) }
func (sb *SymbolBuilder) Extname() string        { return sb.l.SymExtname(sb.symIdx) }
func (sb *SymbolBuilder) CgoExportDynamic() bool { return sb.l.AttrCgoExportDynamic(sb.symIdx) }
func (sb *SymbolBuilder) Dynimplib() string      { return sb.l.SymDynimplib(sb.symIdx) }
func (sb *SymbolBuilder) Dynimpvers() string     { return sb.l.SymDynimpvers(sb.symIdx) }
func (sb *SymbolBuilder) SubSym() Sym            { return sb.l.SubSym(sb.symIdx) }
func (sb *SymbolBuilder) GoType() Sym            { return sb.l.SymGoType(sb.symIdx) }
func (sb *SymbolBuilder) VisibilityHidden() bool { return sb.l.AttrVisibilityHidden(sb.symIdx) }
func (sb *SymbolBuilder) Sect() *sym.Section     { return sb.l.SymSect(sb.symIdx) }

// Setters for symbol properties.

func (sb *SymbolBuilder) SetType(kind sym.SymKind)   { sb.kind = kind }
func (sb *SymbolBuilder) SetSize(size int64)         { sb.size = size }
func (sb *SymbolBuilder) SetData(data []byte)        { sb.data = data }
func (sb *SymbolBuilder) SetOnList(v bool)           { sb.l.SetAttrOnList(sb.symIdx, v) }
func (sb *SymbolBuilder) SetExternal(v bool)         { sb.l.SetAttrExternal(sb.symIdx, v) }
func (sb *SymbolBuilder) SetValue(v int64)           { sb.l.SetSymValue(sb.symIdx, v) }
func (sb *SymbolBuilder) SetAlign(align int32)       { sb.l.SetSymAlign(sb.symIdx, align) }
func (sb *SymbolBuilder) SetLocalentry(value uint8)  { sb.l.SetSymLocalentry(sb.symIdx, value) }
func (sb *SymbolBuilder) SetExtname(value string)    { sb.l.SetSymExtname(sb.symIdx, value) }
func (sb *SymbolBuilder) SetDynimplib(value string)  { sb.l.SetSymDynimplib(sb.symIdx, value) }
func (sb *SymbolBuilder) SetDynimpvers(value string) { sb.l.SetSymDynimpvers(sb.symIdx, value) }
func (sb *SymbolBuilder) SetPlt(value int32)         { sb.l.SetPlt(sb.symIdx, value) }
func (sb *SymbolBuilder) SetGot(value int32)         { sb.l.SetGot(sb.symIdx, value) }
func (sb *SymbolBuilder) SetSpecial(value bool)      { sb.l.SetAttrSpecial(sb.symIdx, value) }
func (sb *SymbolBuilder) SetLocal(value bool)        { sb.l.SetAttrLocal(sb.symIdx, value) }
func (sb *SymbolBuilder) SetVisibilityHidden(value bool) {
	sb.l.SetAttrVisibilityHidden(sb.symIdx, value)
}
func (sb *SymbolBuilder) SetNotInSymbolTable(value bool) {
	sb.l.SetAttrNotInSymbolTable(sb.symIdx, value)
}
func (sb *SymbolBuilder) SetSect(sect *sym.Section) { sb.l.SetSymSect(sb.symIdx, sect) }

func (sb *SymbolBuilder) AddBytes(data []byte) {
	if sb.kind == 0 {
		sb.kind = sym.SDATA
	}
	sb.data = append(sb.data, data...)
	sb.size = int64(len(sb.data))
}

func (sb *SymbolBuilder) Relocs() Relocs {
	return sb.l.Relocs(sb.symIdx)
}

// ResetRelocs removes all relocations on this symbol.
func (sb *SymbolBuilder) ResetRelocs() {
	sb.relocs = sb.relocs[:0]
}

// SetRelocType sets the type of the 'i'-th relocation on this sym to 't'
func (sb *SymbolBuilder) SetRelocType(i int, t objabi.RelocType) {
	sb.relocs[i].SetType(uint16(t))
}

// SetRelocSym sets the target sym of the 'i'-th relocation on this sym to 's'
func (sb *SymbolBuilder) SetRelocSym(i int, tgt Sym) {
	sb.relocs[i].SetSym(goobj.SymRef{PkgIdx: 0, SymIdx: uint32(tgt)})
}

// SetRelocAdd sets the addend of the 'i'-th relocation on this sym to 'a'
func (sb *SymbolBuilder) SetRelocAdd(i int, a int64) {
	sb.relocs[i].SetAdd(a)
}

// Add n relocations, return a handle to the relocations.
func (sb *SymbolBuilder) AddRelocs(n int) Relocs {
	sb.relocs = append(sb.relocs, make([]goobj.Reloc, n)...)
	return sb.l.Relocs(sb.symIdx)
}

// Add a relocation with given type, return its handle and index
// (to set other fields).
func (sb *SymbolBuilder) AddRel(typ objabi.RelocType) (Reloc, int) {
	j := len(sb.relocs)
	sb.relocs = append(sb.relocs, goobj.Reloc{})
	sb.relocs[j].SetType(uint16(typ))
	relocs := sb.Relocs()
	return relocs.At(j), j
}

// SortRelocs Sort relocations by offset.
func (sb *SymbolBuilder) SortRelocs() {
	slices.SortFunc(sb.extSymPayload.relocs, func(a, b goobj.Reloc) int {
		return cmp.Compare(a.Off(), b.Off())
	})
}

func (sb *SymbolBuilder) Reachable() bool {
	return sb.l.AttrReachable(sb.symIdx)
}

func (sb *SymbolBuilder) SetReachable(v bool) {
	sb.l.SetAttrReachable(sb.symIdx, v)
}

func (sb *SymbolBuilder) ReadOnly() bool {
	return sb.l.AttrReadOnly(sb.symIdx)
}

func (sb *SymbolBuilder) SetReadOnly(v bool) {
	sb.l.SetAttrReadOnly(sb.symIdx, v)
}

func (sb *SymbolBuilder) DuplicateOK() bool {
	return sb.l.AttrDuplicateOK(sb.symIdx)
}

func (sb *SymbolBuilder) SetDuplicateOK(v bool) {
	sb.l.SetAttrDuplicateOK(sb.symIdx, v)
}

func (sb *SymbolBuilder) Outer() Sym {
	return sb.l.OuterSym(sb.symIdx)
}

func (sb *SymbolBuilder) Sub() Sym {
	return sb.l.SubSym(sb.symIdx)
}

func (sb *SymbolBuilder) SortSub() {
	sb.l.SortSub(sb.symIdx)
}

func (sb *SymbolBuilder) AddInteriorSym(sub Sym) {
	sb.l.AddInteriorSym(sb.symIdx, sub)
}

func (sb *SymbolBuilder) AddUint8(v uint8) int64 {
	off := sb.size
	if sb.kind == 0 {
		sb.kind = sym.SDATA
	}
	sb.size++
	sb.data = append(sb.data, v)
	return off
}

func (sb *SymbolBuilder) AddUintXX(arch *sys.Arch, v uint64, wid int) int64 {
	off := sb.size
	sb.setUintXX(arch, off, v, int64(wid))
	return off
}

func (sb *SymbolBuilder) setUintXX(arch *sys.Arch, off int64, v uint64, wid int64) int64 {
	if sb.kind == 0 {
		sb.kind = sym.SDATA
	}
	if sb.size < off+wid {
		sb.size = off + wid
		sb.Grow(sb.size)
	}

	switch wid {
	case 1:
		sb.data[off] = uint8(v)
	case 2:
		arch.ByteOrder.PutUint16(sb.data[off:], uint16(v))
	case 4:
		arch.ByteOrder.PutUint32(sb.data[off:], uint32(v))
	case 8:
		arch.ByteOrder.PutUint64(sb.data[off:], v)
	}

	return off + wid
}

func (sb *SymbolBuilder) AddUint16(arch *sys.Arch, v uint16) int64 {
	return sb.AddUintXX(arch, uint64(v), 2)
}

func (sb *SymbolBuilder) AddUint32(arch *sys.Arch, v uint32) int64 {
	return sb.AddUintXX(arch, uint64(v), 4)
}

func (sb *SymbolBuilder) AddUint64(arch *sys.Arch, v uint64) int64 {
	return sb.AddUintXX(arch, v, 8)
}

func (sb *SymbolBuilder) AddUint(arch *sys.Arch, v uint64) int64 {
	return sb.AddUintXX(arch, v, arch.PtrSize)
}

func (sb *SymbolBuilder) SetUint8(arch *sys.Arch, r int64, v uint8) int64 {
	return sb.setUintXX(arch, r, uint64(v), 1)
}

func (sb *SymbolBuilder) SetUint16(arch *sys.Arch, r int64, v uint16) int64 {
	return sb.setUintXX(arch, r, uint64(v), 2)
}

func (sb *SymbolBuilder) SetUint32(arch *sys.Arch, r int64, v uint32) int64 {
	return sb.setUintXX(arch, r, uint64(v), 4)
}

func (sb *SymbolBuilder) SetUint(arch *sys.Arch, r int64, v uint64) int64 {
	return sb.setUintXX(arch, r, v, int64(arch.PtrSize))
}

func (sb *SymbolBuilder) SetUintptr(arch *sys.Arch, r int64, v uintptr) int64 {
	return sb.setUintXX(arch, r, uint64(v), int64(arch.PtrSize))
}

func (sb *SymbolBuilder) SetAddrPlus(arch *sys.Arch, off int64, tgt Sym, add int64) int64 {
	if sb.Type() == 0 {
		sb.SetType(sym.SDATA)
	}
	if off+int64(arch.PtrSize) > sb.size {
		sb.size = off + int64(arch.PtrSize)
		sb.Grow(sb.size)
	}
	r, _ := sb.AddRel(objabi.R_ADDR)
	r.SetSym(tgt)
	r.SetOff(int32(off))
	r.SetSiz(uint8(arch.PtrSize))
	r.SetAdd(add)
	return off + int64(r.Siz())
}

func (sb *SymbolBuilder) SetAddr(arch *sys.Arch, off int64, tgt Sym) int64 {
	return sb.SetAddrPlus(arch, off, tgt, 0)
}

func (sb *SymbolBuilder) AddStringAt(off int64, str string) int64 {
	strLen := int64(len(str))
	if off+strLen > int64(len(sb.data)) {
		panic("attempt to write past end of buffer")
	}
	copy(sb.data[off:off+strLen], str)
	return off + strLen
}

// AddCStringAt adds str plus a null terminating byte.
func (sb *SymbolBuilder) AddCStringAt(off int64, str string) int64 {
	strLen := int64(len(str))
	if off+strLen+1 > int64(len(sb.data)) {
		panic("attempt to write past end of buffer")
	}
	copy(sb.data[off:off+strLen], str)
	sb.data[off+strLen] = 0
	return off + strLen + 1
}

func (sb *SymbolBuilder) Addstring(str string) int64 {
	if sb.kind == 0 {
		sb.kind = sym.SNOPTRDATA
	}
	r := sb.size
	sb.data = append(sb.data, str...)
	sb.data = append(sb.data, 0)
	sb.size = int64(len(sb.data))
	return r
}

func (sb *SymbolBuilder) SetBytesAt(off int64, b []byte) int64 {
	datLen := int64(len(b))
	if off+datLen > int64(len(sb.data)) {
		panic("attempt to write past end of buffer")
	}
	copy(sb.data[off:off+datLen], b)
	return off + datLen
}

func (sb *SymbolBuilder) addSymRef(tgt Sym, add int64, typ objabi.RelocType, rsize int) int64 {
	if sb.kind == 0 {
		sb.kind = sym.SDATA
	}
	i := sb.size

	sb.size += int64(rsize)
	sb.Grow(sb.size)

	r, _ := sb.AddRel(typ)
	r.SetSym(tgt)
	r.SetOff(int32(i))
	r.SetSiz(uint8(rsize))
	r.SetAdd(add)

	return i + int64(rsize)
}

// Add a symbol reference (relocation) with given type, addend, and size
// (the most generic form).
func (sb *SymbolBuilder) AddSymRef(arch *sys.Arch, tgt Sym, add int64, typ objabi.RelocType, rsize int) int64 {
	return sb.addSymRef(tgt, add, typ, rsize)
}

func (sb *SymbolBuilder) AddAddrPlus(arch *sys.Arch, tgt Sym, add int64) int64 {
	return sb.addSymRef(tgt, add, objabi.R_ADDR, arch.PtrSize)
}

func (sb *SymbolBuilder) AddAddrPlus4(arch *sys.Arch, tgt Sym, add int64) int64 {
	return sb.addSymRef(tgt, add, objabi.R_ADDR, 4)
}

func (sb *SymbolBuilder) AddAddr(arch *sys.Arch, tgt Sym) int64 {
	return sb.AddAddrPlus(arch, tgt, 0)
}

func (sb *SymbolBuilder) AddPEImageRelativeAddrPlus(arch *sys.Arch, tgt Sym, add int64) int64 {
	return sb.addSymRef(tgt, add, objabi.R_PEIMAGEOFF, 4)
}

func (sb *SymbolBuilder) AddPCRelPlus(arch *sys.Arch, tgt Sym, add int64) int64 {
	return sb.addSymRef(tgt, add, objabi.R_PCREL, 4)
}

func (sb *SymbolBuilder) AddCURelativeAddrPlus(arch *sys.Arch, tgt Sym, add int64) int64 {
	return sb.addSymRef(tgt, add, objabi.R_ADDRCUOFF, arch.PtrSize)
}

func (sb *SymbolBuilder) AddSize(arch *sys.Arch, tgt Sym) int64 {
	return sb.addSymRef(tgt, 0, objabi.R_SIZE, arch.PtrSize)
}

// GenAddAddrPlusFunc returns a function to be called when capturing
// a function symbol's address. In later stages of the link (when
// address assignment is done) when doing internal linking and
// targeting an executable, we can just emit the address of a function
// directly instead of generating a relocation. Clients can call
// this function (setting 'internalExec' based on build mode and target)
// and then invoke the returned function in roughly the same way that
// loader.*SymbolBuilder.AddAddrPlus would be used.
func GenAddAddrPlusFunc(internalExec bool) func(s *SymbolBuilder, arch *sys.Arch, tgt Sym, add int64) int64 {
	if internalExec {
		return func(s *SymbolBuilder, arch *sys.Arch, tgt Sym, add int64) int64 {
			if v := s.l.SymValue(tgt); v != 0 {
				return s.AddUint(arch, uint64(v+add))
			}
			return s.AddAddrPlus(arch, tgt, add)
		}
	} else {
		return (*SymbolBuilder).AddAddrPlus
	}
}

func (sb *SymbolBuilder) MakeWritable() {
	if sb.ReadOnly() {
		sb.data = append([]byte(nil), sb.data...)
		sb.l.SetAttrReadOnly(sb.symIdx, false)
	}
}

func (sb *SymbolBuilder) AddUleb(v uint64) {
	if v < 128 { // common case: 1 byte
		sb.AddUint8(uint8(v))
		return
	}
	for {
		c := uint8(v & 0x7f)
		v >>= 7
		if v != 0 {
			c |= 0x80
		}
		sb.AddUint8(c)
		if c&0x80 == 0 {
			break
		}
	}
}
