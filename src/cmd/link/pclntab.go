// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generation of runtime function information (pclntab).

package main

import (
	"debug/goobj"
	"encoding/binary"
	"os"
	"sort"
)

var zerofunc goobj.Func

// pclntab collects the runtime function data for each function that will
// be listed in the binary and builds a single table describing all functions.
// This table is used at run time for stack traces and to look up PC-specific
// information during garbage collection. The symbol created is named
// "pclntab" for historical reasons; the scope of the table has grown to
// include more than just PC/line number correspondences.
// The table format is documented at http://golang.org/s/go12symtab.
func (p *Prog) pclntab() {
	// Count number of functions going into the binary,
	// so that we can size the initial index correctly.
	nfunc := 0
	for _, sym := range p.SymOrder {
		if sym.Kind != goobj.STEXT {
			continue
		}
		nfunc++
	}

	// Table header.
	buf := new(SymBuffer)
	buf.Init(p)
	buf.SetSize(8 + p.ptrsize)
	off := 0
	off = buf.Uint32(off, 0xfffffffb)
	off = buf.Uint8(off, 0)
	off = buf.Uint8(off, 0)
	off = buf.Uint8(off, uint8(p.pcquantum))
	off = buf.Uint8(off, uint8(p.ptrsize))
	off = buf.Uint(off, uint64(nfunc), p.ptrsize)
	indexOff := off
	off += (nfunc*2 + 1) * p.ptrsize // function index, to be filled in
	off += 4                         // file table start offset, to be filled in
	buf.SetSize(off)

	// One-file cache for reading PCData tables from package files.
	// TODO(rsc): Better I/O strategy.
	var (
		file  *os.File
		fname string
	)

	// Files gives the file numbering for source file names recorded
	// in the binary.
	files := make(map[string]int)

	// Build the table, build the index, and build the file name numbering.
	// The loop here must visit functions in the same order that they will
	// be stored in the binary, or else binary search over the index will fail.
	// The runtime checks that the index is sorted properly at program start time.
	var lastSym *Sym
	for _, sym := range p.SymOrder {
		if sym.Kind != goobj.STEXT {
			continue
		}
		lastSym = sym

		// Treat no recorded function information same as all zeros.
		f := sym.Func
		if f == nil {
			f = &zerofunc
		}

		// Open package file if needed, for reading PC data.
		if fname != sym.Package.File {
			if file != nil {
				file.Close()
			}
			var err error
			file, err = os.Open(sym.Package.File)
			if err != nil {
				p.errorf("%v: %v", sym, err)
				return
			}
			fname = sym.Package.File
		}

		// off is the offset of the table entry where we're going to write
		// the encoded form of Func.
		// indexOff is the current position in the table index;
		// we add an entry in the index pointing at off.
		off = (buf.Size() + p.ptrsize - 1) &^ (p.ptrsize - 1)
		indexOff = buf.Addr(indexOff, sym.SymID, 0)
		indexOff = buf.Uint(indexOff, uint64(off), p.ptrsize)

		// The Func encoding starts with a header giving offsets
		// to data blobs, and then the data blobs themselves.
		// end gives the current write position for the data blobs.
		end := off + p.ptrsize + 3*4 + 5*4 + len(f.PCData)*4 + len(f.FuncData)*p.ptrsize
		if len(f.FuncData) > 0 {
			end += -end & (p.ptrsize - 1)
		}
		buf.SetSize(end)

		// entry uintptr
		// name int32
		// args int32
		// frame int32
		//
		// The frame recorded in the object file is
		// the frame size used in an assembly listing, which does
		// not include the caller PC on the stack.
		// The frame size we want to list here is the delta from
		// this function's SP to its caller's SP, which does include
		// the caller PC. Add p.ptrsize to f.Frame to adjust.
		// TODO(rsc): Record the same frame size in the object file.
		off = buf.Addr(off, sym.SymID, 0)
		off = buf.Uint32(off, uint32(addString(buf, sym.Name)))
		off = buf.Uint32(off, uint32(f.Args))
		off = buf.Uint32(off, uint32(f.Frame+p.ptrsize))

		// pcdata
		off = buf.Uint32(off, uint32(addPCTable(p, buf, file, f.PCSP)))
		off = buf.Uint32(off, uint32(addPCFileTable(p, buf, file, f.PCFile, sym, files)))
		off = buf.Uint32(off, uint32(addPCTable(p, buf, file, f.PCLine)))
		off = buf.Uint32(off, uint32(len(f.PCData)))
		off = buf.Uint32(off, uint32(len(f.FuncData)))
		for _, pcdata := range f.PCData {
			off = buf.Uint32(off, uint32(addPCTable(p, buf, file, pcdata)))
		}

		// funcdata
		if len(f.FuncData) > 0 {
			off += -off & (p.ptrsize - 1) // must be pointer-aligned
			for _, funcdata := range f.FuncData {
				if funcdata.Sym.Name == "" {
					off = buf.Uint(off, uint64(funcdata.Offset), p.ptrsize)
				} else {
					off = buf.Addr(off, funcdata.Sym, funcdata.Offset)
				}
			}
		}

		if off != end {
			p.errorf("internal error: invalid math in pclntab: off=%#x end=%#x", off, end)
			break
		}
	}
	if file != nil {
		file.Close()
	}

	// Final entry of index is end PC of last function.
	indexOff = buf.Addr(indexOff, lastSym.SymID, int64(lastSym.Size))

	// Start file table.
	// Function index is immediately followed by offset to file table.
	off = (buf.Size() + p.ptrsize - 1) &^ (p.ptrsize - 1)
	buf.Uint32(indexOff, uint32(off))

	// File table is an array of uint32s.
	// The first entry gives 1+n, the size of the array.
	// The following n entries hold offsets to string data.
	// File number n uses the string pointed at by entry n.
	// File number 0 is invalid.
	buf.SetSize(off + (1+len(files))*4)
	buf.Uint32(off, uint32(1+len(files)))
	var filestr []string
	for file := range files {
		filestr = append(filestr, file)
	}
	sort.Strings(filestr)
	for _, file := range filestr {
		id := files[file]
		buf.Uint32(off+4*id, uint32(addString(buf, file)))
	}

	pclntab := &Sym{
		Sym: &goobj.Sym{
			SymID: goobj.SymID{Name: "runtime.pclntab"},
			Kind:  goobj.SPCLNTAB,
			Size:  buf.Size(),
			Reloc: buf.Reloc(),
		},
		Bytes: buf.Bytes(),
	}
	p.addSym(pclntab)
}

// addString appends the string s to the buffer b.
// It returns the offset of the beginning of the string in the buffer.
func addString(b *SymBuffer, s string) int {
	off := b.Size()
	b.SetSize(off + len(s) + 1)
	copy(b.data[off:], s)
	return off
}

// addPCTable appends the PC-data table stored in the file f at the location loc
// to the symbol buffer b. It returns the offset of the beginning of the table
// in the buffer.
func addPCTable(p *Prog, b *SymBuffer, f *os.File, loc goobj.Data) int {
	if loc.Size == 0 {
		return 0
	}
	off := b.Size()
	b.SetSize(off + int(loc.Size))
	_, err := f.ReadAt(b.data[off:off+int(loc.Size)], loc.Offset)
	if err != nil {
		p.errorf("%v", err)
	}
	return off
}

// addPCFileTable is like addPCTable, but it renumbers the file names referred to by the table
// to use the global numbering maintained in the files map. It adds new files to the
// map as necessary.
func addPCFileTable(p *Prog, b *SymBuffer, f *os.File, loc goobj.Data, sym *Sym, files map[string]int) int {
	if loc.Size == 0 {
		return 0
	}
	off := b.Size()

	src := make([]byte, loc.Size)
	_, err := f.ReadAt(src, loc.Offset)
	if err != nil {
		p.errorf("%v", err)
		return 0
	}

	filenum := make([]int, len(sym.Func.File))
	for i, name := range sym.Func.File {
		num := files[name]
		if num == 0 {
			num = len(files) + 1
			files[name] = num
		}
		filenum[i] = num
	}

	var dst []byte
	newval := int32(-1)
	var it PCIter
	for it.Init(p, src); !it.Done; it.Next() {
		// value delta
		oldval := it.Value
		val := oldval
		if oldval != -1 {
			if oldval < 0 || int(oldval) >= len(filenum) {
				p.errorf("%s: corrupt pc-file table", sym)
				break
			}
			val = int32(filenum[oldval])
		}
		dv := val - newval
		newval = val
		uv := uint32(dv<<1) ^ uint32(dv>>31)
		dst = appendVarint(dst, uv)

		// pc delta
		dst = appendVarint(dst, it.NextPC-it.PC)
	}
	if it.Corrupt {
		p.errorf("%s: corrupt pc-file table", sym)
	}

	// terminating value delta
	dst = appendVarint(dst, 0)

	b.SetSize(off + len(dst))
	copy(b.data[off:], dst)
	return off
}

// A SymBuffer is a buffer for preparing the data image of a
// linker-generated symbol.
type SymBuffer struct {
	data    []byte
	reloc   []goobj.Reloc
	order   binary.ByteOrder
	ptrsize int
}

// Init initializes the buffer for writing.
func (b *SymBuffer) Init(p *Prog) {
	b.data = nil
	b.reloc = nil
	b.order = p.byteorder
	b.ptrsize = p.ptrsize
}

// Bytes returns the buffer data.
func (b *SymBuffer) Bytes() []byte {
	return b.data
}

// SetSize sets the buffer's data size to n bytes.
func (b *SymBuffer) SetSize(n int) {
	for cap(b.data) < n {
		b.data = append(b.data[:cap(b.data)], 0)
	}
	b.data = b.data[:n]
}

// Size returns the buffer's data size.
func (b *SymBuffer) Size() int {
	return len(b.data)
}

// Reloc returns the buffered relocations.
func (b *SymBuffer) Reloc() []goobj.Reloc {
	return b.reloc
}

// Uint8 sets the uint8 at offset off to v.
// It returns the offset just beyond v.
func (b *SymBuffer) Uint8(off int, v uint8) int {
	b.data[off] = v
	return off + 1
}

// Uint16 sets the uint16 at offset off to v.
// It returns the offset just beyond v.
func (b *SymBuffer) Uint16(off int, v uint16) int {
	b.order.PutUint16(b.data[off:], v)
	return off + 2
}

// Uint32 sets the uint32 at offset off to v.
// It returns the offset just beyond v.
func (b *SymBuffer) Uint32(off int, v uint32) int {
	b.order.PutUint32(b.data[off:], v)
	return off + 4
}

// Uint64 sets the uint64 at offset off to v.
// It returns the offset just beyond v.
func (b *SymBuffer) Uint64(off int, v uint64) int {
	b.order.PutUint64(b.data[off:], v)
	return off + 8
}

// Uint sets the size-byte unsigned integer at offset off to v.
// It returns the offset just beyond v.
func (b *SymBuffer) Uint(off int, v uint64, size int) int {
	switch size {
	case 1:
		return b.Uint8(off, uint8(v))
	case 2:
		return b.Uint16(off, uint16(v))
	case 4:
		return b.Uint32(off, uint32(v))
	case 8:
		return b.Uint64(off, v)
	}
	panic("invalid use of SymBuffer.SetUint")
}

// Addr sets the pointer-sized address at offset off to refer
// to symoff bytes past the start of sym. It returns the offset
// just beyond the address.
func (b *SymBuffer) Addr(off int, sym goobj.SymID, symoff int64) int {
	b.reloc = append(b.reloc, goobj.Reloc{
		Offset: off,
		Size:   b.ptrsize,
		Sym:    sym,
		Add:    int(symoff),
		Type:   R_ADDR,
	})
	return off + b.ptrsize
}

// A PCIter implements iteration over PC-data tables.
//
//	var it PCIter
//	for it.Init(p, data); !it.Done; it.Next() {
//		it.Value holds from it.PC up to (but not including) it.NextPC
//	}
//	if it.Corrupt {
//		data was malformed
//	}
//
type PCIter struct {
	PC        uint32
	NextPC    uint32
	Value     int32
	Done      bool
	Corrupt   bool
	p         []byte
	start     bool
	pcquantum uint32
}

// Init initializes the iteration.
// On return, if it.Done is true, the iteration is over.
// Otherwise it.Value applies in the pc range [it.PC, it.NextPC).
func (it *PCIter) Init(p *Prog, buf []byte) {
	it.p = buf
	it.PC = 0
	it.NextPC = 0
	it.Value = -1
	it.start = true
	it.pcquantum = uint32(p.pcquantum)
	it.Done = false
	it.Next()
}

// Next steps forward one entry in the table.
// On return, if it.Done is true, the iteration is over.
// Otherwise it.Value applies in the pc range [it.PC, it.NextPC).
func (it *PCIter) Next() {
	it.PC = it.NextPC
	if it.Done {
		return
	}
	if len(it.p) == 0 {
		it.Done = true
		return
	}

	// value delta
	uv, p, ok := decodeVarint(it.p)
	if !ok {
		it.Done = true
		it.Corrupt = true
		return
	}
	it.p = p
	if uv == 0 && !it.start {
		it.Done = true
		return
	}
	it.start = false
	sv := int32(uv>>1) ^ int32(uv<<31)>>31
	it.Value += sv

	// pc delta
	uv, it.p, ok = decodeVarint(it.p)
	if !ok {
		it.Done = true
		it.Corrupt = true
		return
	}
	it.NextPC = it.PC + uv*it.pcquantum
}

// decodeVarint decodes an unsigned varint from p,
// reporting the value, the remainder of the data, and
// whether the decoding was successful.
func decodeVarint(p []byte) (v uint32, rest []byte, ok bool) {
	for shift := uint(0); ; shift += 7 {
		if len(p) == 0 {
			return
		}
		c := uint32(p[0])
		p = p[1:]
		v |= (c & 0x7F) << shift
		if c&0x80 == 0 {
			break
		}
	}
	return v, p, true
}

// appendVarint appends an unsigned varint encoding of v to p
// and returns the resulting slice.
func appendVarint(p []byte, v uint32) []byte {
	for ; v >= 0x80; v >>= 7 {
		p = append(p, byte(v)|0x80)
	}
	p = append(p, byte(v))
	return p
}
