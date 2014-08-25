// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"debug/goobj"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"
)

// Test of pcln table encoding.
// testdata/genpcln.go generates an assembly file with
// pseudorandom values for the data that pclntab stores.
// This test recomputes the same pseudorandom stream
// and checks that the final linked binary uses those values
// as well.
func TestPclntab(t *testing.T) {
	p := &Prog{
		GOOS:        "darwin",
		GOARCH:      "amd64",
		Error:       func(s string) { t.Error(s) },
		StartSym:    "start",
		omitRuntime: true,
	}
	var buf bytes.Buffer
	p.link(&buf, "testdata/pclntab.6")
	if p.NumError > 0 {
		return
	}

	// The algorithm for computing values here must match
	// the one in testdata/genpcln.go.
	for f := 0; f < 3; f++ {
		file := "input"
		line := 1
		rnd := rand.New(rand.NewSource(int64(f)))
		args := rnd.Intn(100) * 8
		frame := 32 + rnd.Intn(32)/8*8
		size := 200 + rnd.Intn(100)*8

		name := fmt.Sprintf("func%d", f)
		r, off, fargs, fframe, ok := findFunc(t, p, name)
		if !ok {
			continue // error already printed
		}
		if fargs != args {
			t.Errorf("%s: args=%d, want %d", name, fargs, args)
		}
		if fframe != frame+8 {
			t.Errorf("%s: frame=%d, want %d", name, fframe, frame+8)
		}

		// Check FUNCDATA 1.
		fdata, ok := loadFuncdata(t, r, name, off, 1)
		if ok {
			fsym := p.Syms[goobj.SymID{Name: fmt.Sprintf("funcdata%d", f)}]
			if fsym == nil {
				t.Errorf("funcdata%d is missing in binary", f)
			} else if fdata != fsym.Addr {
				t.Errorf("%s: funcdata 1 = %#x, want %#x", name, fdata, fsym.Addr)
			}
		}

		// Walk code checking pcdata values.
		spadj := 0
		pcdata1 := -1
		pcdata2 := -1

		checkPCSP(t, r, name, off, 0, 0)
		checkPCData(t, r, name, off, 0, 0, -1)
		checkPCData(t, r, name, off, 0, 1, -1)
		checkPCData(t, r, name, off, 0, 2, -1)

		firstpc := 4
		for i := 0; i < size; i++ {
			pc := firstpc + i // skip SP adjustment to allocate frame
			if i >= 0x100 && t.Failed() {
				break
			}
			// Possible SP adjustment.
			checkPCSP(t, r, name, off, pc, frame+spadj)
			if rnd.Intn(100) == 0 {
				checkPCFileLine(t, r, name, off, pc, file, line)
				checkPCData(t, r, name, off, pc, 1, pcdata1)
				checkPCData(t, r, name, off, pc, 2, pcdata2)
				i += 1
				pc = firstpc + i
				checkPCFileLine(t, r, name, off, pc-1, file, line)
				checkPCData(t, r, name, off, pc-1, 1, pcdata1)
				checkPCData(t, r, name, off, pc-1, 2, pcdata2)
				checkPCSP(t, r, name, off, pc-1, frame+spadj)

				if spadj <= -32 || spadj < 32 && rnd.Intn(2) == 0 {
					spadj += 8
				} else {
					spadj -= 8
				}
				checkPCSP(t, r, name, off, pc, frame+spadj)
			}

			// Possible PCFile change.
			if rnd.Intn(100) == 0 {
				file = fmt.Sprintf("file%d.s", rnd.Intn(10))
				line = rnd.Intn(100) + 1
			}

			// Possible PCLine change.
			if rnd.Intn(10) == 0 {
				line = rnd.Intn(1000) + 1
			}

			// Possible PCData $1 change.
			if rnd.Intn(100) == 0 {
				pcdata1 = rnd.Intn(1000)
			}

			// Possible PCData $2 change.
			if rnd.Intn(100) == 0 {
				pcdata2 = rnd.Intn(1000)
			}

			if i == 0 {
				checkPCFileLine(t, r, name, off, 0, file, line)
				checkPCFileLine(t, r, name, off, pc-1, file, line)
			}
			checkPCFileLine(t, r, name, off, pc, file, line)
			checkPCData(t, r, name, off, pc, 1, pcdata1)
			checkPCData(t, r, name, off, pc, 2, pcdata2)
		}
	}
}

// findFunc finds the function information in the pclntab of p
// for the function with the given name.
// It returns a symbol reader for pclntab, the offset of the function information
// within that symbol, and the args and frame values read out of the information.
func findFunc(t *testing.T, p *Prog, name string) (r *SymReader, off, args, frame int, ok bool) {
	tabsym := p.Syms[goobj.SymID{Name: "pclntab"}]
	if tabsym == nil {
		t.Errorf("pclntab is missing in binary")
		return
	}

	r = new(SymReader)
	r.Init(p, tabsym)

	// pclntab must with 8-byte header
	if r.Uint32(0) != 0xfffffffb || r.Uint8(4) != 0 || r.Uint8(5) != 0 || r.Uint8(6) != uint8(p.pcquantum) || r.Uint8(7) != uint8(p.ptrsize) {
		t.Errorf("pclntab has incorrect header %.8x", r.data[:8])
		return
	}

	sym := p.Syms[goobj.SymID{Name: name}]
	if sym == nil {
		t.Errorf("%s is missing in the binary", name)
		return
	}

	// index is nfunc addr0 off0 addr1 off1 ... addr_nfunc (sentinel)
	nfunc := int(r.Addr(8))
	i := sort.Search(nfunc, func(i int) bool {
		return r.Addr(8+p.ptrsize*(1+2*i)) >= sym.Addr
	})
	if entry := r.Addr(8 + p.ptrsize*(1+2*i)); entry != sym.Addr {
		indexTab := make([]Addr, 2*nfunc+1)
		for j := range indexTab {
			indexTab[j] = r.Addr(8 + p.ptrsize*(1+j))
		}
		t.Errorf("pclntab is missing entry for %s (%#x): %#x", name, sym.Addr, indexTab)
		return
	}

	off = int(r.Addr(8 + p.ptrsize*(1+2*i+1)))

	// func description at off is
	//	entry addr
	//	nameoff uint32
	//	args uint32
	//	frame uint32
	//	pcspoff uint32
	//	pcfileoff uint32
	//	pclineoff uint32
	//	npcdata uint32
	//	nfuncdata uint32
	//	pcdata npcdata*uint32
	//	funcdata nfuncdata*addr
	//
	if entry := r.Addr(off); entry != sym.Addr {
		t.Errorf("pclntab inconsistent: entry for %s addr=%#x has entry=%#x", name, sym.Addr, entry)
		return
	}
	nameoff := int(r.Uint32(off + p.ptrsize))
	args = int(r.Uint32(off + p.ptrsize + 1*4))
	frame = int(r.Uint32(off + p.ptrsize + 2*4))

	fname := r.String(nameoff)
	if fname != name {
		t.Errorf("pclntab inconsistent: entry for %s addr=%#x has name %q", name, sym.Addr, fname)
	}

	ok = true // off, args, frame are usable
	return
}

// loadFuncdata returns the funcdata #fnum value
// loaded from the function information for name.
func loadFuncdata(t *testing.T, r *SymReader, name string, off int, fnum int) (Addr, bool) {
	npcdata := int(r.Uint32(off + r.p.ptrsize + 6*4))
	nfuncdata := int(r.Uint32(off + r.p.ptrsize + 7*4))
	if fnum >= nfuncdata {
		t.Errorf("pclntab(%s): no funcdata %d (only < %d)", name, fnum, nfuncdata)
		return 0, false
	}
	fdataoff := off + r.p.ptrsize + (8+npcdata)*4 + fnum*r.p.ptrsize
	fdataoff += fdataoff & 4
	return r.Addr(fdataoff), true
}

// checkPCSP checks that the PCSP table in the function information at off
// lists spadj as the sp delta for pc.
func checkPCSP(t *testing.T, r *SymReader, name string, off, pc, spadj int) {
	pcoff := r.Uint32(off + r.p.ptrsize + 3*4)
	pcval, ok := readPCData(t, r, name, "PCSP", pcoff, pc)
	if !ok {
		return
	}
	if pcval != spadj {
		t.Errorf("pclntab(%s): at pc=+%#x, pcsp=%d, want %d", name, pc, pcval, spadj)
	}
}

// checkPCSP checks that the PCFile and PCLine tables in the function information at off
// list file, line as the file name and line number for pc.
func checkPCFileLine(t *testing.T, r *SymReader, name string, off, pc int, file string, line int) {
	pcfileoff := r.Uint32(off + r.p.ptrsize + 4*4)
	pclineoff := r.Uint32(off + r.p.ptrsize + 5*4)
	pcfilenum, ok1 := readPCData(t, r, name, "PCFile", pcfileoff, pc)
	pcline, ok2 := readPCData(t, r, name, "PCLine", pclineoff, pc)
	if !ok1 || !ok2 {
		return
	}
	nfunc := int(r.Addr(8))
	filetaboff := r.Uint32(8 + r.p.ptrsize*2*(nfunc+1))
	nfile := int(r.Uint32(int(filetaboff)))
	if pcfilenum <= 0 || pcfilenum >= nfile {
		t.Errorf("pclntab(%s): at pc=+%#x, filenum=%d (invalid; nfile=%d)", name, pc, pcfilenum, nfile)
	}
	pcfile := r.String(int(r.Uint32(int(filetaboff) + pcfilenum*4)))
	if !strings.HasSuffix(pcfile, file) {
		t.Errorf("pclntab(%s): at pc=+%#x, file=%q, want %q", name, pc, pcfile, file)
	}
	if pcline != line {
		t.Errorf("pclntab(%s): at pc=+%#x, line=%d, want %d", name, pc, pcline, line)
	}
}

// checkPCData checks that the PCData#pnum table in the function information at off
// list val as the value for pc.
func checkPCData(t *testing.T, r *SymReader, name string, off, pc, pnum, val int) {
	pcoff := r.Uint32(off + r.p.ptrsize + (8+pnum)*4)
	pcval, ok := readPCData(t, r, name, fmt.Sprintf("PCData#%d", pnum), pcoff, pc)
	if !ok {
		return
	}
	if pcval != val {
		t.Errorf("pclntab(%s): at pc=+%#x, pcdata#%d=%d, want %d", name, pc, pnum, pcval, val)
	}
}

// readPCData reads the PCData table offset off
// to obtain and return the value associated with pc.
func readPCData(t *testing.T, r *SymReader, name, pcdataname string, pcoff uint32, pc int) (int, bool) {
	// "If pcsp, pcfile, pcln, or any of the pcdata offsets is zero,
	// that table is considered missing, and all PCs take value -1."
	if pcoff == 0 {
		return -1, true
	}

	var it PCIter
	for it.Init(r.p, r.data[pcoff:]); !it.Done; it.Next() {
		if it.PC <= uint32(pc) && uint32(pc) < it.NextPC {
			return int(it.Value), true
		}
	}
	if it.Corrupt {
		t.Errorf("pclntab(%s): %s: corrupt pcdata table", name, pcdataname)
	}
	return 0, false
}

// A SymReader provides typed access to the data for a symbol.
type SymReader struct {
	p    *Prog
	data []byte
}

func (r *SymReader) Init(p *Prog, sym *Sym) {
	seg := sym.Section.Segment
	off := sym.Addr - seg.VirtAddr
	data := seg.Data[off : off+Addr(sym.Size)]
	r.p = p
	r.data = data
}

func (r *SymReader) Uint8(off int) uint8 {
	return r.data[off]
}

func (r *SymReader) Uint16(off int) uint16 {
	return r.p.byteorder.Uint16(r.data[off:])
}

func (r *SymReader) Uint32(off int) uint32 {
	return r.p.byteorder.Uint32(r.data[off:])
}

func (r *SymReader) Uint64(off int) uint64 {
	return r.p.byteorder.Uint64(r.data[off:])
}

func (r *SymReader) Addr(off int) Addr {
	if r.p.ptrsize == 4 {
		return Addr(r.Uint32(off))
	}
	return Addr(r.Uint64(off))
}

func (r *SymReader) String(off int) string {
	end := off
	for r.data[end] != '\x00' {
		end++
	}
	return string(r.data[off:end])
}
