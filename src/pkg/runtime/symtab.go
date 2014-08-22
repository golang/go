// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// FuncForPC returns a *Func describing the function that contains the
// given program counter address, or else nil.
func FuncForPC(pc uintptr) *Func {
	if nftab == 0 {
		return nil
	}

	if pc < ftabi(0).entry || pc >= ftabi(nftab).entry {
		return nil
	}

	// binary search to find func with entry <= pc.
	lo := uintptr(0)
	nf := nftab
	for nf > 0 {
		n := nf / 2
		f := ftabi(lo + n)
		if f.entry <= pc && pc < ftabi(lo+n+1).entry {
			return (*Func)(unsafe.Pointer(pclntab + f.funcoff))
		} else if pc < f.entry {
			nf = n
		} else {
			lo += n + 1
			nf -= n + 1
		}
	}

	gothrow("FuncForPC: binary search failed")
	return nil
}

// Name returns the name of the function.
func (f *Func) Name() string {
	return cstringToGo(pclntab + uintptr(f.nameoff))
}

// Entry returns the entry address of the function.
func (f *Func) Entry() uintptr {
	return f.entry
}

// FileLine returns the file name and line number of the
// source code corresponding to the program counter pc.
// The result will not be accurate if pc is not a program
// counter within f.
func (f *Func) FileLine(pc uintptr) (file string, line int) {
	fileno := f.pcvalue(f.pcfile, pc)
	if fileno == -1 || fileno >= int32(nfiletab) {
		return "?", 0
	}
	line = int(f.pcvalue(f.pcln, pc))
	if line == -1 {
		return "?", 0
	}
	file = cstringToGo(pclntab + uintptr(filetabi(uintptr(fileno))))
	return file, line
}

// Return associated data value for targetpc in func f.
func (f *Func) pcvalue(off int32, targetpc uintptr) int32 {
	if off == 0 {
		return -1
	}
	p := pclntab + uintptr(off)
	pc := f.entry
	val := int32(-1)
	for step(&p, &pc, &val, pc == f.entry) {
		if targetpc < pc {
			return val
		}
	}
	return -1
}

// step advances to the next pc, value pair in the encoded table.
func step(p *uintptr, pc *uintptr, val *int32, first bool) bool {
	uvdelta := readvarint(p)
	if uvdelta == 0 && !first {
		return false
	}
	if uvdelta&1 != 0 {
		uvdelta = ^(uvdelta >> 1)
	} else {
		uvdelta >>= 1
	}
	vdelta := int32(uvdelta)
	pcdelta := readvarint(p) * pcquantum
	*pc += uintptr(pcdelta)
	*val += vdelta
	return true
}

// readvarint reads a varint from *p and advances *p.
func readvarint(pp *uintptr) uint32 {
	var v, shift uint32
	p := *pp
	for {
		b := *(*byte)(unsafe.Pointer(p))
		p++
		v |= (uint32(b) & 0x7F) << shift
		if b&0x80 == 0 {
			break
		}
		shift += 7
	}
	*pp = p
	return v
}

// Populated by runtime·symtabinit during bootstrapping. Treat as immutable.
var (
	pclntab   uintptr // address of pclntab
	ftab0     uintptr // address of first ftab entry
	nftab     uintptr
	filetab0  uintptr // address of first filetab entry
	nfiletab  uint32
	pcquantum uint32
)

type Func struct {
	entry   uintptr // start pc
	nameoff int32   // function name

	args  int32 // in/out args size
	frame int32 // legacy frame size; use pcsp if possible

	pcsp      int32
	pcfile    int32
	pcln      int32
	npcdata   int32
	nfuncdata int32
}

type ftab struct {
	entry   uintptr
	funcoff uintptr
}

func ftabi(i uintptr) (f ftab) {
	return *(*ftab)(unsafe.Pointer(ftab0 + i*unsafe.Sizeof(f)))
}

func filetabi(i uintptr) (f uint32) {
	return *(*uint32)(unsafe.Pointer(filetab0 + i*unsafe.Sizeof(f)))
}
