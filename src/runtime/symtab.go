// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// NOTE: Func does not expose the actual unexported fields, because we return *Func
// values to users, and we want to keep them from being able to overwrite the data
// with (say) *f = Func{}.
// All code operating on a *Func must call raw to get the *_func instead.

// A Func represents a Go function in the running binary.
type Func struct {
	opaque struct{} // unexported field to disallow conversions
}

func (f *Func) raw() *_func {
	return (*_func)(unsafe.Pointer(f))
}

// funcdata.h
const (
	_PCDATA_StackMapIndex       = 0
	_FUNCDATA_ArgsPointerMaps   = 0
	_FUNCDATA_LocalsPointerMaps = 1
	_FUNCDATA_DeadValueMaps     = 2
	_ArgsSizeUnknown            = -0x80000000
)

var (
	pclntable []byte
	ftab      []functab
	filetab   []uint32

	pclntab, epclntab struct{} // linker symbols
)

type functab struct {
	entry   uintptr
	funcoff uintptr
}

func symtabinit() {
	// See golang.org/s/go12symtab for header: 0xfffffffb,
	// two zero bytes, a byte giving the PC quantum,
	// and a byte giving the pointer width in bytes.
	pcln := (*[8]byte)(unsafe.Pointer(&pclntab))
	pcln32 := (*[2]uint32)(unsafe.Pointer(&pclntab))
	if pcln32[0] != 0xfffffffb || pcln[4] != 0 || pcln[5] != 0 || pcln[6] != _PCQuantum || pcln[7] != ptrSize {
		println("runtime: function symbol table header:", hex(pcln32[0]), hex(pcln[4]), hex(pcln[5]), hex(pcln[6]), hex(pcln[7]))
		gothrow("invalid function symbol table\n")
	}

	// pclntable is all bytes of pclntab symbol.
	sp := (*sliceStruct)(unsafe.Pointer(&pclntable))
	sp.array = unsafe.Pointer(&pclntab)
	sp.len = int(uintptr(unsafe.Pointer(&epclntab)) - uintptr(unsafe.Pointer(&pclntab)))
	sp.cap = sp.len

	// ftab is lookup table for function by program counter.
	nftab := int(*(*uintptr)(add(unsafe.Pointer(pcln), 8)))
	p := add(unsafe.Pointer(pcln), 8+ptrSize)
	sp = (*sliceStruct)(unsafe.Pointer(&ftab))
	sp.array = p
	sp.len = nftab + 1
	sp.cap = sp.len
	for i := 0; i < nftab; i++ {
		// NOTE: ftab[nftab].entry is legal; it is the address beyond the final function.
		if ftab[i].entry > ftab[i+1].entry {
			f1 := (*_func)(unsafe.Pointer(&pclntable[ftab[i].funcoff]))
			f2 := (*_func)(unsafe.Pointer(&pclntable[ftab[i+1].funcoff]))
			f2name := "end"
			if i+1 < nftab {
				f2name = gofuncname(f2)
			}
			println("function symbol table not sorted by program counter:", hex(ftab[i].entry), gofuncname(f1), ">", hex(ftab[i+1].entry), f2name)
			for j := 0; j <= i; j++ {
				print("\t", hex(ftab[j].entry), " ", gofuncname((*_func)(unsafe.Pointer(&pclntable[ftab[j].funcoff]))), "\n")
			}
			gothrow("invalid runtime symbol table")
		}
	}

	// The ftab ends with a half functab consisting only of
	// 'entry', followed by a uint32 giving the pcln-relative
	// offset of the file table.
	sp = (*sliceStruct)(unsafe.Pointer(&filetab))
	end := unsafe.Pointer(&ftab[nftab].funcoff) // just beyond ftab
	fileoffset := *(*uint32)(end)
	sp.array = unsafe.Pointer(&pclntable[fileoffset])
	// length is in first element of array.
	// set len to 1 so we can get first element.
	sp.len = 1
	sp.cap = 1
	sp.len = int(filetab[0])
	sp.cap = sp.len
}

// FuncForPC returns a *Func describing the function that contains the
// given program counter address, or else nil.
func FuncForPC(pc uintptr) *Func {
	return (*Func)(unsafe.Pointer(findfunc(pc)))
}

// Name returns the name of the function.
func (f *Func) Name() string {
	return gofuncname(f.raw())
}

// Entry returns the entry address of the function.
func (f *Func) Entry() uintptr {
	return f.raw().entry
}

// FileLine returns the file name and line number of the
// source code corresponding to the program counter pc.
// The result will not be accurate if pc is not a program
// counter within f.
func (f *Func) FileLine(pc uintptr) (file string, line int) {
	// Pass strict=false here, because anyone can call this function,
	// and they might just be wrong about targetpc belonging to f.
	file, line32 := funcline1(f.raw(), pc, false)
	return file, int(line32)
}

func findfunc(pc uintptr) *_func {
	if len(ftab) == 0 {
		return nil
	}

	if pc < ftab[0].entry || pc >= ftab[len(ftab)-1].entry {
		return nil
	}

	// binary search to find func with entry <= pc.
	lo := 0
	nf := len(ftab) - 1 // last entry is sentinel
	for nf > 0 {
		n := nf / 2
		f := &ftab[lo+n]
		if f.entry <= pc && pc < ftab[lo+n+1].entry {
			return (*_func)(unsafe.Pointer(&pclntable[f.funcoff]))
		} else if pc < f.entry {
			nf = n
		} else {
			lo += n + 1
			nf -= n + 1
		}
	}

	gothrow("findfunc: binary search failed")
	return nil
}

func pcvalue(f *_func, off int32, targetpc uintptr, strict bool) int32 {
	if off == 0 {
		return -1
	}
	p := pclntable[off:]
	pc := f.entry
	val := int32(-1)
	for {
		var ok bool
		p, ok = step(p, &pc, &val, pc == f.entry)
		if !ok {
			break
		}
		if targetpc < pc {
			return val
		}
	}

	// If there was a table, it should have covered all program counters.
	// If not, something is wrong.
	if panicking != 0 || !strict {
		return -1
	}

	print("runtime: invalid pc-encoded table f=", gofuncname(f), " pc=", hex(pc), " targetpc=", hex(targetpc), " tab=", p, "\n")

	p = pclntable[off:]
	pc = f.entry
	val = -1
	for {
		var ok bool
		p, ok = step(p, &pc, &val, pc == f.entry)
		if !ok {
			break
		}
		print("\tvalue=", val, " until pc=", hex(pc), "\n")
	}

	gothrow("invalid runtime symbol table")
	return -1
}

func funcname(f *_func) *byte {
	if f == nil || f.nameoff == 0 {
		return nil
	}
	return (*byte)(unsafe.Pointer(&pclntable[f.nameoff]))
}

func gofuncname(f *_func) string {
	return gostringnocopy(funcname(f))
}

func funcline1(f *_func, targetpc uintptr, strict bool) (file string, line int32) {
	fileno := int(pcvalue(f, f.pcfile, targetpc, strict))
	line = pcvalue(f, f.pcln, targetpc, strict)
	if fileno == -1 || line == -1 || fileno >= len(filetab) {
		// print("looking for ", hex(targetpc), " in ", gofuncname(f), " got file=", fileno, " line=", lineno, "\n")
		return "?", 0
	}
	file = gostringnocopy(&pclntable[filetab[fileno]])
	return
}

func funcline(f *_func, targetpc uintptr) (file string, line int32) {
	return funcline1(f, targetpc, true)
}

func funcspdelta(f *_func, targetpc uintptr) int32 {
	x := pcvalue(f, f.pcsp, targetpc, true)
	if x&(ptrSize-1) != 0 {
		print("invalid spdelta ", hex(f.entry), " ", hex(targetpc), " ", hex(f.pcsp), " ", x, "\n")
	}
	return x
}

func pcdatavalue(f *_func, table int32, targetpc uintptr) int32 {
	if table < 0 || table >= f.npcdata {
		return -1
	}
	off := *(*int32)(add(unsafe.Pointer(&f.nfuncdata), unsafe.Sizeof(f.nfuncdata)+uintptr(table)*4))
	return pcvalue(f, off, targetpc, true)
}

func funcdata(f *_func, i int32) unsafe.Pointer {
	if i < 0 || i >= f.nfuncdata {
		return nil
	}
	p := add(unsafe.Pointer(&f.nfuncdata), unsafe.Sizeof(f.nfuncdata)+uintptr(f.npcdata)*4)
	if ptrSize == 8 && uintptr(p)&4 != 0 {
		if uintptr(unsafe.Pointer(f))&4 != 0 {
			println("runtime: misaligned func", f)
		}
		p = add(p, 4)
	}
	return *(*unsafe.Pointer)(add(p, uintptr(i)*ptrSize))
}

// step advances to the next pc, value pair in the encoded table.
func step(p []byte, pc *uintptr, val *int32, first bool) (newp []byte, ok bool) {
	p, uvdelta := readvarint(p)
	if uvdelta == 0 && !first {
		return nil, false
	}
	if uvdelta&1 != 0 {
		uvdelta = ^(uvdelta >> 1)
	} else {
		uvdelta >>= 1
	}
	vdelta := int32(uvdelta)
	p, pcdelta := readvarint(p)
	*pc += uintptr(pcdelta * _PCQuantum)
	*val += vdelta
	return p, true
}

// readvarint reads a varint from p.
func readvarint(p []byte) (newp []byte, val uint32) {
	var v, shift uint32
	for {
		b := p[0]
		p = p[1:]
		v |= (uint32(b) & 0x7F) << shift
		if b&0x80 == 0 {
			break
		}
		shift += 7
	}
	return p, v
}
