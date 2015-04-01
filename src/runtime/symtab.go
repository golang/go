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

// moduledata records information about the layout of the executable
// image. It is written by the linker. Any changes here must be
// matched changes to the code in cmd/internal/ld/symtab.go:symtab.
type moduledata struct {
	pclntable    []byte
	ftab         []functab
	filetab      []uint32
	findfunctab  uintptr
	minpc, maxpc uintptr

	text, etext           uintptr
	noptrdata, enoptrdata uintptr
	data, edata           uintptr
	bss, ebss             uintptr
	noptrbss, enoptrbss   uintptr
	end, gcdata, gcbss    uintptr

	typelinks []*_type

	gcdatamask, gcbssmask bitvector

	// write barrier shadow data
	// 64-bit systems only, enabled by GODEBUG=wbshadow=1.
	// See also the shadow_* fields on mheap in mheap.go.
	shadow_data uintptr // data-addr + shadow_data = shadow data addr
	data_start  uintptr // start of shadowed data addresses
	data_end    uintptr // end of shadowed data addresses

	next *moduledata
}

var firstmoduledata moduledata  // linker symbol
var lastmoduledatap *moduledata // linker symbol

type functab struct {
	entry   uintptr
	funcoff uintptr
}

const minfunc = 16                 // minimum function size
const pcbucketsize = 256 * minfunc // size of bucket in the pc->func lookup table

// findfunctab is an array of these structures.
// Each bucket represents 4096 bytes of the text segment.
// Each subbucket represents 256 bytes of the text segment.
// To find a function given a pc, locate the bucket and subbucket for
// that pc.  Add together the idx and subbucket value to obtain a
// function index.  Then scan the functab array starting at that
// index to find the target function.
// This table uses 20 bytes for every 4096 bytes of code, or ~0.5% overhead.
type findfuncbucket struct {
	idx        uint32
	subbuckets [16]byte
}

func symtabverify() {
	// See golang.org/s/go12symtab for header: 0xfffffffb,
	// two zero bytes, a byte giving the PC quantum,
	// and a byte giving the pointer width in bytes.
	pcln := *(**[8]byte)(unsafe.Pointer(&firstmoduledata.pclntable))
	pcln32 := *(**[2]uint32)(unsafe.Pointer(&firstmoduledata.pclntable))
	if pcln32[0] != 0xfffffffb || pcln[4] != 0 || pcln[5] != 0 || pcln[6] != _PCQuantum || pcln[7] != ptrSize {
		println("runtime: function symbol table header:", hex(pcln32[0]), hex(pcln[4]), hex(pcln[5]), hex(pcln[6]), hex(pcln[7]))
		throw("invalid function symbol table\n")
	}

	// ftab is lookup table for function by program counter.
	nftab := len(firstmoduledata.ftab) - 1
	for i := 0; i < nftab; i++ {
		// NOTE: ftab[nftab].entry is legal; it is the address beyond the final function.
		if firstmoduledata.ftab[i].entry > firstmoduledata.ftab[i+1].entry {
			f1 := (*_func)(unsafe.Pointer(&firstmoduledata.pclntable[firstmoduledata.ftab[i].funcoff]))
			f2 := (*_func)(unsafe.Pointer(&firstmoduledata.pclntable[firstmoduledata.ftab[i+1].funcoff]))
			f2name := "end"
			if i+1 < nftab {
				f2name = funcname(f2)
			}
			println("function symbol table not sorted by program counter:", hex(firstmoduledata.ftab[i].entry), funcname(f1), ">", hex(firstmoduledata.ftab[i+1].entry), f2name)
			for j := 0; j <= i; j++ {
				print("\t", hex(firstmoduledata.ftab[j].entry), " ", funcname((*_func)(unsafe.Pointer(&firstmoduledata.pclntable[firstmoduledata.ftab[j].funcoff]))), "\n")
			}
			throw("invalid runtime symbol table")
		}
	}

	if firstmoduledata.minpc != firstmoduledata.ftab[0].entry ||
		firstmoduledata.maxpc != firstmoduledata.ftab[nftab].entry {
		throw("minpc or maxpc invalid")
	}
}

// FuncForPC returns a *Func describing the function that contains the
// given program counter address, or else nil.
func FuncForPC(pc uintptr) *Func {
	return (*Func)(unsafe.Pointer(findfunc(pc)))
}

// Name returns the name of the function.
func (f *Func) Name() string {
	return funcname(f.raw())
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

func findmoduledatap(pc uintptr) *moduledata {
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if datap.minpc <= pc && pc <= datap.maxpc {
			return datap
		}
	}
	return nil
}

func findfunc(pc uintptr) *_func {
	datap := findmoduledatap(pc)
	if datap == nil {
		return nil
	}
	const nsub = uintptr(len(findfuncbucket{}.subbuckets))

	x := pc - datap.minpc
	b := x / pcbucketsize
	i := x % pcbucketsize / (pcbucketsize / nsub)

	ffb := (*findfuncbucket)(add(unsafe.Pointer(datap.findfunctab), b*unsafe.Sizeof(findfuncbucket{})))
	idx := ffb.idx + uint32(ffb.subbuckets[i])
	if pc < datap.ftab[idx].entry {
		throw("findfunc: bad findfunctab entry")
	}

	// linear search to find func with pc >= entry.
	for datap.ftab[idx+1].entry <= pc {
		idx++
	}
	return (*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[idx].funcoff]))
}

func pcvalue(f *_func, off int32, targetpc uintptr, strict bool) int32 {
	if off == 0 {
		return -1
	}
	datap := findmoduledatap(f.entry) // inefficient
	p := datap.pclntable[off:]
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

	print("runtime: invalid pc-encoded table f=", funcname(f), " pc=", hex(pc), " targetpc=", hex(targetpc), " tab=", p, "\n")

	p = datap.pclntable[off:]
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

	throw("invalid runtime symbol table")
	return -1
}

func cfuncname(f *_func) *byte {
	if f == nil || f.nameoff == 0 {
		return nil
	}
	datap := findmoduledatap(f.entry) // inefficient
	return (*byte)(unsafe.Pointer(&datap.pclntable[f.nameoff]))
}

func funcname(f *_func) string {
	return gostringnocopy(cfuncname(f))
}

func funcline1(f *_func, targetpc uintptr, strict bool) (file string, line int32) {
	datap := findmoduledatap(f.entry) // inefficient
	fileno := int(pcvalue(f, f.pcfile, targetpc, strict))
	line = pcvalue(f, f.pcln, targetpc, strict)
	if fileno == -1 || line == -1 || fileno >= len(datap.filetab) {
		// print("looking for ", hex(targetpc), " in ", funcname(f), " got file=", fileno, " line=", lineno, "\n")
		return "?", 0
	}
	file = gostringnocopy(&datap.pclntable[datap.filetab[fileno]])
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

type stackmap struct {
	n        int32   // number of bitmaps
	nbit     int32   // number of bits in each bitmap
	bytedata [1]byte // bitmaps, each starting on a 32-bit boundary
}

//go:nowritebarrier
func stackmapdata(stkmap *stackmap, n int32) bitvector {
	if n < 0 || n >= stkmap.n {
		throw("stackmapdata: index out of range")
	}
	return bitvector{stkmap.nbit, (*byte)(add(unsafe.Pointer(&stkmap.bytedata), uintptr(n*((stkmap.nbit+31)/32*4))))}
}
