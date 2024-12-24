// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of runtime/debug.WriteHeapDump. Writes all
// objects in the heap plus additional info (roots, threads,
// finalizers, etc.) to a file.

// The format of the dumped file is described at
// https://golang.org/s/go15heapdump.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

//go:linkname runtime_debug_WriteHeapDump runtime/debug.WriteHeapDump
func runtime_debug_WriteHeapDump(fd uintptr) {
	stw := stopTheWorld(stwWriteHeapDump)

	// Keep m on this G's stack instead of the system stack.
	// Both readmemstats_m and writeheapdump_m have pretty large
	// peak stack depths and we risk blowing the system stack.
	// This is safe because the world is stopped, so we don't
	// need to worry about anyone shrinking and therefore moving
	// our stack.
	var m MemStats
	systemstack(func() {
		// Call readmemstats_m here instead of deeper in
		// writeheapdump_m because we might blow the system stack
		// otherwise.
		readmemstats_m(&m)
		writeheapdump_m(fd, &m)
	})

	startTheWorld(stw)
}

const (
	fieldKindEol       = 0
	fieldKindPtr       = 1
	fieldKindIface     = 2
	fieldKindEface     = 3
	tagEOF             = 0
	tagObject          = 1
	tagOtherRoot       = 2
	tagType            = 3
	tagGoroutine       = 4
	tagStackFrame      = 5
	tagParams          = 6
	tagFinalizer       = 7
	tagItab            = 8
	tagOSThread        = 9
	tagMemStats        = 10
	tagQueuedFinalizer = 11
	tagData            = 12
	tagBSS             = 13
	tagDefer           = 14
	tagPanic           = 15
	tagMemProf         = 16
	tagAllocSample     = 17
)

var dumpfd uintptr // fd to write the dump to.
var tmpbuf []byte

// buffer of pending write data
const (
	bufSize = 4096
)

var buf [bufSize]byte
var nbuf uintptr

func dwrite(data unsafe.Pointer, len uintptr) {
	if len == 0 {
		return
	}
	if nbuf+len <= bufSize {
		copy(buf[nbuf:], (*[bufSize]byte)(data)[:len])
		nbuf += len
		return
	}

	write(dumpfd, unsafe.Pointer(&buf), int32(nbuf))
	if len >= bufSize {
		write(dumpfd, data, int32(len))
		nbuf = 0
	} else {
		copy(buf[:], (*[bufSize]byte)(data)[:len])
		nbuf = len
	}
}

func dwritebyte(b byte) {
	dwrite(unsafe.Pointer(&b), 1)
}

func flush() {
	write(dumpfd, unsafe.Pointer(&buf), int32(nbuf))
	nbuf = 0
}

// Cache of types that have been serialized already.
// We use a type's hash field to pick a bucket.
// Inside a bucket, we keep a list of types that
// have been serialized so far, most recently used first.
// Note: when a bucket overflows we may end up
// serializing a type more than once. That's ok.
const (
	typeCacheBuckets = 256
	typeCacheAssoc   = 4
)

type typeCacheBucket struct {
	t [typeCacheAssoc]*_type
}

var typecache [typeCacheBuckets]typeCacheBucket

// dump a uint64 in a varint format parseable by encoding/binary.
func dumpint(v uint64) {
	var buf [10]byte
	var n int
	for v >= 0x80 {
		buf[n] = byte(v | 0x80)
		n++
		v >>= 7
	}
	buf[n] = byte(v)
	n++
	dwrite(unsafe.Pointer(&buf), uintptr(n))
}

func dumpbool(b bool) {
	if b {
		dumpint(1)
	} else {
		dumpint(0)
	}
}

// dump varint uint64 length followed by memory contents.
func dumpmemrange(data unsafe.Pointer, len uintptr) {
	dumpint(uint64(len))
	dwrite(data, len)
}

func dumpslice(b []byte) {
	dumpint(uint64(len(b)))
	if len(b) > 0 {
		dwrite(unsafe.Pointer(&b[0]), uintptr(len(b)))
	}
}

func dumpstr(s string) {
	dumpmemrange(unsafe.Pointer(unsafe.StringData(s)), uintptr(len(s)))
}

// dump information for a type.
func dumptype(t *_type) {
	if t == nil {
		return
	}

	// If we've definitely serialized the type before,
	// no need to do it again.
	b := &typecache[t.Hash&(typeCacheBuckets-1)]
	if t == b.t[0] {
		return
	}
	for i := 1; i < typeCacheAssoc; i++ {
		if t == b.t[i] {
			// Move-to-front
			for j := i; j > 0; j-- {
				b.t[j] = b.t[j-1]
			}
			b.t[0] = t
			return
		}
	}

	// Might not have been dumped yet. Dump it and
	// remember we did so.
	for j := typeCacheAssoc - 1; j > 0; j-- {
		b.t[j] = b.t[j-1]
	}
	b.t[0] = t

	// dump the type
	dumpint(tagType)
	dumpint(uint64(uintptr(unsafe.Pointer(t))))
	dumpint(uint64(t.Size_))
	rt := toRType(t)
	if x := t.Uncommon(); x == nil || rt.nameOff(x.PkgPath).Name() == "" {
		dumpstr(rt.string())
	} else {
		pkgpath := rt.nameOff(x.PkgPath).Name()
		name := rt.name()
		dumpint(uint64(uintptr(len(pkgpath)) + 1 + uintptr(len(name))))
		dwrite(unsafe.Pointer(unsafe.StringData(pkgpath)), uintptr(len(pkgpath)))
		dwritebyte('.')
		dwrite(unsafe.Pointer(unsafe.StringData(name)), uintptr(len(name)))
	}
	dumpbool(t.Kind_&abi.KindDirectIface == 0 || t.Pointers())
}

// dump an object.
func dumpobj(obj unsafe.Pointer, size uintptr, bv bitvector) {
	dumpint(tagObject)
	dumpint(uint64(uintptr(obj)))
	dumpmemrange(obj, size)
	dumpfields(bv)
}

func dumpotherroot(description string, to unsafe.Pointer) {
	dumpint(tagOtherRoot)
	dumpstr(description)
	dumpint(uint64(uintptr(to)))
}

func dumpfinalizer(obj unsafe.Pointer, fn *funcval, fint *_type, ot *ptrtype) {
	dumpint(tagFinalizer)
	dumpint(uint64(uintptr(obj)))
	dumpint(uint64(uintptr(unsafe.Pointer(fn))))
	dumpint(uint64(uintptr(unsafe.Pointer(fn.fn))))
	dumpint(uint64(uintptr(unsafe.Pointer(fint))))
	dumpint(uint64(uintptr(unsafe.Pointer(ot))))
}

type childInfo struct {
	// Information passed up from the callee frame about
	// the layout of the outargs region.
	argoff uintptr   // where the arguments start in the frame
	arglen uintptr   // size of args region
	args   bitvector // if args.n >= 0, pointer map of args region
	sp     *uint8    // callee sp
	depth  uintptr   // depth in call stack (0 == most recent)
}

// dump kinds & offsets of interesting fields in bv.
func dumpbv(cbv *bitvector, offset uintptr) {
	for i := uintptr(0); i < uintptr(cbv.n); i++ {
		if cbv.ptrbit(i) == 1 {
			dumpint(fieldKindPtr)
			dumpint(uint64(offset + i*goarch.PtrSize))
		}
	}
}

func dumpframe(s *stkframe, child *childInfo) {
	f := s.fn

	// Figure out what we can about our stack map
	pc := s.pc
	pcdata := int32(-1) // Use the entry map at function entry
	if pc != f.entry() {
		pc--
		pcdata = pcdatavalue(f, abi.PCDATA_StackMapIndex, pc)
	}
	if pcdata == -1 {
		// We do not have a valid pcdata value but there might be a
		// stackmap for this function. It is likely that we are looking
		// at the function prologue, assume so and hope for the best.
		pcdata = 0
	}
	stkmap := (*stackmap)(funcdata(f, abi.FUNCDATA_LocalsPointerMaps))

	var bv bitvector
	if stkmap != nil && stkmap.n > 0 {
		bv = stackmapdata(stkmap, pcdata)
	} else {
		bv.n = -1
	}

	// Dump main body of stack frame.
	dumpint(tagStackFrame)
	dumpint(uint64(s.sp))                              // lowest address in frame
	dumpint(uint64(child.depth))                       // # of frames deep on the stack
	dumpint(uint64(uintptr(unsafe.Pointer(child.sp)))) // sp of child, or 0 if bottom of stack
	dumpmemrange(unsafe.Pointer(s.sp), s.fp-s.sp)      // frame contents
	dumpint(uint64(f.entry()))
	dumpint(uint64(s.pc))
	dumpint(uint64(s.continpc))
	name := funcname(f)
	if name == "" {
		name = "unknown function"
	}
	dumpstr(name)

	// Dump fields in the outargs section
	if child.args.n >= 0 {
		dumpbv(&child.args, child.argoff)
	} else {
		// conservative - everything might be a pointer
		for off := child.argoff; off < child.argoff+child.arglen; off += goarch.PtrSize {
			dumpint(fieldKindPtr)
			dumpint(uint64(off))
		}
	}

	// Dump fields in the local vars section
	if stkmap == nil {
		// No locals information, dump everything.
		for off := child.arglen; off < s.varp-s.sp; off += goarch.PtrSize {
			dumpint(fieldKindPtr)
			dumpint(uint64(off))
		}
	} else if stkmap.n < 0 {
		// Locals size information, dump just the locals.
		size := uintptr(-stkmap.n)
		for off := s.varp - size - s.sp; off < s.varp-s.sp; off += goarch.PtrSize {
			dumpint(fieldKindPtr)
			dumpint(uint64(off))
		}
	} else if stkmap.n > 0 {
		// Locals bitmap information, scan just the pointers in
		// locals.
		dumpbv(&bv, s.varp-uintptr(bv.n)*goarch.PtrSize-s.sp)
	}
	dumpint(fieldKindEol)

	// Record arg info for parent.
	child.argoff = s.argp - s.fp
	child.arglen = s.argBytes()
	child.sp = (*uint8)(unsafe.Pointer(s.sp))
	child.depth++
	stkmap = (*stackmap)(funcdata(f, abi.FUNCDATA_ArgsPointerMaps))
	if stkmap != nil {
		child.args = stackmapdata(stkmap, pcdata)
	} else {
		child.args.n = -1
	}
	return
}

func dumpgoroutine(gp *g) {
	var sp, pc, lr uintptr
	if gp.syscallsp != 0 {
		sp = gp.syscallsp
		pc = gp.syscallpc
		lr = 0
	} else {
		sp = gp.sched.sp
		pc = gp.sched.pc
		lr = gp.sched.lr
	}

	dumpint(tagGoroutine)
	dumpint(uint64(uintptr(unsafe.Pointer(gp))))
	dumpint(uint64(sp))
	dumpint(gp.goid)
	dumpint(uint64(gp.gopc))
	dumpint(uint64(readgstatus(gp)))
	dumpbool(isSystemGoroutine(gp, false))
	dumpbool(false) // isbackground
	dumpint(uint64(gp.waitsince))
	dumpstr(gp.waitreason.String())
	dumpint(uint64(uintptr(gp.sched.ctxt)))
	dumpint(uint64(uintptr(unsafe.Pointer(gp.m))))
	dumpint(uint64(uintptr(unsafe.Pointer(gp._defer))))
	dumpint(uint64(uintptr(unsafe.Pointer(gp._panic))))

	// dump stack
	var child childInfo
	child.args.n = -1
	child.arglen = 0
	child.sp = nil
	child.depth = 0
	var u unwinder
	for u.initAt(pc, sp, lr, gp, 0); u.valid(); u.next() {
		dumpframe(&u.frame, &child)
	}

	// dump defer & panic records
	for d := gp._defer; d != nil; d = d.link {
		dumpint(tagDefer)
		dumpint(uint64(uintptr(unsafe.Pointer(d))))
		dumpint(uint64(uintptr(unsafe.Pointer(gp))))
		dumpint(uint64(d.sp))
		dumpint(uint64(d.pc))
		fn := *(**funcval)(unsafe.Pointer(&d.fn))
		dumpint(uint64(uintptr(unsafe.Pointer(fn))))
		if d.fn == nil {
			// d.fn can be nil for open-coded defers
			dumpint(uint64(0))
		} else {
			dumpint(uint64(uintptr(unsafe.Pointer(fn.fn))))
		}
		dumpint(uint64(uintptr(unsafe.Pointer(d.link))))
	}
	for p := gp._panic; p != nil; p = p.link {
		dumpint(tagPanic)
		dumpint(uint64(uintptr(unsafe.Pointer(p))))
		dumpint(uint64(uintptr(unsafe.Pointer(gp))))
		eface := efaceOf(&p.arg)
		dumpint(uint64(uintptr(unsafe.Pointer(eface._type))))
		dumpint(uint64(uintptr(eface.data)))
		dumpint(0) // was p->defer, no longer recorded
		dumpint(uint64(uintptr(unsafe.Pointer(p.link))))
	}
}

func dumpgs() {
	assertWorldStopped()

	// goroutines & stacks
	forEachG(func(gp *g) {
		status := readgstatus(gp) // The world is stopped so gp will not be in a scan state.
		switch status {
		default:
			print("runtime: unexpected G.status ", hex(status), "\n")
			throw("dumpgs in STW - bad status")
		case _Gdead:
			// ok
		case _Grunnable,
			_Gsyscall,
			_Gwaiting:
			dumpgoroutine(gp)
		}
	})
}

func finq_callback(fn *funcval, obj unsafe.Pointer, nret uintptr, fint *_type, ot *ptrtype) {
	dumpint(tagQueuedFinalizer)
	dumpint(uint64(uintptr(obj)))
	dumpint(uint64(uintptr(unsafe.Pointer(fn))))
	dumpint(uint64(uintptr(unsafe.Pointer(fn.fn))))
	dumpint(uint64(uintptr(unsafe.Pointer(fint))))
	dumpint(uint64(uintptr(unsafe.Pointer(ot))))
}

func dumproots() {
	// To protect mheap_.allspans.
	assertWorldStopped()

	// TODO(mwhudson): dump datamask etc from all objects
	// data segment
	dumpint(tagData)
	dumpint(uint64(firstmoduledata.data))
	dumpmemrange(unsafe.Pointer(firstmoduledata.data), firstmoduledata.edata-firstmoduledata.data)
	dumpfields(firstmoduledata.gcdatamask)

	// bss segment
	dumpint(tagBSS)
	dumpint(uint64(firstmoduledata.bss))
	dumpmemrange(unsafe.Pointer(firstmoduledata.bss), firstmoduledata.ebss-firstmoduledata.bss)
	dumpfields(firstmoduledata.gcbssmask)

	// mspan.types
	for _, s := range mheap_.allspans {
		if s.state.get() == mSpanInUse {
			// Finalizers
			for sp := s.specials; sp != nil; sp = sp.next {
				if sp.kind != _KindSpecialFinalizer {
					continue
				}
				spf := (*specialfinalizer)(unsafe.Pointer(sp))
				p := unsafe.Pointer(s.base() + uintptr(spf.special.offset))
				dumpfinalizer(p, spf.fn, spf.fint, spf.ot)
			}
		}
	}

	// Finalizer queue
	iterate_finq(finq_callback)
}

// Bit vector of free marks.
// Needs to be as big as the largest number of objects per span.
var freemark [_PageSize / 8]bool

func dumpobjs() {
	// To protect mheap_.allspans.
	assertWorldStopped()

	for _, s := range mheap_.allspans {
		if s.state.get() != mSpanInUse {
			continue
		}
		p := s.base()
		size := s.elemsize
		n := (s.npages << _PageShift) / size
		if n > uintptr(len(freemark)) {
			throw("freemark array doesn't have enough entries")
		}

		for freeIndex := uint16(0); freeIndex < s.nelems; freeIndex++ {
			if s.isFree(uintptr(freeIndex)) {
				freemark[freeIndex] = true
			}
		}

		for j := uintptr(0); j < n; j, p = j+1, p+size {
			if freemark[j] {
				freemark[j] = false
				continue
			}
			dumpobj(unsafe.Pointer(p), size, makeheapobjbv(p, size))
		}
	}
}

func dumpparams() {
	dumpint(tagParams)
	x := uintptr(1)
	if *(*byte)(unsafe.Pointer(&x)) == 1 {
		dumpbool(false) // little-endian ptrs
	} else {
		dumpbool(true) // big-endian ptrs
	}
	dumpint(goarch.PtrSize)
	var arenaStart, arenaEnd uintptr
	for i1 := range mheap_.arenas {
		if mheap_.arenas[i1] == nil {
			continue
		}
		for i, ha := range mheap_.arenas[i1] {
			if ha == nil {
				continue
			}
			base := arenaBase(arenaIdx(i1)<<arenaL1Shift | arenaIdx(i))
			if arenaStart == 0 || base < arenaStart {
				arenaStart = base
			}
			if base+heapArenaBytes > arenaEnd {
				arenaEnd = base + heapArenaBytes
			}
		}
	}
	dumpint(uint64(arenaStart))
	dumpint(uint64(arenaEnd))
	dumpstr(goarch.GOARCH)
	dumpstr(buildVersion)
	dumpint(uint64(ncpu))
}

func itab_callback(tab *itab) {
	t := tab.Type
	dumptype(t)
	dumpint(tagItab)
	dumpint(uint64(uintptr(unsafe.Pointer(tab))))
	dumpint(uint64(uintptr(unsafe.Pointer(t))))
}

func dumpitabs() {
	iterate_itabs(itab_callback)
}

func dumpms() {
	for mp := allm; mp != nil; mp = mp.alllink {
		dumpint(tagOSThread)
		dumpint(uint64(uintptr(unsafe.Pointer(mp))))
		dumpint(uint64(mp.id))
		dumpint(mp.procid)
	}
}

//go:systemstack
func dumpmemstats(m *MemStats) {
	assertWorldStopped()

	// These ints should be identical to the exported
	// MemStats structure and should be ordered the same
	// way too.
	dumpint(tagMemStats)
	dumpint(m.Alloc)
	dumpint(m.TotalAlloc)
	dumpint(m.Sys)
	dumpint(m.Lookups)
	dumpint(m.Mallocs)
	dumpint(m.Frees)
	dumpint(m.HeapAlloc)
	dumpint(m.HeapSys)
	dumpint(m.HeapIdle)
	dumpint(m.HeapInuse)
	dumpint(m.HeapReleased)
	dumpint(m.HeapObjects)
	dumpint(m.StackInuse)
	dumpint(m.StackSys)
	dumpint(m.MSpanInuse)
	dumpint(m.MSpanSys)
	dumpint(m.MCacheInuse)
	dumpint(m.MCacheSys)
	dumpint(m.BuckHashSys)
	dumpint(m.GCSys)
	dumpint(m.OtherSys)
	dumpint(m.NextGC)
	dumpint(m.LastGC)
	dumpint(m.PauseTotalNs)
	for i := 0; i < 256; i++ {
		dumpint(m.PauseNs[i])
	}
	dumpint(uint64(m.NumGC))
}

func dumpmemprof_callback(b *bucket, nstk uintptr, pstk *uintptr, size, allocs, frees uintptr) {
	stk := (*[100000]uintptr)(unsafe.Pointer(pstk))
	dumpint(tagMemProf)
	dumpint(uint64(uintptr(unsafe.Pointer(b))))
	dumpint(uint64(size))
	dumpint(uint64(nstk))
	for i := uintptr(0); i < nstk; i++ {
		pc := stk[i]
		f := findfunc(pc)
		if !f.valid() {
			var buf [64]byte
			n := len(buf)
			n--
			buf[n] = ')'
			if pc == 0 {
				n--
				buf[n] = '0'
			} else {
				for pc > 0 {
					n--
					buf[n] = "0123456789abcdef"[pc&15]
					pc >>= 4
				}
			}
			n--
			buf[n] = 'x'
			n--
			buf[n] = '0'
			n--
			buf[n] = '('
			dumpslice(buf[n:])
			dumpstr("?")
			dumpint(0)
		} else {
			dumpstr(funcname(f))
			if i > 0 && pc > f.entry() {
				pc--
			}
			file, line := funcline(f, pc)
			dumpstr(file)
			dumpint(uint64(line))
		}
	}
	dumpint(uint64(allocs))
	dumpint(uint64(frees))
}

func dumpmemprof() {
	// To protect mheap_.allspans.
	assertWorldStopped()

	iterate_memprof(dumpmemprof_callback)
	for _, s := range mheap_.allspans {
		if s.state.get() != mSpanInUse {
			continue
		}
		for sp := s.specials; sp != nil; sp = sp.next {
			if sp.kind != _KindSpecialProfile {
				continue
			}
			spp := (*specialprofile)(unsafe.Pointer(sp))
			p := s.base() + uintptr(spp.special.offset)
			dumpint(tagAllocSample)
			dumpint(uint64(p))
			dumpint(uint64(uintptr(unsafe.Pointer(spp.b))))
		}
	}
}

var dumphdr = []byte("go1.7 heap dump\n")

func mdump(m *MemStats) {
	assertWorldStopped()

	// make sure we're done sweeping
	for _, s := range mheap_.allspans {
		if s.state.get() == mSpanInUse {
			s.ensureSwept()
		}
	}
	memclrNoHeapPointers(unsafe.Pointer(&typecache), unsafe.Sizeof(typecache))
	dwrite(unsafe.Pointer(&dumphdr[0]), uintptr(len(dumphdr)))
	dumpparams()
	dumpitabs()
	dumpobjs()
	dumpgs()
	dumpms()
	dumproots()
	dumpmemstats(m)
	dumpmemprof()
	dumpint(tagEOF)
	flush()
}

func writeheapdump_m(fd uintptr, m *MemStats) {
	assertWorldStopped()

	gp := getg()
	casGToWaiting(gp.m.curg, _Grunning, waitReasonDumpingHeap)

	// Set dump file.
	dumpfd = fd

	// Call dump routine.
	mdump(m)

	// Reset dump file.
	dumpfd = 0
	if tmpbuf != nil {
		sysFree(unsafe.Pointer(&tmpbuf[0]), uintptr(len(tmpbuf)), &memstats.other_sys)
		tmpbuf = nil
	}

	casgstatus(gp.m.curg, _Gwaiting, _Grunning)
}

// dumpint() the kind & offset of each field in an object.
func dumpfields(bv bitvector) {
	dumpbv(&bv, 0)
	dumpint(fieldKindEol)
}

func makeheapobjbv(p uintptr, size uintptr) bitvector {
	// Extend the temp buffer if necessary.
	nptr := size / goarch.PtrSize
	if uintptr(len(tmpbuf)) < nptr/8+1 {
		if tmpbuf != nil {
			sysFree(unsafe.Pointer(&tmpbuf[0]), uintptr(len(tmpbuf)), &memstats.other_sys)
		}
		n := nptr/8 + 1
		p := sysAlloc(n, &memstats.other_sys)
		if p == nil {
			throw("heapdump: out of memory")
		}
		tmpbuf = (*[1 << 30]byte)(p)[:n]
	}
	// Convert heap bitmap to pointer bitmap.
	clear(tmpbuf[:nptr/8+1])
	s := spanOf(p)
	tp := s.typePointersOf(p, size)
	for {
		var addr uintptr
		if tp, addr = tp.next(p + size); addr == 0 {
			break
		}
		i := (addr - p) / goarch.PtrSize
		tmpbuf[i/8] |= 1 << (i % 8)
	}
	return bitvector{int32(nptr), &tmpbuf[0]}
}
