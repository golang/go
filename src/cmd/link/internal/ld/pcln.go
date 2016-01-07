// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

// funcpctab writes to dst a pc-value table mapping the code in func to the values
// returned by valfunc parameterized by arg. The invocation of valfunc to update the
// current value is, for each p,
//
//	val = valfunc(func, val, p, 0, arg);
//	record val as value at p->pc;
//	val = valfunc(func, val, p, 1, arg);
//
// where func is the function, val is the current value, p is the instruction being
// considered, and arg can be used to further parameterize valfunc.

// pctofileline computes either the file number (arg == 0)
// or the line number (arg == 1) to use at p.
// Because p->lineno applies to p, phase == 0 (before p)
// takes care of the update.

// pctospadj computes the sp adjustment in effect.
// It is oldval plus any adjustment made by p itself.
// The adjustment by p takes effect only after p, so we
// apply the change during phase == 1.

// pctopcdata computes the pcdata value in effect at p.
// A PCDATA instruction sets the value in effect at future
// non-PCDATA instructions.
// Since PCDATA instructions have no width in the final code,
// it does not matter which phase we use for the update.

// iteration over encoded pcdata tables.

func getvarint(pp *[]byte) uint32 {
	v := uint32(0)
	p := *pp
	for shift := 0; ; shift += 7 {
		v |= uint32(p[0]&0x7F) << uint(shift)
		tmp4 := p
		p = p[1:]
		if tmp4[0]&0x80 == 0 {
			break
		}
	}

	*pp = p
	return v
}

func pciternext(it *Pciter) {
	it.pc = it.nextpc
	if it.done != 0 {
		return
	}
	if -cap(it.p) >= -cap(it.d.P[len(it.d.P):]) {
		it.done = 1
		return
	}

	// value delta
	v := getvarint(&it.p)

	if v == 0 && it.start == 0 {
		it.done = 1
		return
	}

	it.start = 0
	dv := int32(v>>1) ^ (int32(v<<31) >> 31)
	it.value += dv

	// pc delta
	v = getvarint(&it.p)

	it.nextpc = it.pc + v*it.pcscale
}

func pciterinit(ctxt *Link, it *Pciter, d *Pcdata) {
	it.d = *d
	it.p = it.d.P
	it.pc = 0
	it.nextpc = 0
	it.value = -1
	it.start = 1
	it.done = 0
	it.pcscale = uint32(ctxt.Arch.Minlc)
	pciternext(it)
}

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

func addvarint(d *Pcdata, val uint32) {
	n := int32(0)
	for v := val; v >= 0x80; v >>= 7 {
		n++
	}
	n++

	old := len(d.P)
	for cap(d.P) < len(d.P)+int(n) {
		d.P = append(d.P[:cap(d.P)], 0)
	}
	d.P = d.P[:old+int(n)]

	p := d.P[old:]
	var v uint32
	for v = val; v >= 0x80; v >>= 7 {
		p[0] = byte(v | 0x80)
		p = p[1:]
	}
	p[0] = byte(v)
}

func addpctab(ftab *LSym, off int32, d *Pcdata) int32 {
	var start int32
	if len(d.P) > 0 {
		start = int32(len(ftab.P))
		Symgrow(Ctxt, ftab, int64(start)+int64(len(d.P)))
		copy(ftab.P[start:], d.P)
	}
	return int32(setuint32(Ctxt, ftab, int64(off), uint32(start)))
}

func ftabaddstring(ftab *LSym, s string) int32 {
	n := int32(len(s)) + 1
	start := int32(len(ftab.P))
	Symgrow(Ctxt, ftab, int64(start)+int64(n)+1)
	copy(ftab.P[start:], s)
	return start
}

func renumberfiles(ctxt *Link, files []*LSym, d *Pcdata) {
	var f *LSym

	// Give files numbers.
	for i := 0; i < len(files); i++ {
		f = files[i]
		if f.Type != obj.SFILEPATH {
			ctxt.Nhistfile++
			f.Value = int64(ctxt.Nhistfile)
			f.Type = obj.SFILEPATH
			f.Next = ctxt.Filesyms
			f.Name = expandGoroot(f.Name)
			ctxt.Filesyms = f
		}
	}

	newval := int32(-1)
	var out Pcdata

	var dv int32
	var it Pciter
	var oldval int32
	var v uint32
	var val int32
	for pciterinit(ctxt, &it, d); it.done == 0; pciternext(&it) {
		// value delta
		oldval = it.value

		if oldval == -1 {
			val = -1
		} else {
			if oldval < 0 || oldval >= int32(len(files)) {
				log.Fatalf("bad pcdata %d", oldval)
			}
			val = int32(files[oldval].Value)
		}

		dv = val - newval
		newval = val
		v = (uint32(dv) << 1) ^ uint32(int32(dv>>31))
		addvarint(&out, v)

		// pc delta
		addvarint(&out, (it.nextpc-it.pc)/it.pcscale)
	}

	// terminating value delta
	addvarint(&out, 0)

	*d = out
}

func container(s *LSym) int {
	// We want to generate func table entries only for the "lowest level" symbols,
	// not containers of subsymbols.
	if s != nil && s.Type&obj.SCONTAINER != 0 {
		return 1
	}
	return 0
}

// pclntab initializes the pclntab symbol with
// runtime function and file name information.

var pclntab_zpcln Pcln

// These variables are used to initialize runtime.firstmoduledata, see symtab.go:symtab.
var pclntabNfunc int32
var pclntabFiletabOffset int32
var pclntabPclntabOffset int32
var pclntabFirstFunc *LSym
var pclntabLastFunc *LSym

func pclntab() {
	funcdata_bytes := int64(0)
	ftab := Linklookup(Ctxt, "runtime.pclntab", 0)
	ftab.Type = obj.SPCLNTAB
	ftab.Reachable = true

	// See golang.org/s/go12symtab for the format. Briefly:
	//	8-byte header
	//	nfunc [thearch.ptrsize bytes]
	//	function table, alternating PC and offset to func struct [each entry thearch.ptrsize bytes]
	//	end PC [thearch.ptrsize bytes]
	//	offset to file table [4 bytes]
	nfunc := int32(0)

	// Find container symbols, mark them with SCONTAINER
	for Ctxt.Cursym = Ctxt.Textp; Ctxt.Cursym != nil; Ctxt.Cursym = Ctxt.Cursym.Next {
		if Ctxt.Cursym.Outer != nil {
			Ctxt.Cursym.Outer.Type |= obj.SCONTAINER
		}
	}

	for Ctxt.Cursym = Ctxt.Textp; Ctxt.Cursym != nil; Ctxt.Cursym = Ctxt.Cursym.Next {
		if container(Ctxt.Cursym) == 0 {
			nfunc++
		}
	}

	pclntabNfunc = nfunc
	Symgrow(Ctxt, ftab, 8+int64(Thearch.Ptrsize)+int64(nfunc)*2*int64(Thearch.Ptrsize)+int64(Thearch.Ptrsize)+4)
	setuint32(Ctxt, ftab, 0, 0xfffffffb)
	setuint8(Ctxt, ftab, 6, uint8(Thearch.Minlc))
	setuint8(Ctxt, ftab, 7, uint8(Thearch.Ptrsize))
	setuintxx(Ctxt, ftab, 8, uint64(nfunc), int64(Thearch.Ptrsize))
	pclntabPclntabOffset = int32(8 + Thearch.Ptrsize)

	nfunc = 0
	var last *LSym
	var end int32
	var funcstart int32
	var i int32
	var it Pciter
	var off int32
	var pcln *Pcln
	for Ctxt.Cursym = Ctxt.Textp; Ctxt.Cursym != nil; Ctxt.Cursym = Ctxt.Cursym.Next {
		last = Ctxt.Cursym
		if container(Ctxt.Cursym) != 0 {
			continue
		}
		pcln = Ctxt.Cursym.Pcln
		if pcln == nil {
			pcln = &pclntab_zpcln
		}

		if pclntabFirstFunc == nil {
			pclntabFirstFunc = Ctxt.Cursym
		}

		funcstart = int32(len(ftab.P))
		funcstart += int32(-len(ftab.P)) & (int32(Thearch.Ptrsize) - 1)

		setaddr(Ctxt, ftab, 8+int64(Thearch.Ptrsize)+int64(nfunc)*2*int64(Thearch.Ptrsize), Ctxt.Cursym)
		setuintxx(Ctxt, ftab, 8+int64(Thearch.Ptrsize)+int64(nfunc)*2*int64(Thearch.Ptrsize)+int64(Thearch.Ptrsize), uint64(funcstart), int64(Thearch.Ptrsize))

		// fixed size of struct, checked below
		off = funcstart

		end = funcstart + int32(Thearch.Ptrsize) + 3*4 + 5*4 + int32(pcln.Npcdata)*4 + int32(pcln.Nfuncdata)*int32(Thearch.Ptrsize)
		if pcln.Nfuncdata > 0 && (end&int32(Thearch.Ptrsize-1) != 0) {
			end += 4
		}
		Symgrow(Ctxt, ftab, int64(end))

		// entry uintptr
		off = int32(setaddr(Ctxt, ftab, int64(off), Ctxt.Cursym))

		// name int32
		off = int32(setuint32(Ctxt, ftab, int64(off), uint32(ftabaddstring(ftab, Ctxt.Cursym.Name))))

		// args int32
		// TODO: Move into funcinfo.
		off = int32(setuint32(Ctxt, ftab, int64(off), uint32(Ctxt.Cursym.Args)))

		// frame int32
		// This has been removed (it was never set quite correctly anyway).
		// Nothing should use it.
		// Leave an obviously incorrect value.
		// TODO: Remove entirely.
		off = int32(setuint32(Ctxt, ftab, int64(off), 0x1234567))

		if pcln != &pclntab_zpcln {
			renumberfiles(Ctxt, pcln.File, &pcln.Pcfile)
			if false {
				// Sanity check the new numbering
				for pciterinit(Ctxt, &it, &pcln.Pcfile); it.done == 0; pciternext(&it) {
					if it.value < 1 || it.value > Ctxt.Nhistfile {
						Diag("bad file number in pcfile: %d not in range [1, %d]\n", it.value, Ctxt.Nhistfile)
						errorexit()
					}
				}
			}
		}

		// pcdata
		off = addpctab(ftab, off, &pcln.Pcsp)

		off = addpctab(ftab, off, &pcln.Pcfile)
		off = addpctab(ftab, off, &pcln.Pcline)
		off = int32(setuint32(Ctxt, ftab, int64(off), uint32(pcln.Npcdata)))
		off = int32(setuint32(Ctxt, ftab, int64(off), uint32(pcln.Nfuncdata)))
		for i = 0; i < int32(pcln.Npcdata); i++ {
			off = addpctab(ftab, off, &pcln.Pcdata[i])
		}

		// funcdata, must be pointer-aligned and we're only int32-aligned.
		// Missing funcdata will be 0 (nil pointer).
		if pcln.Nfuncdata > 0 {
			if off&int32(Thearch.Ptrsize-1) != 0 {
				off += 4
			}
			for i = 0; i < int32(pcln.Nfuncdata); i++ {
				if pcln.Funcdata[i] == nil {
					setuintxx(Ctxt, ftab, int64(off)+int64(Thearch.Ptrsize)*int64(i), uint64(pcln.Funcdataoff[i]), int64(Thearch.Ptrsize))
				} else {
					// TODO: Dedup.
					funcdata_bytes += pcln.Funcdata[i].Size

					setaddrplus(Ctxt, ftab, int64(off)+int64(Thearch.Ptrsize)*int64(i), pcln.Funcdata[i], pcln.Funcdataoff[i])
				}
			}

			off += int32(pcln.Nfuncdata) * int32(Thearch.Ptrsize)
		}

		if off != end {
			Diag("bad math in functab: funcstart=%d off=%d but end=%d (npcdata=%d nfuncdata=%d ptrsize=%d)", funcstart, off, end, pcln.Npcdata, pcln.Nfuncdata, Thearch.Ptrsize)
			errorexit()
		}

		nfunc++
	}

	pclntabLastFunc = last
	// Final entry of table is just end pc.
	setaddrplus(Ctxt, ftab, 8+int64(Thearch.Ptrsize)+int64(nfunc)*2*int64(Thearch.Ptrsize), last, last.Size)

	// Start file table.
	start := int32(len(ftab.P))

	start += int32(-len(ftab.P)) & (int32(Thearch.Ptrsize) - 1)
	pclntabFiletabOffset = start
	setuint32(Ctxt, ftab, 8+int64(Thearch.Ptrsize)+int64(nfunc)*2*int64(Thearch.Ptrsize)+int64(Thearch.Ptrsize), uint32(start))

	Symgrow(Ctxt, ftab, int64(start)+(int64(Ctxt.Nhistfile)+1)*4)
	setuint32(Ctxt, ftab, int64(start), uint32(Ctxt.Nhistfile))
	for s := Ctxt.Filesyms; s != nil; s = s.Next {
		setuint32(Ctxt, ftab, int64(start)+s.Value*4, uint32(ftabaddstring(ftab, s.Name)))
	}

	ftab.Size = int64(len(ftab.P))

	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f pclntab=%d bytes, funcdata total %d bytes\n", obj.Cputime(), int64(ftab.Size), int64(funcdata_bytes))
	}
}

func expandGoroot(s string) string {
	const n = len("$GOROOT")
	if len(s) >= n+1 && s[:n] == "$GOROOT" && (s[n] == '/' || s[n] == '\\') {
		root := goroot
		if final := os.Getenv("GOROOT_FINAL"); final != "" {
			root = final
		}
		return filepath.ToSlash(filepath.Join(root, s[n:]))
	}
	return s
}

const (
	BUCKETSIZE    = 256 * MINFUNC
	SUBBUCKETS    = 16
	SUBBUCKETSIZE = BUCKETSIZE / SUBBUCKETS
	NOIDX         = 0x7fffffff
)

// findfunctab generates a lookup table to quickly find the containing
// function for a pc.  See src/runtime/symtab.go:findfunc for details.
func findfunctab() {
	t := Linklookup(Ctxt, "runtime.findfunctab", 0)
	t.Type = obj.SRODATA
	t.Reachable = true
	t.Local = true

	// find min and max address
	min := Ctxt.Textp.Value

	max := int64(0)
	for s := Ctxt.Textp; s != nil; s = s.Next {
		max = s.Value + s.Size
	}

	// for each subbucket, compute the minimum of all symbol indexes
	// that map to that subbucket.
	n := int32((max - min + SUBBUCKETSIZE - 1) / SUBBUCKETSIZE)

	indexes := make([]int32, n)
	for i := int32(0); i < n; i++ {
		indexes[i] = NOIDX
	}
	idx := int32(0)
	var e *LSym
	var i int32
	var p int64
	var q int64
	for s := Ctxt.Textp; s != nil; s = s.Next {
		if container(s) != 0 {
			continue
		}
		p = s.Value
		e = s.Next
		for container(e) != 0 {
			e = e.Next
		}
		if e != nil {
			q = e.Value
		} else {
			q = max
		}

		//print("%d: [%lld %lld] %s\n", idx, p, q, s->name);
		for ; p < q; p += SUBBUCKETSIZE {
			i = int32((p - min) / SUBBUCKETSIZE)
			if indexes[i] > idx {
				indexes[i] = idx
			}
		}

		i = int32((q - 1 - min) / SUBBUCKETSIZE)
		if indexes[i] > idx {
			indexes[i] = idx
		}
		idx++
	}

	// allocate table
	nbuckets := int32((max - min + BUCKETSIZE - 1) / BUCKETSIZE)

	Symgrow(Ctxt, t, 4*int64(nbuckets)+int64(n))

	// fill in table
	var base int32
	var j int32
	for i := int32(0); i < nbuckets; i++ {
		base = indexes[i*SUBBUCKETS]
		if base == NOIDX {
			Diag("hole in findfunctab")
		}
		setuint32(Ctxt, t, int64(i)*(4+SUBBUCKETS), uint32(base))
		for j = 0; j < SUBBUCKETS && i*SUBBUCKETS+j < n; j++ {
			idx = indexes[i*SUBBUCKETS+j]
			if idx == NOIDX {
				Diag("hole in findfunctab")
			}
			if idx-base >= 256 {
				Diag("too many functions in a findfunc bucket! %d/%d %d %d", i, nbuckets, j, idx-base)
			}

			setuint8(Ctxt, t, int64(i)*(4+SUBBUCKETS)+4+int64(j), uint8(idx-base))
		}
	}
}
