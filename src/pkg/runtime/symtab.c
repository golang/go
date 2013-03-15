// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime symbol table parsing.
//
// The Go tools use a symbol table derived from the Plan 9 symbol table
// format. The symbol table is kept in its own section treated as
// read-only memory when the binary is running: the binary consults the
// table.
// 
// The format used by Go 1.0 was basically the Plan 9 format. Each entry
// is variable sized but had this format:
// 
// 	4-byte value, big endian
// 	1-byte type ([A-Za-z] + 0x80)
// 	name, NUL terminated (or for 'z' and 'Z' entries, double-NUL terminated)
// 	4-byte Go type address, big endian (new in Go)
// 
// In order to support greater interoperation with standard toolchains,
// Go 1.1 uses a more flexible yet smaller encoding of the entries.
// The overall structure is unchanged from Go 1.0 and, for that matter,
// from Plan 9.
// 
// The Go 1.1 table is a re-encoding of the data in a Go 1.0 table.
// To identify a new table as new, it begins one of two eight-byte
// sequences:
// 
// 	FF FF FF FD 00 00 00 xx - big endian new table
// 	FD FF FF FF 00 00 00 xx - little endian new table
// 
// This sequence was chosen because old tables stop at an entry with type
// 0, so old code reading a new table will see only an empty table. The
// first four bytes are the target-endian encoding of 0xfffffffd. The
// final xx gives AddrSize, the width of a full-width address.
// 
// After that header, each entry is encoded as follows.
// 
// 	1-byte type (0-51 + two flag bits)
// 	AddrSize-byte value, host-endian OR varint-encoded value
// 	AddrSize-byte Go type address OR nothing
// 	[n] name, terminated as before
// 
// The type byte comes first, but 'A' encodes as 0 and 'a' as 26, so that
// the type itself is only in the low 6 bits. The upper two bits specify
// the format of the next two fields. If the 0x40 bit is set, the value
// is encoded as an full-width 4- or 8-byte target-endian word. Otherwise
// the value is a varint-encoded number. If the 0x80 bit is set, the Go
// type is present, again as a 4- or 8-byte target-endian word. If not,
// there is no Go type in this entry. The NUL-terminated name ends the
// entry.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "arch_GOARCH.h"
#include "malloc.h"

extern byte pclntab[], epclntab[], symtab[], esymtab[];

typedef struct Sym Sym;
struct Sym
{
	uintptr value;
	byte symtype;
	byte *name;
//	byte *gotype;
};

static uintptr mainoffset;

// A dynamically allocated string containing multiple substrings.
// Individual strings are slices of hugestring.
static String hugestring;
static int32 hugestring_len;

extern void main·main(void);

static uintptr
readword(byte **pp, byte *ep)
{
	byte *p; 

	p = *pp;
	if(ep - p < sizeof(void*)) {
		*pp = ep;
		return 0;
	}
	*pp = p + sizeof(void*);

	// Hairy, but only one of these four cases gets compiled.
	if(sizeof(void*) == 8) {
		if(BigEndian) {
			return ((uint64)p[0]<<56) | ((uint64)p[1]<<48) | ((uint64)p[2]<<40) | ((uint64)p[3]<<32) |
				((uint64)p[4]<<24) | ((uint64)p[5]<<16) | ((uint64)p[6]<<8) | ((uint64)p[7]);
		}
		return ((uint64)p[7]<<56) | ((uint64)p[6]<<48) | ((uint64)p[5]<<40) | ((uint64)p[4]<<32) |
			((uint64)p[3]<<24) | ((uint64)p[2]<<16) | ((uint64)p[1]<<8) | ((uint64)p[0]);
	}
	if(BigEndian) {
		return ((uint32)p[0]<<24) | ((uint32)p[1]<<16) | ((uint32)p[2]<<8) | ((uint32)p[3]);
	}
	return ((uint32)p[3]<<24) | ((uint32)p[2]<<16) | ((uint32)p[1]<<8) | ((uint32)p[0]);
}

// Walk over symtab, calling fn(&s) for each symbol.
static void
walksymtab(void (*fn)(Sym*))
{
	byte *p, *ep, *q;
	Sym s;
	int32 widevalue, havetype, shift;

	p = symtab;
	ep = esymtab;

	// Table must begin with correct magic number.
	if(ep - p < 8 || p[4] != 0x00 || p[5] != 0x00 || p[6] != 0x00 || p[7] != sizeof(void*))
		return;
	if(BigEndian) {
		if(p[0] != 0xff || p[1] != 0xff || p[2] != 0xff || p[3] != 0xfd)
			return;
	} else {
		if(p[0] != 0xfd || p[1] != 0xff || p[2] != 0xff || p[3] != 0xff)
			return;
	}
	p += 8;

	while(p < ep) {
		s.symtype = p[0]&0x3F;
		widevalue = p[0]&0x40;
		havetype = p[0]&0x80;
		if(s.symtype < 26)
			s.symtype += 'A';
		else
			s.symtype += 'a' - 26;
		p++;

		// Value, either full-width or varint-encoded.
		if(widevalue) {
			s.value = readword(&p, ep);
		} else {
			s.value = 0;
			shift = 0;
			while(p < ep && (p[0]&0x80) != 0) {
				s.value |= (uintptr)(p[0]&0x7F)<<shift;
				shift += 7;
				p++;
			}
			if(p >= ep)
				break;
			s.value |= (uintptr)p[0]<<shift;
			p++;
		}
		
		// Go type, if present. Ignored but must skip over.
		if(havetype)
			readword(&p, ep);

		// Name.
		if(ep - p < 2)
			break;

		s.name = p;
		if(s.symtype == 'z' || s.symtype == 'Z') {
			// path reference string - skip first byte,
			// then 2-byte pairs ending at two zeros.
			q = p+1;
			for(;;) {
				if(q+2 > ep)
					return;
				if(q[0] == '\0' && q[1] == '\0')
					break;
				q += 2;
			}
			p = q+2;
		}else{
			q = runtime·mchr(p, '\0', ep);
			if(q == nil)
				break;
			p = q+1;
		}
	
		fn(&s);
	}
}

// Symtab walker; accumulates info about functions.

static Func *func;
static int32 nfunc;

static byte **fname;
static int32 nfname;

static uint32 funcinit;
static Lock funclock;
static uintptr lastvalue;

static void
dofunc(Sym *sym)
{
	Func *f;
	
	switch(sym->symtype) {
	case 't':
	case 'T':
	case 'l':
	case 'L':
		if(runtime·strcmp(sym->name, (byte*)"etext") == 0)
			break;
		if(sym->value < lastvalue) {
			runtime·printf("symbols out of order: %p before %p\n", lastvalue, sym->value);
			runtime·throw("malformed symbol table");
		}
		lastvalue = sym->value;
		if(func == nil) {
			nfunc++;
			break;
		}
		f = &func[nfunc++];
		f->name = runtime·gostringnocopy(sym->name);
		f->entry = sym->value;
		if(sym->symtype == 'L' || sym->symtype == 'l')
			f->frame = -sizeof(uintptr);
		break;
	case 'm':
		if(nfunc <= 0 || func == nil)
			break;
		if(runtime·strcmp(sym->name, (byte*)".frame") == 0)
			func[nfunc-1].frame = sym->value;
		else if(runtime·strcmp(sym->name, (byte*)".locals") == 0)
			func[nfunc-1].locals = sym->value;
		else if(runtime·strcmp(sym->name, (byte*)".args") == 0)
			func[nfunc-1].args = sym->value;
		else {
			runtime·printf("invalid 'm' symbol named '%s'\n", sym->name);
			runtime·throw("mangled symbol table");
		}
		break;
	case 'f':
		if(fname == nil) {
			if(sym->value >= nfname) {
				if(sym->value >= 0x10000) {
					runtime·printf("runtime: invalid symbol file index %p\n", sym->value);
					runtime·throw("mangled symbol table");
				}
				nfname = sym->value+1;
			}
			break;
		}
		fname[sym->value] = sym->name;
		break;
	}
}

// put together the path name for a z entry.
// the f entries have been accumulated into fname already.
// returns the length of the path name.
static int32
makepath(byte *buf, int32 nbuf, byte *path)
{
	int32 n, len;
	byte *p, *ep, *q;

	if(nbuf <= 0)
		return 0;

	p = buf;
	ep = buf + nbuf;
	*p = '\0';
	for(;;) {
		if(path[0] == 0 && path[1] == 0)
			break;
		n = (path[0]<<8) | path[1];
		path += 2;
		if(n >= nfname)
			break;
		q = fname[n];
		len = runtime·findnull(q);
		if(p+1+len >= ep)
			break;
		if(p > buf && p[-1] != '/')
			*p++ = '/';
		runtime·memmove(p, q, len+1);
		p += len;
	}
	return p - buf;
}

// appends p to hugestring
static String
gostringn(byte *p, int32 l)
{
	String s;

	if(l == 0)
		return runtime·emptystring;
	if(hugestring.str == nil) {
		hugestring_len += l;
		return runtime·emptystring;
	}
	s.str = hugestring.str + hugestring.len;
	s.len = l;
	hugestring.len += s.len;
	runtime·memmove(s.str, p, l);
	return s;
}

// walk symtab accumulating path names for use by pc/ln table.
// don't need the full generality of the z entry history stack because
// there are no includes in go (and only sensible includes in our c);
// assume code only appear in top-level files.
static void
dosrcline(Sym *sym)
{
	static byte srcbuf[1000];
	static struct {
		String srcstring;
		int32 aline;
		int32 delta;
	} files[200];
	static int32 incstart;
	static int32 nfunc, nfile, nhist;
	Func *f;
	int32 i, l;

	switch(sym->symtype) {
	case 't':
	case 'T':
		if(hugestring.str == nil)
			break;
		if(runtime·strcmp(sym->name, (byte*)"etext") == 0)
			break;
		f = &func[nfunc++];
		// find source file
		for(i = 0; i < nfile - 1; i++) {
			if (files[i+1].aline > f->ln0)
				break;
		}
		f->src = files[i].srcstring;
		f->ln0 -= files[i].delta;
		break;
	case 'z':
		if(sym->value == 1) {
			// entry for main source file for a new object.
			l = makepath(srcbuf, sizeof srcbuf, sym->name+1);
			nhist = 0;
			nfile = 0;
			if(nfile == nelem(files))
				return;
			files[nfile].srcstring = gostringn(srcbuf, l);
			files[nfile].aline = 0;
			files[nfile++].delta = 0;
		} else {
			// push or pop of included file.
			l = makepath(srcbuf, sizeof srcbuf, sym->name+1);
			if(srcbuf[0] != '\0') {
				if(nhist++ == 0)
					incstart = sym->value;
				if(nhist == 0 && nfile < nelem(files)) {
					// new top-level file
					files[nfile].srcstring = gostringn(srcbuf, l);
					files[nfile].aline = sym->value;
					// this is "line 0"
					files[nfile++].delta = sym->value - 1;
				}
			}else{
				if(--nhist == 0)
					files[nfile-1].delta += sym->value - incstart;
			}
		}
	}
}

// Interpret pc/ln table, saving the subpiece for each func.
static void
splitpcln(void)
{
	int32 line;
	uintptr pc;
	byte *p, *ep;
	Func *f, *ef;
	int32 pcquant;

	if(pclntab == epclntab || nfunc == 0)
		return;

	switch(thechar) {
	case '5':
		pcquant = 4;
		break;
	default:	// 6, 8
		pcquant = 1;
		break;
	}

	// pc/ln table bounds
	p = pclntab;
	ep = epclntab;

	f = func;
	ef = func + nfunc;
	pc = func[0].entry;	// text base
	f->pcln.array = p;
	f->pc0 = pc;
	line = 0;
	for(;;) {
		while(p < ep && *p > 128)
			pc += pcquant * (*p++ - 128);
		// runtime·printf("pc<%p targetpc=%p line=%d\n", pc, targetpc, line);
		if(*p == 0) {
			if(p+5 > ep)
				break;
			// 4 byte add to line
			line += (p[1]<<24) | (p[2]<<16) | (p[3]<<8) | p[4];
			p += 5;
		} else if(*p <= 64)
			line += *p++;
		else
			line -= *p++ - 64;

		// pc, line now match.
		// Because the state machine begins at pc==entry and line==0,
		// it can happen - just at the beginning! - that the update may
		// have updated line but left pc alone, to tell us the true line
		// number for pc==entry.  In that case, update f->ln0.
		// Having the correct initial line number is important for choosing
		// the correct file in dosrcline above.
		if(f == func && pc == f->pc0) {
			f->pcln.array = p;
			f->pc0 = pc + pcquant;
			f->ln0 = line;
		}

		if(f < ef && pc >= (f+1)->entry) {
			f->pcln.len = p - f->pcln.array;
			f->pcln.cap = f->pcln.len;
			do
				f++;
			while(f < ef && pc >= (f+1)->entry);
			f->pcln.array = p;
			// pc0 and ln0 are the starting values for
			// the loop over f->pcln, so pc must be
			// adjusted by the same pcquant update
			// that we're going to do as we continue our loop.
			f->pc0 = pc + pcquant;
			f->ln0 = line;
		}

		pc += pcquant;
	}
	if(f < ef) {
		f->pcln.len = p - f->pcln.array;
		f->pcln.cap = f->pcln.len;
	}
}


// Return actual file line number for targetpc in func f.
// (Source file is f->src.)
// NOTE(rsc): If you edit this function, also edit extern.go:/FileLine
int32
runtime·funcline(Func *f, uintptr targetpc)
{
	byte *p, *ep;
	uintptr pc;
	int32 line;
	int32 pcquant;

	enum {
		debug = 0
	};

	switch(thechar) {
	case '5':
		pcquant = 4;
		break;
	default:	// 6, 8
		pcquant = 1;
		break;
	}

	p = f->pcln.array;
	ep = p + f->pcln.len;
	pc = f->pc0;
	line = f->ln0;
	if(debug && !runtime·panicking)
		runtime·printf("funcline start pc=%p targetpc=%p line=%d tab=%p+%d\n",
			pc, targetpc, line, p, (int32)f->pcln.len);
	for(;;) {
		// Table is a sequence of updates.

		// Each update says first how to adjust the pc,
		// in possibly multiple instructions...
		while(p < ep && *p > 128)
			pc += pcquant * (*p++ - 128);

		if(debug && !runtime·panicking)
			runtime·printf("pc<%p targetpc=%p line=%d\n", pc, targetpc, line);

		// If the pc has advanced too far or we're out of data,
		// stop and the last known line number.
		if(pc > targetpc || p >= ep)
			break;

		// ... and then how to adjust the line number,
		// in a single instruction.
		if(*p == 0) {
			if(p+5 > ep)
				break;
			line += (p[1]<<24) | (p[2]<<16) | (p[3]<<8) | p[4];
			p += 5;
		} else if(*p <= 64)
			line += *p++;
		else
			line -= *p++ - 64;
		// Now pc, line pair is consistent.
		if(debug && !runtime·panicking)
			runtime·printf("pc=%p targetpc=%p line=%d\n", pc, targetpc, line);

		// PC increments implicitly on each iteration.
		pc += pcquant;
	}
	return line;
}

void
runtime·funcline_go(Func *f, uintptr targetpc, String retfile, intgo retline)
{
	retfile = f->src;
	retline = runtime·funcline(f, targetpc);
	FLUSH(&retfile);
	FLUSH(&retline);
}

static void
buildfuncs(void)
{
	extern byte etext[];

	if(func != nil)
		return;

	// Memory profiling uses this code;
	// can deadlock if the profiler ends
	// up back here.
	m->nomemprof++;

	// count funcs, fnames
	nfunc = 0;
	nfname = 0;
	lastvalue = 0;
	walksymtab(dofunc);

	// Initialize tables.
	// Can use FlagNoPointers - all pointers either point into sections of the executable
	// or point into hugestring.
	func = runtime·mallocgc((nfunc+1)*sizeof func[0], FlagNoPointers, 0, 1);
	func[nfunc].entry = (uint64)etext;
	fname = runtime·mallocgc(nfname*sizeof fname[0], FlagNoPointers, 0, 1);
	nfunc = 0;
	lastvalue = 0;
	walksymtab(dofunc);

	// split pc/ln table by func
	splitpcln();

	// record src file and line info for each func
	walksymtab(dosrcline);  // pass 1: determine hugestring_len
	hugestring.str = runtime·mallocgc(hugestring_len, FlagNoPointers, 0, 0);
	hugestring.len = 0;
	walksymtab(dosrcline);  // pass 2: fill and use hugestring

	if(hugestring.len != hugestring_len)
		runtime·throw("buildfunc: problem in initialization procedure");

	m->nomemprof--;
}

Func*
runtime·findfunc(uintptr addr)
{
	Func *f;
	int32 nf, n;

	// Use atomic double-checked locking,
	// because when called from pprof signal
	// handler, findfunc must run without
	// grabbing any locks.
	// (Before enabling the signal handler,
	// SetCPUProfileRate calls findfunc to trigger
	// the initialization outside the handler.)
	// Avoid deadlock on fault during malloc
	// by not calling buildfuncs if we're already in malloc.
	if(!m->mallocing && !m->gcing) {
		if(runtime·atomicload(&funcinit) == 0) {
			runtime·lock(&funclock);
			if(funcinit == 0) {
				buildfuncs();
				runtime·atomicstore(&funcinit, 1);
			}
			runtime·unlock(&funclock);
		}
	}

	if(nfunc == 0)
		return nil;
	if(addr < func[0].entry || addr >= func[nfunc].entry)
		return nil;

	// binary search to find func with entry <= addr.
	f = func;
	nf = nfunc;
	while(nf > 0) {
		n = nf/2;
		if(f[n].entry <= addr && addr < f[n+1].entry)
			return &f[n];
		else if(addr < f[n].entry)
			nf = n;
		else {
			f += n+1;
			nf -= n+1;
		}
	}

	// can't get here -- we already checked above
	// that the address was in the table bounds.
	// this can only happen if the table isn't sorted
	// by address or if the binary search above is buggy.
	runtime·prints("findfunc unreachable\n");
	return nil;
}

static bool
hasprefix(String s, int8 *p)
{
	int32 i;

	for(i=0; i<s.len; i++) {
		if(p[i] == 0)
			return 1;
		if(p[i] != s.str[i])
			return 0;
	}
	return p[i] == 0;
}

static bool
contains(String s, int8 *p)
{
	int32 i;

	if(p[0] == 0)
		return 1;
	for(i=0; i<s.len; i++) {
		if(s.str[i] != p[0])
			continue;
		if(hasprefix((String){s.str + i, s.len - i}, p))
			return 1;
	}
	return 0;
}

bool
runtime·showframe(Func *f, bool current)
{
	static int32 traceback = -1;

	if(current && m->throwing > 0)
		return 1;
	if(traceback < 0)
		traceback = runtime·gotraceback(nil);
	return traceback > 1 || f != nil && contains(f->name, ".") && !hasprefix(f->name, "runtime.");
}
