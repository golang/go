// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <ctype.h>
#include <time.h>

#include <ureg_amd64.h>
#include <mach_amd64.h>

int pid;
char* file = "6.out";
static Fhdr fhdr;
int have_syms;
int fd;
Map *map;
Map	*symmap;
struct Ureg ureg;
int total_sec = 10;
int delta_msec = 100;
int collapse = 1;	// collapse histogram trace points in same function

// output formats
int functions;	// print functions
int histograms;	// print histograms
int linenums;	// print file and line numbers rather than function names
int registers;	// print registers
int stacks;		// print stack traces

void
Usage()
{
	fprint(2, "Usage: prof -p pid [-t total_secs] [-d delta_msec] [6.out]\n");
	fprint(2, "\tformats (default -h):\n");
	fprint(2, "\t\t-h: histograms\n");
	fprint(2, "\t\t-f: dynamic functions\n");
	fprint(2, "\t\t-l: dynamic file and line numbers\n");
	fprint(2, "\t\t-r: dynamic registers\n");
	fprint(2, "\t\t-s: dynamic function stack traces\n");
	exit(2);
}

typedef struct PC PC;
struct PC {
	uvlong pc;
	unsigned int count;
	PC* next;
};

enum {
	Ncounters = 256
};

PC *counters[Ncounters];

void
regprint(void)
{
	print("ax\t0x%llux\n", ureg.ax);
	print("bx\t0x%llux\n", ureg.bx);
	print("cx\t0x%llux\n", ureg.cx);
	print("dx\t0x%llux\n", ureg.dx);
	print("si\t0x%llux\n", ureg.si);
	print("di\t0x%llux\n", ureg.di);
	print("bp\t0x%llux\n", ureg.bp);
	print("r8\t0x%llux\n", ureg.r8);
	print("r9\t0x%llux\n", ureg.r9);
	print("r10\t0x%llux\n", ureg.r10);
	print("r11\t0x%llux\n", ureg.r11);
	print("r12\t0x%llux\n", ureg.r12);
	print("r13\t0x%llux\n", ureg.r13);
	print("r14\t0x%llux\n", ureg.r14);
	print("r15\t0x%llux\n", ureg.r15);
	print("ds\t0x%llux\n", ureg.ds);
	print("es\t0x%llux\n", ureg.es);
	print("fs\t0x%llux\n", ureg.fs);
	print("gs\t0x%llux\n", ureg.gs);
	print("type\t0x%llux\n", ureg.type);
	print("error\t0x%llux\n", ureg.error);
	print("pc\t0x%llux\n", ureg.ip);
	print("cs\t0x%llux\n", ureg.cs);
	print("flags\t0x%llux\n", ureg.flags);
	print("sp\t0x%llux\n", ureg.sp);
	print("ss\t0x%llux\n", ureg.ss);
}

int
sample()
{
	int i;

	ctlproc(pid, "stop");
	for(i = 0; i < sizeof ureg; i+=8) {
		if(get8(map, (uvlong)i, &((uvlong*)&ureg)[i/8]) < 0) {
			fprint(2, "prof: can't read registers at %d: %r\n", i);
			return 0;
		}
	}
	ctlproc(pid, "start");
	return 1;
}

uvlong nextpc;

void
ptrace(Map *map, uvlong pc, uvlong sp, Symbol *sym)
{
	char buf[1024];
	if(nextpc == 0)
		nextpc = sym->value;
	print("%s(", sym->name);
	print(")");
	if(nextpc != sym->value)
		print("+%#llux ", nextpc - sym->value);
	if(have_syms && linenums && fileline(buf, sizeof buf, pc)) {
		print(" %s", buf);
	}
	print("\n");
	nextpc = pc;
}

void
stacktracepcsp(uvlong pc, uvlong sp)
{
	nextpc = 0;
	if(machdata->ctrace==nil)
		fprint(2, "no machdata->ctrace\n");
	else if(machdata->ctrace(map, pc, sp, 0, ptrace) <= 0)
		fprint(2, "no stack frame: pc=%#p sp=%#p\n", pc, sp);
}

void
addtohistogram(uvlong pc, uvlong sp)
{
	int h;
	PC *x;
	
	h = pc % Ncounters;
	for(x = counters[h]; x != NULL; x = x->next) {
		if(x->pc == pc) {
			x->count++;
			return;
		}
	}
	x = malloc(sizeof(PC));
	x->pc = pc;
	x->count = 1;
	x->next = counters[h];
	counters[h] = x;
}

void
printpc(uvlong pc, uvlong sp)
{
	char buf[1024];
	if(registers)
		regprint();
	if(have_syms > 0 && linenums &&  fileline(buf, sizeof buf, pc))
		print("%s\n", buf);
	if(have_syms > 0 && functions) {
		symoff(buf, sizeof(buf), pc, CANY);
		print("%s\n", buf);
	}
	if(stacks){
		stacktracepcsp(pc, sp);
	}
	if(histograms){
		addtohistogram(pc, sp);
	}
}

void samples()
{
	int msec;
	struct timespec req;

	req.tv_sec = delta_msec/1000;
	req.tv_nsec = 1000000*(delta_msec % 1000);
	for(msec = 0; msec < 1000*total_sec; msec += delta_msec) {
		if(!sample())
			break;
		printpc(ureg.ip, ureg.sp);
		nanosleep(&req, NULL);
	}
}

int
comparepc(const void *va, const void *vb)
{
	const PC *const*a = va;
	const PC *const*b = vb;
	return (*a)->pc - (*b)->pc;
}

int
comparecount(const void *va, const void *vb)
{
	const PC *const*a = va;
	const PC *const*b = vb;
	return (*b)->count - (*a)->count;  // sort downwards
}

void
func(char *s, int n, uvlong pc)
{
	char *p;

	symoff(s, n, pc, CANY);
	p = strchr(s, '+');
	if(p != NULL)
		*p = 0;
}

void
dumphistogram()
{
	int h;
	PC *x;
	PC **pcs;
	uint i;
	uint j;
	uint npc;
	uint ncount;
	char b1[100];
	char b2[100];

	if(!histograms)
		return;

	// count samples
	ncount = 0;
	npc = 0;
	for(h = 0; h < Ncounters; h++)
		for(x = counters[h]; x != NULL; x = x->next) {
			ncount += x->count;
			npc++;
		}
	// build array
	pcs = malloc(npc*sizeof(PC*));
	i = 0;
	for(h = 0; h < Ncounters; h++)
		for(x = counters[h]; x != NULL; x = x->next)
			pcs[i++] = x;
	if(collapse) {
		// combine counts in same function
		// sort by address
		qsort(pcs, npc, sizeof(PC*), comparepc);
		for(i = j = 0; i < npc; i++){
			x = pcs[i];
			func(b2, sizeof(b2), x->pc);
			if(j > 0 && strcmp(b1, b2) == 0) {
				pcs[i-1]->count += x->count;
			} else {
				strcpy(b1, b2);
				pcs[j++] = x;
			}
		}
		npc = j;
	}
	// sort by count
	qsort(pcs, npc, sizeof(PC*), comparecount);
	// print array
	for(i = 0; i < npc; i++){
		x = pcs[i];
		print("%5.2f%%\t", 100.0*(double)x->count/(double)ncount);
		if(collapse)
			func(b2, sizeof b2, x->pc);
		else
			symoff(b2, sizeof(b2), x->pc, CANY);
		print("%s\n", b2);
	}
}

int
main(int argc, char *argv[])
{
	ARGBEGIN{
	case 'c':
		collapse = 0;
		break;
	case 'd':
		delta_msec = atoi(EARGF(Usage));
		break;
	case 't':
		total_sec = atoi(EARGF(Usage));
		break;
	case 'p':
		pid = atoi(EARGF(Usage));
		break;
	case 'f':
		functions = 1;
		break;
	case 'h':
		histograms = 1;
	case 'l':
		linenums = 1;
		break;
	case 'r':
		registers = 1;
		break;
	case 's':
		stacks = 1;
		break;
	}ARGEND
	if(pid <= 0)
		Usage();
	if(functions+linenums+registers+stacks == 0)
		histograms = 1;
	if(!machbyname("amd64")) {
		fprint(2, "prof: no amd64 support\n", pid);
		exit(1);
	}
	if(argc > 0)
		file = argv[0];
	fd = open(file, 0);
	if(fd < 0) {
		fprint(2, "prof: can't open %s: %r\n", file);
		exit(1);
	}
	map = attachproc(pid, &fhdr);
	if(map == nil) {
		fprint(2, "prof: can't attach to %d: %r\n", pid);
		exit(1);
	}
	if(crackhdr(fd, &fhdr)) {
		have_syms = syminit(fd, &fhdr);
		if(!have_syms) {
			fprint(2, "prof: no symbols for %s: %r\n", file);
		}
	} else {
		fprint(2, "prof: crack header for %s: %r\n", file);
		exit(1);
	}
	samples();
	detachproc(map);
	dumphistogram();
	exit(0);
}
