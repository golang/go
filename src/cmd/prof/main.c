// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <time.h>
#include <libc.h>
#include <bio.h>
#include <ctype.h>

#define Ureg Ureg_amd64
	#include <ureg_amd64.h>
#undef Ureg
#define Ureg Ureg_x86
	#include <ureg_x86.h>
#undef Ureg
#include <mach.h>

char* file = "6.out";
static Fhdr fhdr;
int have_syms;
int fd;
Map	*symmap;
struct Ureg_amd64 ureg_amd64;
struct Ureg_x86 ureg_x86;
int total_sec = 0;
int delta_msec = 100;
int nsample;
int nsamplethread;

// output formats
int functions;	// print functions
int histograms;	// print histograms
int linenums;	// print file and line numbers rather than function names
int registers;	// print registers
int stacks;		// print stack traces

int pid;		// main process pid

int nthread;	// number of threads
int thread[32];	// thread pids
Map *map[32];	// thread maps

void
Usage(void)
{
	fprint(2, "Usage: prof -p pid [-t total_secs] [-d delta_msec] [6.out args ...]\n");
	fprint(2, "\tformats (default -h):\n");
	fprint(2, "\t\t-h: histograms\n");
	fprint(2, "\t\t-f: dynamic functions\n");
	fprint(2, "\t\t-l: dynamic file and line numbers\n");
	fprint(2, "\t\t-r: dynamic registers\n");
	fprint(2, "\t\t-s: dynamic function stack traces\n");
	fprint(2, "\t\t-hs: include stack info in histograms\n");
	exit(2);
}

typedef struct PC PC;
struct PC {
	uvlong pc;
	uvlong callerpc;
	unsigned int count;
	PC* next;
};

enum {
	Ncounters = 256
};

PC *counters[Ncounters];

// Set up by setarch() to make most of the code architecture-independent.
typedef struct Arch Arch;
struct Arch {
	char*	name;
	void	(*regprint)(void);
	int	(*getregs)(Map*);
	int	(*getPC)(Map*);
	int	(*getSP)(Map*);
	uvlong	(*uregPC)(void);
	uvlong	(*uregSP)(void);
};

void
amd64_regprint(void)
{
	fprint(2, "ax\t0x%llux\n", ureg_amd64.ax);
	fprint(2, "bx\t0x%llux\n", ureg_amd64.bx);
	fprint(2, "cx\t0x%llux\n", ureg_amd64.cx);
	fprint(2, "dx\t0x%llux\n", ureg_amd64.dx);
	fprint(2, "si\t0x%llux\n", ureg_amd64.si);
	fprint(2, "di\t0x%llux\n", ureg_amd64.di);
	fprint(2, "bp\t0x%llux\n", ureg_amd64.bp);
	fprint(2, "r8\t0x%llux\n", ureg_amd64.r8);
	fprint(2, "r9\t0x%llux\n", ureg_amd64.r9);
	fprint(2, "r10\t0x%llux\n", ureg_amd64.r10);
	fprint(2, "r11\t0x%llux\n", ureg_amd64.r11);
	fprint(2, "r12\t0x%llux\n", ureg_amd64.r12);
	fprint(2, "r13\t0x%llux\n", ureg_amd64.r13);
	fprint(2, "r14\t0x%llux\n", ureg_amd64.r14);
	fprint(2, "r15\t0x%llux\n", ureg_amd64.r15);
	fprint(2, "ds\t0x%llux\n", ureg_amd64.ds);
	fprint(2, "es\t0x%llux\n", ureg_amd64.es);
	fprint(2, "fs\t0x%llux\n", ureg_amd64.fs);
	fprint(2, "gs\t0x%llux\n", ureg_amd64.gs);
	fprint(2, "type\t0x%llux\n", ureg_amd64.type);
	fprint(2, "error\t0x%llux\n", ureg_amd64.error);
	fprint(2, "pc\t0x%llux\n", ureg_amd64.ip);
	fprint(2, "cs\t0x%llux\n", ureg_amd64.cs);
	fprint(2, "flags\t0x%llux\n", ureg_amd64.flags);
	fprint(2, "sp\t0x%llux\n", ureg_amd64.sp);
	fprint(2, "ss\t0x%llux\n", ureg_amd64.ss);
}

int
amd64_getregs(Map *map)
{
	int i;

	for(i = 0; i < sizeof ureg_amd64; i+=8) {
		if(get8(map, (uvlong)i, &((uvlong*)&ureg_amd64)[i/4]) < 0)
		return -1;
	}
	return 0;
}

int
amd64_getPC(Map *map)
{
	return get8(map, offsetof(struct Ureg_amd64, ip), (uvlong*)&ureg_amd64.ip);
}

int
amd64_getSP(Map *map)
{
	return get8(map, offsetof(struct Ureg_amd64, sp), (uvlong*)&ureg_amd64.sp);
}

uvlong
amd64_uregPC(void)
{
	return ureg_amd64.ip;
}

uvlong
amd64_uregSP(void) {
	return ureg_amd64.sp;
}

void
x86_regprint(void)
{
	fprint(2, "ax\t0x%llux\n", ureg_x86.ax);
	fprint(2, "bx\t0x%llux\n", ureg_x86.bx);
	fprint(2, "cx\t0x%llux\n", ureg_x86.cx);
	fprint(2, "dx\t0x%llux\n", ureg_x86.dx);
	fprint(2, "si\t0x%llux\n", ureg_x86.si);
	fprint(2, "di\t0x%llux\n", ureg_x86.di);
	fprint(2, "bp\t0x%llux\n", ureg_x86.bp);
	fprint(2, "ds\t0x%llux\n", ureg_x86.ds);
	fprint(2, "es\t0x%llux\n", ureg_x86.es);
	fprint(2, "fs\t0x%llux\n", ureg_x86.fs);
	fprint(2, "gs\t0x%llux\n", ureg_x86.gs);
	fprint(2, "cs\t0x%llux\n", ureg_x86.cs);
	fprint(2, "flags\t0x%llux\n", ureg_x86.flags);
	fprint(2, "pc\t0x%llux\n", ureg_x86.pc);
	fprint(2, "sp\t0x%llux\n", ureg_x86.sp);
	fprint(2, "ss\t0x%llux\n", ureg_x86.ss);
}

int
x86_getregs(Map *map)
{
	int i;

	for(i = 0; i < sizeof ureg_x86; i+=4) {
		if(get4(map, (uvlong)i, &((uint32*)&ureg_x86)[i/4]) < 0)
		return -1;
	}
	return 0;
}

int
x86_getPC(Map* map)
{
	return get4(map, offsetof(struct Ureg_x86, pc), &ureg_x86.pc);
}

int
x86_getSP(Map* map)
{
	return get4(map, offsetof(struct Ureg_x86, sp), &ureg_x86.sp);
}

uvlong
x86_uregPC(void)
{
	return (uvlong)ureg_x86.pc;
}

uvlong
x86_uregSP(void)
{
	return (uvlong)ureg_x86.sp;
}

Arch archtab[] = {
	{
		"amd64",
		amd64_regprint,
		amd64_getregs,
		amd64_getPC,
		amd64_getSP,
		amd64_uregPC,
		amd64_uregSP,
	},
	{
		"386",
		x86_regprint,
		x86_getregs,
		x86_getPC,
		x86_getSP,
		x86_uregPC,
		x86_uregSP,
	},
	{
		nil
	}
};

Arch *arch;

int
setarch(void)
{
	int i;

	if(mach != nil) {
		for(i = 0; archtab[i].name != nil; i++) {
			if (strcmp(mach->name, archtab[i].name) == 0) {
				arch = &archtab[i];
				return 0;
			}
		}
	}
	return -1;
}

int
getthreads(void)
{
	int i, j, curn, found;
	Map *curmap[nelem(map)];
	int curthread[nelem(map)];
	static int complained = 0;

	curn = procthreadpids(pid, curthread, nelem(curthread));
	if(curn <= 0)
		return curn;

	if(curn > nelem(map)) {
		if(complained == 0) {
			fprint(2, "prof: too many threads; limiting to %d\n", nthread, nelem(map));
			complained = 1;
		}
		curn = nelem(map);
	}
	if(curn == nthread && memcmp(thread, curthread, curn*sizeof(*thread)) == 0)
		return curn;	// no changes

	// Number of threads has changed (might be the init case).
	// A bit expensive but rare enough not to bother being clever.
	for(i = 0; i < curn; i++) {
		found = 0;
		for(j = 0; j < nthread; j++) {
			if(curthread[i] == thread[j]) {
				found = 1;
				curmap[i] = map[j];
				map[j] = nil;
				break;
			}
		}
		if(found)
			continue;

		// map new thread
		curmap[i] = attachproc(curthread[i], &fhdr);
		if(curmap[i] == nil) {
			fprint(2, "prof: can't attach to %d: %r\n", curthread[i]);
			return -1;
		}
	}

	for(j = 0; j < nthread; j++)
		if(map[j] != nil)
			detachproc(map[j]);

	nthread = curn;
	memmove(thread, curthread, nthread*sizeof thread[0]);
	memmove(map, curmap, sizeof map);
	return nthread;
}

int
sample(Map *map)
{
	static int n;

	n++;
	if(registers) {
		if(arch->getregs(map) < 0)
			goto bad;
	} else {
		// we need only two registers
		if(arch->getPC(map) < 0)
			goto bad;
		if(arch->getSP(map) < 0)
			goto bad;
	}
	return 1;
bad:
	if(n == 1)
		fprint(2, "prof: can't read registers: %r\n");
	return 0;
}

void
addtohistogram(uvlong pc, uvlong callerpc, uvlong sp)
{
	int h;
	PC *x;

	h = (pc + callerpc*101) % Ncounters;
	for(x = counters[h]; x != NULL; x = x->next) {
		if(x->pc == pc && x->callerpc == callerpc) {
			x->count++;
			return;
		}
	}
	x = malloc(sizeof(PC));
	x->pc = pc;
	x->callerpc = callerpc;
	x->count = 1;
	x->next = counters[h];
	counters[h] = x;
}

uvlong nextpc;

void
xptrace(Map *map, uvlong pc, uvlong sp, Symbol *sym)
{
	char buf[1024];
	if(sym == nil){
		fprint(2, "syms\n");
		return;
	}
	if(histograms)
		addtohistogram(nextpc, pc, sp);
	if(!histograms || stacks > 1) {
		if(nextpc == 0)
			nextpc = sym->value;
		fprint(2, "%s(", sym->name);
		fprint(2, ")");
		if(nextpc != sym->value)
			fprint(2, "+%#llux ", nextpc - sym->value);
		if(have_syms && linenums && fileline(buf, sizeof buf, pc)) {
			fprint(2, " %s", buf);
		}
		fprint(2, "\n");
	}
	nextpc = pc;
}

void
stacktracepcsp(Map *map, uvlong pc, uvlong sp)
{
	nextpc = pc;
	if(machdata->ctrace==nil)
		fprint(2, "no machdata->ctrace\n");
	else if(machdata->ctrace(map, pc, sp, 0, xptrace) <= 0)
		fprint(2, "no stack frame: pc=%#p sp=%#p\n", pc, sp);
	else {
		addtohistogram(nextpc, 0, sp);
		if(!histograms || stacks > 1)
			fprint(2, "\n");
	}
}

void
printpc(Map *map, uvlong pc, uvlong sp)
{
	char buf[1024];
	if(registers)
		arch->regprint();
	if(have_syms > 0 && linenums &&  fileline(buf, sizeof buf, pc))
		fprint(2, "%s\n", buf);
	if(have_syms > 0 && functions) {
		symoff(buf, sizeof(buf), pc, CANY);
		fprint(2, "%s\n", buf);
	}
	if(stacks){
		stacktracepcsp(map, pc, sp);
	}
	else if(histograms){
		addtohistogram(pc, 0, sp);
	}
}

void
samples(void)
{
	int i, pid, msec;
	struct timespec req;

	req.tv_sec = delta_msec/1000;
	req.tv_nsec = 1000000*(delta_msec % 1000);
	for(msec = 0; total_sec <= 0 || msec < 1000*total_sec; msec += delta_msec) {
		nsample++;
		nsamplethread += nthread;
		for(i = 0; i < nthread; i++) {
			pid = thread[i];
			if(ctlproc(pid, "stop") < 0)
				return;
			if(!sample(map[i])) {
				ctlproc(pid, "start");
				return;
			}
			printpc(map[i], arch->uregPC(), arch->uregSP());
			ctlproc(pid, "start");
		}
		nanosleep(&req, NULL);
		getthreads();
		if(nthread == 0)
			break;
	}
}

typedef struct Func Func;
struct Func
{
	Func *next;
	Symbol s;
	uint onstack;
	uint leaf;
};

Func *func[257];
int nfunc;

Func*
findfunc(uvlong pc)
{
	Func *f;
	uint h;
	Symbol s;

	if(pc == 0)
		return nil;

	if(!findsym(pc, CTEXT, &s))
		return nil;

	h = s.value % nelem(func);
	for(f = func[h]; f != NULL; f = f->next)
		if(f->s.value == s.value)
			return f;

	f = malloc(sizeof *f);
	memset(f, 0, sizeof *f);
	f->s = s;
	f->next = func[h];
	func[h] = f;
	nfunc++;
	return f;
}

int
compareleaf(const void *va, const void *vb)
{
	Func *a, *b;

	a = *(Func**)va;
	b = *(Func**)vb;
	if(a->leaf != b->leaf)
		return b->leaf - a->leaf;
	if(a->onstack != b->onstack)
		return b->onstack - a->onstack;
	return strcmp(a->s.name, b->s.name);
}

void
dumphistogram()
{
	int i, h, n;
	PC *x;
	Func *f, **ff;

	if(!histograms)
		return;

	// assign counts to functions.
	for(h = 0; h < Ncounters; h++) {
		for(x = counters[h]; x != NULL; x = x->next) {
			f = findfunc(x->pc);
			if(f) {
				f->onstack += x->count;
				f->leaf += x->count;
			}
			f = findfunc(x->callerpc);
			if(f)
				f->leaf -= x->count;
		}
	}

	// build array
	ff = malloc(nfunc*sizeof ff[0]);
	n = 0;
	for(h = 0; h < nelem(func); h++)
		for(f = func[h]; f != NULL; f = f->next)
			ff[n++] = f;

	// sort by leaf counts
	qsort(ff, nfunc, sizeof ff[0], compareleaf);

	// print.
	fprint(2, "%d samples (avg %.1g threads)\n", nsample, (double)nsamplethread/nsample);
	for(i = 0; i < nfunc; i++) {
		f = ff[i];
		fprint(2, "%6.2f%%\t", 100.0*(double)f->leaf/nsample);
		if(stacks)
			fprint(2, "%6.2f%%\t", 100.0*(double)f->onstack/nsample);
		fprint(2, "%s\n", f->s.name);
	}
}

int
startprocess(char **argv)
{
	int pid;

	if((pid = fork()) == 0) {
		pid = getpid();
		if(ctlproc(pid, "hang") < 0){
			fprint(2, "prof: child process could not hang\n");
			exits(0);
		}
		execv(argv[0], argv);
		fprint(2, "prof: could not exec %s: %r\n", argv[0]);
		exits(0);
	}

	if(pid == -1) {
		fprint(2, "prof: could not fork\n");
		exit(1);
	}
	if(ctlproc(pid, "attached") < 0 || ctlproc(pid, "waitstop") < 0) {
		fprint(2, "prof: could not attach to child process: %r\n");
		exit(1);
	}
	return pid;
}

void
detach(void)
{
	int i;

	for(i = 0; i < nthread; i++)
		detachproc(map[i]);
}

int
main(int argc, char *argv[])
{
	int i;

	ARGBEGIN{
	case 'd':
		delta_msec = atoi(EARGF(Usage()));
		break;
	case 't':
		total_sec = atoi(EARGF(Usage()));
		break;
	case 'p':
		pid = atoi(EARGF(Usage()));
		break;
	case 'f':
		functions = 1;
		break;
	case 'h':
		histograms = 1;
		break;
	case 'l':
		linenums = 1;
		break;
	case 'r':
		registers = 1;
		break;
	case 's':
		stacks++;
		break;
	}ARGEND
	if(pid <= 0 && argc == 0)
		Usage();
	if(functions+linenums+registers+stacks == 0)
		histograms = 1;
	if(!machbyname("amd64")) {
		fprint(2, "prof: no amd64 support\n", pid);
		exit(1);
	}
	if(argc > 0)
		file = argv[0];
	else if(pid) {
		file = proctextfile(pid);
		if (file == NULL) {
			fprint(2, "prof: can't find file for pid %d: %r\n", pid);
			fprint(2, "prof: on Darwin, need to provide file name explicitly\n");
			exit(1);
		}
	}
	fd = open(file, 0);
	if(fd < 0) {
		fprint(2, "prof: can't open %s: %r\n", file);
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
	if(pid <= 0)
		pid = startprocess(argv);
	attachproc(pid, &fhdr);	// initializes thread list
	if(setarch() < 0) {
		detach();
		fprint(2, "prof: can't identify binary architecture for pid %d\n", pid);
		exit(1);
	}
	if(getthreads() <= 0) {
		detach();
		fprint(2, "prof: can't find threads for pid %d\n", pid);
		exit(1);
	}
	for(i = 0; i < nthread; i++)
		ctlproc(thread[i], "start");
	samples();
	detach();
	dumphistogram();
	exit(0);
}
