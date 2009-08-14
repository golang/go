// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * code coverage
 */

#include <u.h>
#include <time.h>
#include <libc.h>
#include <bio.h>
#include <ctype.h>
#include <regexp9.h>
#include "tree.h"

#include <ureg_amd64.h>
#include <mach.h>
typedef struct Ureg Ureg;

void
usage(void)
{
	fprint(2, "usage: cov [-lsv] [-g regexp] [-m minlines] [6.out args...]\n");
	fprint(2, "-g specifies pattern of interesting functions or files\n");
	exits("usage");
}

typedef struct Range Range;
struct Range
{
	uvlong pc;
	uvlong epc;
};

int chatty;
int fd;
int longnames;
int pid;
int doshowsrc;
Map *mem;
Map *text;
Fhdr fhdr;
Reprog *grep;
char cwd[1000];
int ncwd;
int minlines = -1000;

Tree breakpoints;	// code ranges not run

/*
 * comparison for Range structures
 * they are "equal" if they overlap, so
 * that a search for [pc, pc+1) finds the
 * Range containing pc.
 */
int
rangecmp(void *va, void *vb)
{
	Range *a = va, *b = vb;
	if(a->epc <= b->pc)
		return 1;
	if(b->epc <= a->pc)
		return -1;
	return 0;
}

/*
 * remember that we ran the section of code [pc, epc).
 */
void
ran(uvlong pc, uvlong epc)
{
	Range key;
	Range *r;
	uvlong oldepc;

	if(chatty)
		print("run %#llux-%#llux\n", pc, epc);

	key.pc = pc;
	key.epc = pc+1;
	r = treeget(&breakpoints, &key);
	if(r == nil)
		sysfatal("unchecked breakpoint at %#lux+%d", pc, (int)(epc-pc));

	// Might be that the tail of the sequence
	// was run already, so r->epc is before the end.
	// Adjust len.
	if(epc > r->epc)
		epc = r->epc;

	if(r->pc == pc) {
		r->pc = epc;
	} else {
		// Chop r to before pc;
		// add new entry for after if needed.
		// Changing r->epc does not affect r's position in the tree.
		oldepc = r->epc;
		r->epc = pc;
		if(epc < oldepc) {
			Range *n;
			n = malloc(sizeof *n);
			n->pc = epc;
			n->epc = oldepc;
			treeput(&breakpoints, n, n);
		}
	}
}

void
showsrc(char *file, int line1, int line2)
{
	Biobuf *b;
	char *p;
	int n, stop;

	if((b = Bopen(file, OREAD)) == nil) {
		print("\topen %s: %r\n", file);
		return;
	}

	for(n=1; n<line1 && (p = Brdstr(b, '\n', 1)) != nil; n++)
		free(p);

	// print up to five lines (this one and 4 more).
	// if there are more than five lines, print 4 and "..."
	stop = n+4;
	if(stop > line2)
		stop = line2;
	if(stop < line2)
		stop--;
	for(; n<=stop && (p = Brdstr(b, '\n', 1)) != nil; n++) {
		print("  %d %s\n", n, p);
		free(p);
	}
	if(n < line2)
		print("  ...\n");
	Bterm(b);
}

/*
 * if s is in the current directory or below,
 * return the relative path.
 */
char*
shortname(char *s)
{
	if(!longnames && strlen(s) > ncwd && memcmp(s, cwd, ncwd) == 0 && s[ncwd] == '/')
		return s+ncwd+1;
	return s;
}

/*
 * we've decided that [pc, epc) did not run.
 * do something about it.
 */
void
missing(uvlong pc, uvlong epc)
{
	char file[1000];
	int line1, line2;
	char buf[100];
	Symbol s;
	char *p;
	uvlong uv;

	if(!findsym(pc, CTEXT, &s) || !fileline(file, sizeof file, pc)) {
	notfound:
		print("%#llux-%#llux\n", pc, epc);
		return;
	}
	p = strrchr(file, ':');
	*p++ = 0;
	line1 = atoi(p);
	for(uv=pc; uv<epc; ) {
		if(!fileline(file, sizeof file, epc-2))
			goto notfound;
		uv += machdata->instsize(text, uv);
	}
	p = strrchr(file, ':');
	*p++ = 0;
	line2 = atoi(p);

	if(line2+1-line2 < minlines)
		return;

	if(pc == s.value) {
		// never entered function
		print("%s:%d %s never called (%#llux-%#llux)\n", shortname(file), line1, s.name, pc, epc);
		return;
	}
	if(pc <= s.value+13) {
		// probably stub for stack growth.
		// check whether last instruction is call to morestack.
		// the -5 below is the length of
		//	CALL sys.morestack.
		buf[0] = 0;
		machdata->das(text, epc-5, 0, buf, sizeof buf);
		if(strstr(buf, "morestack"))
			return;
	}

	if(epc - pc == 5) {
		// check for CALL sys.throwindex
		buf[0] = 0;
		machdata->das(text, pc, 0, buf, sizeof buf);
		if(strstr(buf, "throwindex"))
			return;
	}

	if(epc - pc == 2 || epc -pc == 3) {
		// check for XORL inside shift.
		// (on x86 have to implement large left or unsigned right shift with explicit zeroing).
		//	f+90 0x00002c9f	CMPL	CX,$20
		//	f+93 0x00002ca2	JCS	f+97(SB)
		//	f+95 0x00002ca4	XORL	AX,AX <<<
		//	f+97 0x00002ca6	SHLL	CL,AX
		//	f+99 0x00002ca8	MOVL	$1,CX
		//
		//	f+c8 0x00002cd7	CMPL	CX,$40
		//	f+cb 0x00002cda	JCS	f+d0(SB)
		//	f+cd 0x00002cdc	XORQ	AX,AX <<<
		//	f+d0 0x00002cdf	SHLQ	CL,AX
		//	f+d3 0x00002ce2	MOVQ	$1,CX
		buf[0] = 0;
		machdata->das(text, pc, 0, buf, sizeof buf);
		if(strncmp(buf, "XOR", 3) == 0) {
			machdata->das(text, epc, 0, buf, sizeof buf);
			if(strncmp(buf, "SHL", 3) == 0 || strncmp(buf, "SHR", 3) == 0)
				return;
		}
	}

	if(epc - pc == 3) {
		// check for SAR inside shift.
		// (on x86 have to implement large signed right shift as >>31).
		//	f+36 0x00016216	CMPL	CX,$20
		//	f+39 0x00016219	JCS	f+3e(SB)
		//	f+3b 0x0001621b	SARL	$1f,AX <<<
		//	f+3e 0x0001621e	SARL	CL,AX
		//	f+40 0x00016220	XORL	CX,CX
		//	f+42 0x00016222	CMPL	CX,AX
		buf[0] = 0;
		machdata->das(text, pc, 0, buf, sizeof buf);
		if(strncmp(buf, "SAR", 3) == 0) {
			machdata->das(text, epc, 0, buf, sizeof buf);
			if(strncmp(buf, "SAR", 3) == 0)
				return;
		}
	}

	// show first instruction to make clear where we were.
	machdata->das(text, pc, 0, buf, sizeof buf);

	if(line1 != line2)
		print("%s:%d,%d %#llux-%#llux %s\n",
			shortname(file), line1, line2, pc, epc, buf);
	else
		print("%s:%d %#llux-%#llux %s\n",
			shortname(file), line1, pc, epc, buf);
	if(doshowsrc)
		showsrc(file, line1, line2);
}

/*
 * walk the tree, calling missing for each non-empty
 * section of missing code.
 */
void
walktree(TreeNode *t)
{
	Range *n;

	if(t == nil)
		return;
	walktree(t->left);
	n = t->key;
	if(n->pc < n->epc)
		missing(n->pc, n->epc);
	walktree(t->right);
}

/*
 * set a breakpoint all over [pc, epc)
 * and remember that we did.
 */
void
breakpoint(uvlong pc, uvlong epc)
{
	Range *r;

	r = malloc(sizeof *r);
	r->pc = pc;
	r->epc = epc;
	treeput(&breakpoints, r, r);

	for(; pc < epc; pc+=machdata->bpsize)
		put1(mem, pc, machdata->bpinst, machdata->bpsize);
}

/*
 * install breakpoints over all text symbols
 * that match the pattern.
 */
void
cover(void)
{
	Symbol s;
	char *lastfn;
	uvlong lastpc;
	int i;
	char buf[200];

	lastfn = nil;
	lastpc = 0;
	for(i=0; textsym(&s, i); i++) {
		switch(s.type) {
		case 'T':
		case 't':
			if(lastpc != 0) {
				breakpoint(lastpc, s.value);
				lastpc = 0;
			}
			// Ignore second entry for a given name;
			// that's the debugging blob.
			if(lastfn && strcmp(s.name, lastfn) == 0)
				break;
			lastfn = s.name;
			buf[0] = 0;
			fileline(buf, sizeof buf, s.value);
			if(grep == nil || regexec9(grep, buf, nil, 0) > 0 || regexec9(grep, s.name, nil, 0) > 0)
				lastpc = s.value;
		}
	}
}

uvlong
rgetzero(Map *map, char *reg)
{
	return 0;
}

/*
 * remove the breakpoints at pc and successive instructions,
 * up to and including the first jump or other control flow transfer.
 */
void
uncover(uvlong pc)
{
	uchar buf[1000];
	int n, n1, n2;
	uvlong foll[2];

	// Double-check that we stopped at a breakpoint.
	if(get1(mem, pc, buf, machdata->bpsize) < 0)
		sysfatal("read mem inst at %#llux: %r", pc);
	if(memcmp(buf, machdata->bpinst, machdata->bpsize) != 0)
		sysfatal("stopped at %#llux; not at breakpoint %d", pc, machdata->bpsize);

	// Figure out how many bytes of straight-line code
	// there are in the text starting at pc.
	n = 0;
	while(n < sizeof buf) {
		n1 = machdata->instsize(text, pc+n);
		if(n+n1 > sizeof buf)
			break;
		n2 = machdata->foll(text, pc+n, rgetzero, foll);
		n += n1;
		if(n2 != 1 || foll[0] != pc+n)
			break;
	}

	// Record that this section of code ran.
	ran(pc, pc+n);

	// Put original instructions back.
	if(get1(text, pc, buf, n) < 0)
		sysfatal("get1: %r");
	if(put1(mem, pc, buf, n) < 0)
		sysfatal("put1: %r");
}

int
startprocess(char **argv)
{
	int pid;

	if((pid = fork()) < 0)
		sysfatal("fork: %r");
	if(pid == 0) {
		pid = getpid();
		if(ctlproc(pid, "hang") < 0)
			sysfatal("ctlproc hang: %r");
		execv(argv[0], argv);
		sysfatal("exec %s: %r", argv[0]);
	}
	if(ctlproc(pid, "attached") < 0 || ctlproc(pid, "waitstop") < 0)
		sysfatal("attach %d %s: %r", pid, argv[0]);
	return pid;
}

int
go(void)
{
	uvlong pc;
	char buf[100];
	int n;

	for(n = 0;; n++) {
		ctlproc(pid, "startstop");
		if(get8(mem, offsetof(Ureg, ip), &pc) < 0) {
			rerrstr(buf, sizeof buf);
			if(strstr(buf, "exited") || strstr(buf, "No such process"))
				return n;
			sysfatal("cannot read pc: %r");
		}
		pc--;
		if(put8(mem, offsetof(Ureg, ip), pc) < 0)
			sysfatal("cannot write pc: %r");
		uncover(pc);
	}
}

void
main(int argc, char **argv)
{
	int n;
	char *regexp;

	ARGBEGIN{
	case 'g':
		regexp = EARGF(usage());
		if((grep = regcomp9(regexp)) == nil)
			sysfatal("bad regexp %s", regexp);
		break;
	case 'l':
		longnames++;
		break;
	case 'n':
		minlines = atoi(EARGF(usage()));
		break;
	case 's':
		doshowsrc = 1;
		break;
	case 'v':
		chatty++;
		break;
	default:
		usage();
	}ARGEND

	getwd(cwd, sizeof cwd);
	ncwd = strlen(cwd);

	if(argc == 0) {
		*--argv = "6.out";
		argc++;
	}
	fd = open(argv[0], OREAD);
	if(fd < 0)
		sysfatal("open %s: %r", argv[0]);
	if(crackhdr(fd, &fhdr) <= 0)
		sysfatal("crackhdr: %r");
	machbytype(fhdr.type);
	if(syminit(fd, &fhdr) <= 0)
		sysfatal("syminit: %r");
	text = loadmap(nil, fd, &fhdr);
	if(text == nil)
		sysfatal("loadmap: %r");
	pid = startprocess(argv);
	mem = attachproc(pid, &fhdr);
	if(mem == nil)
		sysfatal("attachproc: %r");
	breakpoints.cmp = rangecmp;
	cover();
	n = go();
	walktree(breakpoints.root);
	if(chatty)
		print("%d breakpoints\n", n);
	detachproc(mem);
	exits(0);
}

