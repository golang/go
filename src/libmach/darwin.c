//	Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define __DARWIN_UNIX03 0

#include <u.h>
#include <sys/ptrace.h>
#include <sys/signal.h>
#include <mach/mach.h>
#include <mach/mach_traps.h>
#include <errno.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>
#define Ureg Ureg32
#include <ureg_x86.h>
#undef Ureg
#define Ureg Ureg64
#include <ureg_amd64.h>
#undef Ureg
#undef waitpid	/* want Unix waitpid, not Plan 9 */

typedef struct Ureg32 Ureg32;
typedef struct Ureg64 Ureg64;

extern mach_port_t mach_reply_port(void);	// should be in system headers, is not

// Mach-error wrapper.
// Takes a mach return code and converts it into 0 / -1,
// setting errstr when it returns -1.

static struct {
	int code;
	char *name;
} macherr[] = {
	KERN_INVALID_ADDRESS,	"invalid address",
	KERN_PROTECTION_FAILURE,	"protection failure",
	KERN_NO_SPACE,	"no space",
	KERN_INVALID_ARGUMENT,	"invalid argument",
	KERN_FAILURE,	"failure",
	KERN_RESOURCE_SHORTAGE,	"resource shortage",
	KERN_NOT_RECEIVER,	"not receiver",
	KERN_NO_ACCESS,	"no access",
	KERN_MEMORY_FAILURE,	"memory failure",
	KERN_MEMORY_ERROR,	"memory error",
	KERN_ALREADY_IN_SET,	"already in set",
	KERN_NOT_IN_SET,	"not in set",
	KERN_NAME_EXISTS,	"name exists",
	KERN_ABORTED,	"aborted",
	KERN_INVALID_NAME,	"invalid name",
	KERN_INVALID_TASK,	"invalid task",
	KERN_INVALID_RIGHT,	"invalid right",
	KERN_INVALID_VALUE,	"invalid value",
	KERN_UREFS_OVERFLOW,	"urefs overflow",
	KERN_INVALID_CAPABILITY,	"invalid capability",
	KERN_RIGHT_EXISTS,	"right exists",
	KERN_INVALID_HOST,	"invalid host",
	KERN_MEMORY_PRESENT,	"memory present",
	KERN_MEMORY_DATA_MOVED,	"memory data moved",
	KERN_MEMORY_RESTART_COPY,	"memory restart copy",
	KERN_INVALID_PROCESSOR_SET,	"invalid processor set",
	KERN_POLICY_LIMIT,	"policy limit",
	KERN_INVALID_POLICY,	"invalid policy",
	KERN_INVALID_OBJECT,	"invalid object",
	KERN_ALREADY_WAITING,	"already waiting",
	KERN_DEFAULT_SET,	"default set",
	KERN_EXCEPTION_PROTECTED,	"exception protected",
	KERN_INVALID_LEDGER,	"invalid ledger",
	KERN_INVALID_MEMORY_CONTROL,	"invalid memory control",
	KERN_INVALID_SECURITY,	"invalid security",
	KERN_NOT_DEPRESSED,	"not depressed",
	KERN_TERMINATED,	"terminated",
	KERN_LOCK_SET_DESTROYED,	"lock set destroyed",
	KERN_LOCK_UNSTABLE,	"lock unstable",
	KERN_LOCK_OWNED,	"lock owned",
	KERN_LOCK_OWNED_SELF,	"lock owned self",
	KERN_SEMAPHORE_DESTROYED,	"semaphore destroyed",
	KERN_RPC_SERVER_TERMINATED,	"rpc server terminated",
	KERN_RPC_TERMINATE_ORPHAN,	"rpc terminate orphan",
	KERN_RPC_CONTINUE_ORPHAN,	"rpc continue orphan",
	KERN_NOT_SUPPORTED,	"not supported",
	KERN_NODE_DOWN,	"node down",
	KERN_NOT_WAITING,	"not waiting",
	KERN_OPERATION_TIMED_OUT,	"operation timed out",
	KERN_RETURN_MAX,	"return max",

	MACH_SEND_IN_PROGRESS,	"send in progress",
	MACH_SEND_INVALID_DATA,	"send invalid data",
	MACH_SEND_INVALID_DEST,	"send invalid dest",
	MACH_SEND_TIMED_OUT,	"send timed out",
	MACH_SEND_INTERRUPTED,	"send interrupted",
	MACH_SEND_MSG_TOO_SMALL,	"send msg too small",
	MACH_SEND_INVALID_REPLY,	"send invalid reply",
	MACH_SEND_INVALID_RIGHT,	"send invalid right",
	MACH_SEND_INVALID_NOTIFY,	"send invalid notify",
	MACH_SEND_INVALID_MEMORY,	"send invalid memory",
	MACH_SEND_NO_BUFFER,	"send no buffer",
	MACH_SEND_TOO_LARGE,	"send too large",
	MACH_SEND_INVALID_TYPE,	"send invalid type",
	MACH_SEND_INVALID_HEADER,	"send invalid header",
	MACH_SEND_INVALID_TRAILER,	"send invalid trailer",
	MACH_SEND_INVALID_RT_OOL_SIZE,	"send invalid rt ool size",
	MACH_RCV_IN_PROGRESS,	"rcv in progress",
	MACH_RCV_INVALID_NAME,	"rcv invalid name",
	MACH_RCV_TIMED_OUT,	"rcv timed out",
	MACH_RCV_TOO_LARGE,	"rcv too large",
	MACH_RCV_INTERRUPTED,	"rcv interrupted",
	MACH_RCV_PORT_CHANGED,	"rcv port changed",
	MACH_RCV_INVALID_NOTIFY,	"rcv invalid notify",
	MACH_RCV_INVALID_DATA,	"rcv invalid data",
	MACH_RCV_PORT_DIED,	"rcv port died",
	MACH_RCV_IN_SET,	"rcv in set",
	MACH_RCV_HEADER_ERROR,	"rcv header error",
	MACH_RCV_BODY_ERROR,	"rcv body error",
	MACH_RCV_INVALID_TYPE,	"rcv invalid type",
	MACH_RCV_SCATTER_SMALL,	"rcv scatter small",
	MACH_RCV_INVALID_TRAILER,	"rcv invalid trailer",
	MACH_RCV_IN_PROGRESS_TIMED,	"rcv in progress timed",

	MIG_TYPE_ERROR,	"mig type error",
	MIG_REPLY_MISMATCH,	"mig reply mismatch",
	MIG_REMOTE_ERROR,	"mig remote error",
	MIG_BAD_ID,	"mig bad id",
	MIG_BAD_ARGUMENTS,	"mig bad arguments",
	MIG_NO_REPLY,	"mig no reply",
	MIG_EXCEPTION,	"mig exception",
	MIG_ARRAY_TOO_LARGE,	"mig array too large",
	MIG_SERVER_DIED,	"server died",
	MIG_TRAILER_ERROR,	"trailer has an unknown format",
};

static int
me(kern_return_t r)
{
	int i;

	if(r == 0)
		return 0;

	for(i=0; i<nelem(macherr); i++){
		if(r == macherr[i].code){
			werrstr("mach: %s", macherr[i].name);
			return -1;
		}
	}
	werrstr("mach error %#x", r);
	return -1;
}

// Plan 9 and Linux do not distinguish between
// process ids and thread ids, so the interface here doesn't either.
// Unfortunately, Mach has three kinds of identifiers: process ids,
// handles to tasks (processes), and handles to threads within a
// process.  All of them are small integers.
//
// To accommodate Mach, we employ a clumsy hack: in this interface,
// if you pass in a positive number, that's a process id.
// If you pass in a negative number, that identifies a thread that
// has been previously returned by procthreadpids (it indexes
// into the Thread table below).

// Table of threads we have handles for.
typedef struct Thread Thread;
struct Thread
{
	int pid;
	mach_port_t task;
	mach_port_t thread;
	int stopped;
	int exc;
	int code[10];
	Map *map;
};
static Thread thr[1000];
static int nthr;
static pthread_mutex_t mu;
static pthread_cond_t cond;
static void* excthread(void*);
static void* waitthread(void*);
static mach_port_t excport;

enum {
	ExcMask = EXC_MASK_BAD_ACCESS |
		EXC_MASK_BAD_INSTRUCTION |
		EXC_MASK_ARITHMETIC |
		EXC_MASK_BREAKPOINT |
		EXC_MASK_SOFTWARE
};

// Add process pid to the thread table.
// If it's already there, don't re-add it (unless force != 0).
static Thread*
addpid(int pid, int force)
{
	int i, j;
	mach_port_t task;
	mach_port_t *thread;
	uint nthread;
	Thread *ret;
	static int first = 1;

	if(first){
		// Allocate a port for exception messages and
		// send all thread exceptions to that port.
		// The excthread reads that port and signals
		// us if we are waiting on that thread.
		pthread_t p;
		int err;

		excport = mach_reply_port();
		pthread_mutex_init(&mu, nil);
		pthread_cond_init(&cond, nil);
		err = pthread_create(&p, nil, excthread, nil);
		if (err != 0) {
			fprint(2, "pthread_create failed: %s\n", strerror(err));
			abort();
		}
		err = pthread_create(&p, nil, waitthread, (void*)(uintptr)pid);
		if (err != 0) {
			fprint(2, "pthread_create failed: %s\n", strerror(err));
			abort();
		}
		first = 0;
	}

	if(!force){
		for(i=0; i<nthr; i++)
			if(thr[i].pid == pid)
				return &thr[i];
	}
	if(me(task_for_pid(mach_task_self(), pid, &task)) < 0)
		return nil;
	if(me(task_threads(task, &thread, &nthread)) < 0)
		return nil;
	mach_port_insert_right(mach_task_self(), excport, excport, MACH_MSG_TYPE_MAKE_SEND);
	if(me(task_set_exception_ports(task, ExcMask,
			excport, EXCEPTION_DEFAULT, MACHINE_THREAD_STATE)) < 0){
		fprint(2, "warning: cannot set excport: %r\n");
	}
	ret = nil;
	for(j=0; j<nthread; j++){
		if(force){
			// If we're forcing a refresh, don't re-add existing threads.
			for(i=0; i<nthr; i++)
				if(thr[i].pid == pid && thr[i].thread == thread[j]){
					if(ret == nil)
						ret = &thr[i];
					goto skip;
				}
		}
		if(nthr >= nelem(thr))
			return nil;
		// TODO: We probably should save the old thread exception
		// ports for each bit and then put them back when we exit.
		// Probably the BSD signal handlers have put stuff there.
		mach_port_insert_right(mach_task_self(), excport, excport, MACH_MSG_TYPE_MAKE_SEND);
		if(me(thread_set_exception_ports(thread[j], ExcMask,
				excport, EXCEPTION_DEFAULT, MACHINE_THREAD_STATE)) < 0){
			fprint(2, "warning: cannot set excport: %r\n");
		}
		thr[nthr].pid = pid;
		thr[nthr].task = task;
		thr[nthr].thread = thread[j];
		if(ret == nil)
			ret = &thr[nthr];
		nthr++;
	skip:;
	}
	return ret;
}

static Thread*
idtotable(int id)
{
	if(id >= 0)
		return addpid(id, 1);

	id = -(id+1);
	if(id >= nthr)
		return nil;
	return &thr[id];
}

/*
static int
idtopid(int id)
{
	Thread *t;

	if((t = idtotable(id)) == nil)
		return -1;
	return t->pid;
}
*/

static mach_port_t
idtotask(int id)
{
	Thread *t;

	if((t = idtotable(id)) == nil)
		return -1;
	return t->task;
}

static mach_port_t
idtothread(int id)
{
	Thread *t;

	if((t = idtotable(id)) == nil)
		return -1;
	return t->thread;
}

static int machsegrw(Map *map, Seg *seg, uvlong addr, void *v, uint n, int isr);
static int machregrw(Map *map, Seg *seg, uvlong addr, void *v, uint n, int isr);

Map*
attachproc(int id, Fhdr *fp)
{
	Thread *t;
	Map *map;

	if((t = idtotable(id)) == nil)
		return nil;
	if(t->map)
		return t->map;
	map = newmap(0, 4);
	if(!map)
		return nil;
	map->pid = -((t-thr) + 1);
	if(mach->regsize)
		setmap(map, -1, 0, mach->regsize, 0, "regs", machregrw);
	setmap(map, -1, fp->txtaddr, fp->txtaddr+fp->txtsz, fp->txtaddr, "*text", machsegrw);
	setmap(map, -1, fp->dataddr, mach->utop, fp->dataddr, "*data", machsegrw);
	t->map = map;
	return map;
}

// Return list of ids for threads in id.
int
procthreadpids(int id, int *out, int nout)
{
	Thread *t;
	int i, n, pid;

	t = idtotable(id);
	if(t == nil)
		return -1;
	pid = t->pid;
	addpid(pid, 1);	// force refresh of thread list
	n = 0;
	for(i=0; i<nthr; i++) {
		if(thr[i].pid == pid) {
			if(n < nout)
				out[n] = -(i+1);
			n++;
		}
	}
	return n;
}

// Detach from proc.
// TODO(rsc): Perhaps should unsuspend any threads and clean-up the table.
void
detachproc(Map *m)
{
	free(m);
}

// Should return array of pending signals (notes)
// but don't know how to do that on OS X.
int
procnotes(int pid, char ***pnotes)
{
	*pnotes = 0;
	return 0;
}

// There must be a way to do this.  Gdb can do it.
// But I don't see, in the Apple gdb sources, how.
char*
proctextfile(int pid)
{
	return nil;
}

// Read/write from a Mach data segment.
static int
machsegrw(Map *map, Seg *seg, uvlong addr, void *v, uint n, int isr)
{
	mach_port_t task;
	int r;

	task = idtotask(map->pid);
	if(task == -1)
		return -1;

	if(isr){
		vm_size_t nn;
		nn = n;
		if(me(vm_read_overwrite(task, addr, n, (uintptr)v, &nn)) < 0) {
			fprint(2, "vm_read_overwrite %#llux %d to %p: %r\n", addr, n, v);
			return -1;
		}
		return nn;
	}else{
		r = vm_write(task, addr, (uintptr)v, n);
		if(r == KERN_INVALID_ADDRESS){
			// Happens when writing to text segment.
			// Change protections.
			if(me(vm_protect(task, addr, n, 0, VM_PROT_WRITE|VM_PROT_READ|VM_PROT_EXECUTE)) < 0){
				fprint(2, "vm_protect: %s\n", r);
				return -1;
			}
			r = vm_write(task, addr, (uintptr)v, n);
		}
		if(r != 0){
			me(r);
			return -1;
		}
		return n;
	}
}

// Convert Ureg offset to x86_thread_state32_t offset.
static int
go2darwin32(uvlong addr)
{
	switch(addr){
	case offsetof(Ureg32, ax):
		return offsetof(x86_thread_state32_t, eax);
	case offsetof(Ureg32, bx):
		return offsetof(x86_thread_state32_t, ebx);
	case offsetof(Ureg32, cx):
		return offsetof(x86_thread_state32_t, ecx);
	case offsetof(Ureg32, dx):
		return offsetof(x86_thread_state32_t, edx);
	case offsetof(Ureg32, si):
		return offsetof(x86_thread_state32_t, esi);
	case offsetof(Ureg32, di):
		return offsetof(x86_thread_state32_t, edi);
	case offsetof(Ureg32, bp):
		return offsetof(x86_thread_state32_t, ebp);
	case offsetof(Ureg32, fs):
		return offsetof(x86_thread_state32_t, fs);
	case offsetof(Ureg32, gs):
		return offsetof(x86_thread_state32_t, gs);
	case offsetof(Ureg32, pc):
		return offsetof(x86_thread_state32_t, eip);
	case offsetof(Ureg32, cs):
		return offsetof(x86_thread_state32_t, cs);
	case offsetof(Ureg32, flags):
		return offsetof(x86_thread_state32_t, eflags);
	case offsetof(Ureg32, sp):
		return offsetof(x86_thread_state32_t, esp);
	}
	return -1;
}

// Convert Ureg offset to x86_thread_state64_t offset.
static int
go2darwin64(uvlong addr)
{
	switch(addr){
	case offsetof(Ureg64, ax):
		return offsetof(x86_thread_state64_t, rax);
	case offsetof(Ureg64, bx):
		return offsetof(x86_thread_state64_t, rbx);
	case offsetof(Ureg64, cx):
		return offsetof(x86_thread_state64_t, rcx);
	case offsetof(Ureg64, dx):
		return offsetof(x86_thread_state64_t, rdx);
	case offsetof(Ureg64, si):
		return offsetof(x86_thread_state64_t, rsi);
	case offsetof(Ureg64, di):
		return offsetof(x86_thread_state64_t, rdi);
	case offsetof(Ureg64, bp):
		return offsetof(x86_thread_state64_t, rbp);
	case offsetof(Ureg64, r8):
		return offsetof(x86_thread_state64_t, r8);
	case offsetof(Ureg64, r9):
		return offsetof(x86_thread_state64_t, r9);
	case offsetof(Ureg64, r10):
		return offsetof(x86_thread_state64_t, r10);
	case offsetof(Ureg64, r11):
		return offsetof(x86_thread_state64_t, r11);
	case offsetof(Ureg64, r12):
		return offsetof(x86_thread_state64_t, r12);
	case offsetof(Ureg64, r13):
		return offsetof(x86_thread_state64_t, r13);
	case offsetof(Ureg64, r14):
		return offsetof(x86_thread_state64_t, r14);
	case offsetof(Ureg64, r15):
		return offsetof(x86_thread_state64_t, r15);
	case offsetof(Ureg64, fs):
		return offsetof(x86_thread_state64_t, fs);
	case offsetof(Ureg64, gs):
		return offsetof(x86_thread_state64_t, gs);
	case offsetof(Ureg64, ip):
		return offsetof(x86_thread_state64_t, rip);
	case offsetof(Ureg64, cs):
		return offsetof(x86_thread_state64_t, cs);
	case offsetof(Ureg64, flags):
		return offsetof(x86_thread_state64_t, rflags);
	case offsetof(Ureg64, sp):
		return offsetof(x86_thread_state64_t, rsp);
	}
	return -1;
}

extern Mach mi386;

// Read/write from fake register segment.
static int
machregrw(Map *map, Seg *seg, uvlong addr, void *v, uint n, int isr)
{
	uint nn, count, state;
	mach_port_t thread;
	int reg;
	char buf[100];
	union {
		x86_thread_state64_t reg64;
		x86_thread_state32_t reg32;
		uchar p[1];
	} u;
	uchar *p;

	if(n > 8){
		werrstr("asked for %d-byte register", n);
		return -1;
	}

	thread = idtothread(map->pid);
	if(thread == -1){
		werrstr("no such id");
		return -1;
	}

	if(mach == &mi386) {
		count = x86_THREAD_STATE32_COUNT;
		state = x86_THREAD_STATE32;
		if((reg = go2darwin32(addr)) < 0 || reg+n > sizeof u){
			if(isr){
				memset(v, 0, n);
				return 0;
			}
			werrstr("register %llud not available", addr);
			return -1;
		}
	} else {
		count = x86_THREAD_STATE64_COUNT;
		state = x86_THREAD_STATE64;
		if((reg = go2darwin64(addr)) < 0 || reg+n > sizeof u){
			if(isr){
				memset(v, 0, n);
				return 0;
			}
			werrstr("register %llud not available", addr);
			return -1;
		}
	}

	if(!isr && me(thread_suspend(thread)) < 0){
		werrstr("thread suspend %#x: %r", thread);
		return -1;
	}
	nn = count;
	if(me(thread_get_state(thread, state, (void*)u.p, &nn)) < 0){
		if(!isr)
			thread_resume(thread);
		rerrstr(buf, sizeof buf);
		if(strstr(buf, "send invalid dest") != nil) 
			werrstr("process exited");
		else
			werrstr("thread_get_state: %r");
		return -1;
	}

	p = u.p+reg;
	if(isr)
		memmove(v, p, n);
	else{
		memmove(p, v, n);
		nn = count;
		if(me(thread_set_state(thread, state, (void*)u.p, nn)) < 0){
			thread_resume(thread);
			werrstr("thread_set_state: %r");
			return -1;
		}

		if(me(thread_resume(thread)) < 0){
			werrstr("thread_resume: %r");
			return -1;
		}
	}
	return 0;
}

enum
{
	FLAGS_TF = 0x100		// x86 single-step processor flag
};

// Is thread t suspended?
static int
threadstopped(Thread *t)
{
	struct thread_basic_info info;
	uint size;

	size = sizeof info;
	if(me(thread_info(t->thread, THREAD_BASIC_INFO, (thread_info_t)&info, &size)) <  0){
		fprint(2, "threadstopped thread_info %#x: %r\n");
		return 1;
	}
	return info.suspend_count > 0;
}

// If thread t is suspended, start it up again.
// If singlestep is set, only let it execute one instruction.
static int
threadstart(Thread *t, int singlestep)
{
	int i;
	uint n;
	struct thread_basic_info info;

	if(!threadstopped(t))
		return 0;

	// Set or clear the processor single-step flag, as appropriate.
	if(mach == &mi386) {
		x86_thread_state32_t regs;
		n = x86_THREAD_STATE32_COUNT;
		if(me(thread_get_state(t->thread, x86_THREAD_STATE32,
				(thread_state_t)&regs,
				&n)) < 0)
			return -1;
		if(singlestep)
			regs.eflags |= FLAGS_TF;
		else
			regs.eflags &= ~FLAGS_TF;
		if(me(thread_set_state(t->thread, x86_THREAD_STATE32,
				(thread_state_t)&regs,
				x86_THREAD_STATE32_COUNT)) < 0)
			return -1;
	} else {
		x86_thread_state64_t regs;
		n = x86_THREAD_STATE64_COUNT;
		if(me(thread_get_state(t->thread, x86_THREAD_STATE64,
				(thread_state_t)&regs,
				&n)) < 0)
			return -1;
		if(singlestep)
			regs.rflags |= FLAGS_TF;
		else
			regs.rflags &= ~FLAGS_TF;
		if(me(thread_set_state(t->thread, x86_THREAD_STATE64,
				(thread_state_t)&regs,
				x86_THREAD_STATE64_COUNT)) < 0)
			return -1;
	}

	// Run.
	n = sizeof info;
	if(me(thread_info(t->thread, THREAD_BASIC_INFO, (thread_info_t)&info, &n)) < 0)
		return -1;
	for(i=0; i<info.suspend_count; i++)
		if(me(thread_resume(t->thread)) < 0)
			return -1;
	return 0;
}

// Stop thread t.
static int
threadstop(Thread *t)
{
	if(threadstopped(t))
		return 0;
	if(me(thread_suspend(t->thread)) < 0)
		return -1;
	return 0;
}

// Callback for exc_server below.  Called when a thread we are
// watching has an exception like hitting a breakpoint.
kern_return_t
catch_exception_raise(mach_port_t eport, mach_port_t thread,
	mach_port_t task, exception_type_t exception,
	exception_data_t code, mach_msg_type_number_t ncode)
{
	Thread *t;
	int i;

	t = nil;
	for(i=0; i<nthr; i++){
		if(thr[i].thread == thread){
			t = &thr[i];
			goto havet;
		}
	}
	if(nthr > 0)
		addpid(thr[0].pid, 1);
	for(i=0; i<nthr; i++){
		if(thr[i].thread == thread){
			t = &thr[i];
			goto havet;
		}
	}
	fprint(2, "did not find thread in catch_exception_raise\n");
	return KERN_SUCCESS;	// let thread continue

havet:
	t->exc = exception;
	if(ncode > nelem(t->code))
		ncode = nelem(t->code);
	memmove(t->code, code, ncode*sizeof t->code[0]);

	// Suspend thread, so that we can look at it & restart it later.
	if(me(thread_suspend(thread)) < 0)
		fprint(2, "catch_exception_raise thread_suspend: %r\n");

	// Synchronize with waitstop below.
	pthread_mutex_lock(&mu);
	pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&mu);

	return KERN_SUCCESS;
}

// Exception watching thread, started in addpid above.
static void*
excthread(void *v)
{
	extern boolean_t exc_server(mach_msg_header_t *, mach_msg_header_t *);
	mach_msg_server(exc_server, 2048, excport, 0);
	return 0;
}

// Wait for pid to exit.
static int exited;
static void*
waitthread(void *v)
{
	int pid, status;

	pid = (int)(uintptr)v;
	waitpid(pid, &status, 0);
	exited = 1;
	// Synchronize with waitstop below.
	pthread_mutex_lock(&mu);
	pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&mu);
	return nil;
}

// Wait for thread t to stop.
static int
waitstop(Thread *t)
{
	pthread_mutex_lock(&mu);
	while(!exited && !threadstopped(t))
		pthread_cond_wait(&cond, &mu);
	pthread_mutex_unlock(&mu);
	return 0;
}

int
ctlproc(int id, char *msg)
{
	Thread *t;
	int status;

	// Hang/attached dance is for debugging newly exec'ed programs.
	// After fork, the child does ctlproc("hang") before exec,
	// and the parent does ctlproc("attached") and then waitstop.
	// Using these requires the BSD ptrace interface, unlike everything
	// else we do, which uses only the Mach interface.  Our goal here
	// is to do as little as possible using ptrace and then flip over to Mach.

	if(strcmp(msg, "hang") == 0)
		return ptrace(PT_TRACE_ME, 0, 0, 0);

	if(strcmp(msg, "attached") == 0){
		// The pid "id" has done a ctlproc "hang" and then
		// exec, so we should find it stoppped just before exec
		// of the new program.
		#undef waitpid
		if(waitpid(id, &status, WUNTRACED) < 0){
			fprint(2, "ctlproc attached waitpid: %r\n");
			return -1;
		}
		if(WIFEXITED(status) || !WIFSTOPPED(status)){
			fprint(2, "ctlproc attached: bad process state\n");
			return -1;
		}

		// Find Mach thread for pid and suspend it.
		t = addpid(id, 1);
		if(t == nil) {
			fprint(2, "ctlproc attached: addpid: %r\n");
			return -1;
		}
		if(me(thread_suspend(t->thread)) < 0){
			fprint(2, "ctlproc attached: thread_suspend: %r\n");
			return -1;
		}

		// Let ptrace tell the process to keep going:
		// then ptrace is out of the way and we're back in Mach land.
		if(ptrace(PT_CONTINUE, id, (caddr_t)1, 0) < 0) {
			fprint(2, "ctlproc attached: ptrace continue: %r\n");
			return -1;
		}
		
		return 0;
	}

	// All the other control messages require a Thread structure.
	if((t = idtotable(id)) == nil){
		werrstr("no such thread");
		return -1;
	}

	if(strcmp(msg, "kill") == 0)
		return ptrace(PT_KILL, t->pid, 0, 0);

	if(strcmp(msg, "start") == 0)
		return threadstart(t, 0);

	if(strcmp(msg, "stop") == 0)
		return threadstop(t);

	if(strcmp(msg, "startstop") == 0){
		if(threadstart(t, 0) < 0)
			return -1;
		return waitstop(t);
	}

	if(strcmp(msg, "step") == 0){
		if(threadstart(t, 1) < 0)
			return -1;
		return waitstop(t);
	}

	if(strcmp(msg, "waitstop") == 0)
		return waitstop(t);

	// sysstop not available on OS X

	werrstr("unknown control message");
	return -1;
}

char*
procstatus(int id)
{
	Thread *t;

	if((t = idtotable(id)) == nil)
		return "gone!";

	if(threadstopped(t))
		return "Stopped";

	return "Running";
}

