// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "amd64_darwin.h"
#include "signals.h"

extern void _rt0_amd64_darwin();
byte* startsym = (byte*)_rt0_amd64_darwin;

typedef uint64 __uint64_t;

/* From /usr/include/mach/i386/_structs.h */
#define	_STRUCT_X86_THREAD_STATE64	struct __darwin_x86_thread_state64
_STRUCT_X86_THREAD_STATE64
{
	__uint64_t	__rax;
	__uint64_t	__rbx;
	__uint64_t	__rcx;
	__uint64_t	__rdx;
	__uint64_t	__rdi;
	__uint64_t	__rsi;
	__uint64_t	__rbp;
	__uint64_t	__rsp;
	__uint64_t	__r8;
	__uint64_t	__r9;
	__uint64_t	__r10;
	__uint64_t	__r11;
	__uint64_t	__r12;
	__uint64_t	__r13;
	__uint64_t	__r14;
	__uint64_t	__r15;
	__uint64_t	__rip;
	__uint64_t	__rflags;
	__uint64_t	__cs;
	__uint64_t	__fs;
	__uint64_t	__gs;
};


void
print_thread_state(_STRUCT_X86_THREAD_STATE64* ss)
{
	prints("\nrax     0x");  sys·printpointer((void*)ss->__rax);
	prints("\nrbx     0x");  sys·printpointer((void*)ss->__rbx);
	prints("\nrcx     0x");  sys·printpointer((void*)ss->__rcx);
	prints("\nrdx     0x");  sys·printpointer((void*)ss->__rdx);
	prints("\nrdi     0x");  sys·printpointer((void*)ss->__rdi);
	prints("\nrsi     0x");  sys·printpointer((void*)ss->__rsi);
	prints("\nrbp     0x");  sys·printpointer((void*)ss->__rbp);
	prints("\nrsp     0x");  sys·printpointer((void*)ss->__rsp);
	prints("\nr8      0x");  sys·printpointer((void*)ss->__r8 );
	prints("\nr9      0x");  sys·printpointer((void*)ss->__r9 );
	prints("\nr10     0x");  sys·printpointer((void*)ss->__r10);
	prints("\nr11     0x");  sys·printpointer((void*)ss->__r11);
	prints("\nr12     0x");  sys·printpointer((void*)ss->__r12);
	prints("\nr13     0x");  sys·printpointer((void*)ss->__r13);
	prints("\nr14     0x");  sys·printpointer((void*)ss->__r14);
	prints("\nr15     0x");  sys·printpointer((void*)ss->__r15);
	prints("\nrip     0x");  sys·printpointer((void*)ss->__rip);
	prints("\nrflags  0x");  sys·printpointer((void*)ss->__rflags);
	prints("\ncs      0x");  sys·printpointer((void*)ss->__cs);
	prints("\nfs      0x");  sys·printpointer((void*)ss->__fs);
	prints("\ngs      0x");  sys·printpointer((void*)ss->__gs);
	prints("\n");
}


/* Code generated via: g++ -m64 gen_signals_support.cc && a.out */

static void *adr_at(void *ptr, int32 offs) {
  return (void *)((uint8 *)ptr + offs);
}

static void *ptr_at(void *ptr, int32 offs) {
  return *(void **)((uint8 *)ptr + offs);
}

typedef void ucontext_t;
typedef void _STRUCT_MCONTEXT64;
typedef void _STRUCT_X86_EXCEPTION_STATE64;
typedef void _STRUCT_X86_FLOAT_STATE64;

static _STRUCT_MCONTEXT64 *get_uc_mcontext(ucontext_t *ptr) {
  return (_STRUCT_MCONTEXT64 *)ptr_at(ptr, 48);
}

static _STRUCT_X86_EXCEPTION_STATE64 *get___es(_STRUCT_MCONTEXT64 *ptr) {
  return (_STRUCT_X86_EXCEPTION_STATE64 *)adr_at(ptr, 0);
}

static _STRUCT_X86_THREAD_STATE64 *get___ss(_STRUCT_MCONTEXT64 *ptr) {
  return (_STRUCT_X86_THREAD_STATE64 *)adr_at(ptr, 16);
}

static _STRUCT_X86_FLOAT_STATE64 *get___fs(_STRUCT_MCONTEXT64 *ptr) {
  return (_STRUCT_X86_FLOAT_STATE64 *)adr_at(ptr, 184);
}

/* End of generated code */


/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void sigtramp();

/*
 * Rudimentary reverse-engineered definition of signal interface.
 * You'd think it would be documented.
 */
typedef struct siginfo {
	int32	si_signo;		/* signal number */
	int32	si_errno;		/* errno association */
	int32	si_code;		/* signal code */
	int32	si_pid;			/* sending process */
	int32	si_uid;			/* sender's ruid */
	int32	si_status;		/* exit value */
	void	*si_addr;		/* faulting address */
	/* more stuff here */
} siginfo;


typedef struct  sigaction {
 	union {
		void (*sa_handler)(int32);
		void (*sa_sigaction)(int32, siginfo *, void *);
	} u;				/* signal handler */
	void (*sa_trampoline)(void);	/* kernel callback point; calls sighandler() */
	uint8 sa_mask[4];		/* signal mask to apply */
	int32 sa_flags;			/* see signal options below */
} sigaction;

void
sighandler(int32 sig, siginfo *info, void *context)
{
	if(panicking)	// traceback already printed
		sys·exit(2);

        _STRUCT_MCONTEXT64 *uc_mcontext = get_uc_mcontext(context);
        _STRUCT_X86_THREAD_STATE64 *ss = get___ss(uc_mcontext);

	if(!inlinetrap(sig, (byte *)ss->__rip)) {
		if(sig < 0 || sig >= NSIG){
			prints("Signal ");
			sys·printint(sig);
		}else{
			prints(sigtab[sig].name);
		}
	}

	prints("\nFaulting address: 0x");  sys·printpointer(info->si_addr);
	prints("\npc: 0x");  sys·printpointer((void *)ss->__rip);
	prints("\n\n");

	traceback((void *)ss->__rip, (void *)ss->__rsp, (void*)ss->__r15);
	tracebackothers((void*)ss->__r15);
	print_thread_state(ss);

	sys·exit(2);
}


sigaction a;
extern void sigtramp(void);

void
initsig(void)
{
	int32 i;
	a.u.sa_sigaction = (void*)sigtramp;
	a.sa_flags |= 0x40;  /* SA_SIGINFO */
	for(i=0; i<sizeof(a.sa_mask); i++)
		a.sa_mask[i] = 0xFF;
	a.sa_trampoline = sigtramp;

	for(i = 0; i <NSIG; i++)
		if(sigtab[i].catch){
			sys·sigaction(i, &a, (void*)0);
		}
}

static void
unimplemented(int8 *name)
{
	prints(name);
	prints(" not implemented\n");
	*(int32*)1231 = 1231;
}

void
sys·sleep(int64 ms)
{
	struct timeval tv;

	tv.tv_sec = ms/1000;
	tv.tv_usec = ms%1000 * 1000;
	select(0, nil, nil, nil, &tv);
}

// Thread-safe allocation of a semaphore.
// Psema points at a kernel semaphore key.
// It starts out zero, meaning no semaphore.
// Fill it in, being careful of others calling initsema
// simultaneously.
static void
initsema(uint32 *psema)
{
	uint32 sema;

	if(*psema != 0)	// already have one
		return;

	sema = semcreate();
	if(!cas(psema, 0, sema)){
		// Someone else filled it in.  Use theirs.
		semdestroy(sema);
		return;
	}
}


// Atomic add and return new value.
static uint32
xadd(uint32 volatile *val, int32 delta)
{
	uint32 oval, nval;

	for(;;){
		oval = *val;
		nval = oval + delta;
		if(cas(val, oval, nval))
			return nval;
	}
}


// Blocking locks.

// Implement Locks, using semaphores.
// l->key is the number of threads who want the lock.
// In a race, one thread increments l->key from 0 to 1
// and the others increment it from >0 to >1.  The thread
// who does the 0->1 increment gets the lock, and the
// others wait on the semaphore.  When the 0->1 thread
// releases the lock by decrementing l->key, l->key will
// be >0, so it will increment the semaphore to wake up
// one of the others.  This is the same algorithm used
// in Plan 9's user-space locks.
//
// Note that semaphores are never destroyed (the kernel
// will clean up when the process exits).  We assume for now
// that Locks are only used for long-lived structures like M and G.

void
lock(Lock *l)
{
	// Allocate semaphore if needed.
	if(l->sema == 0)
		initsema(&l->sema);

	if(xadd(&l->key, 1) > 1)	// someone else has it; wait
		semacquire(l->sema);
}

void
unlock(Lock *l)
{
	if(xadd(&l->key, -1) > 0)	// someone else is waiting
		semrelease(l->sema);
}


// Event notifications.
void
noteclear(Note *n)
{
	n->wakeup = 0;
}

void
notesleep(Note *n)
{
	if(n->sema == 0)
		initsema(&n->sema);
	while(!n->wakeup)
		semacquire(n->sema);
}

void
notewakeup(Note *n)
{
	if(n->sema == 0)
		initsema(&n->sema);
	n->wakeup = 1;
	semrelease(n->sema);
}


// BSD interface for threading.
void
osinit(void)
{
	// Register our thread-creation callback (see sys_amd64_darwin.s).
	bsdthread_register();
}

void
newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	bsdthread_create(stk, m, g, fn);
}


// Mach IPC, to get at semaphores
// Definitions are in /usr/include/mach on a Mac.

static void
macherror(kern_return_t r, int8 *fn)
{
	prints("mach error ");
	prints(fn);
	prints(": ");
	sys·printint(r);
	prints("\n");
	throw("mach error");
}

enum
{
	DebugMach = 0
};

typedef int32 mach_msg_option_t;
typedef uint32 mach_msg_bits_t;
typedef uint32 mach_msg_id_t;
typedef uint32 mach_msg_size_t;
typedef uint32 mach_msg_timeout_t;
typedef uint32 mach_port_name_t;
typedef uint64 mach_vm_address_t;

typedef struct mach_msg_header_t mach_msg_header_t;
typedef struct mach_msg_body_t mach_msg_body_t;
typedef struct mach_msg_port_descriptor_t mach_msg_port_descriptor_t;
typedef struct NDR_record_t NDR_record_t;

enum
{
	MACH_MSG_TYPE_MOVE_RECEIVE = 16,
	MACH_MSG_TYPE_MOVE_SEND = 17,
	MACH_MSG_TYPE_MOVE_SEND_ONCE = 18,
	MACH_MSG_TYPE_COPY_SEND = 19,
	MACH_MSG_TYPE_MAKE_SEND = 20,
	MACH_MSG_TYPE_MAKE_SEND_ONCE = 21,
	MACH_MSG_TYPE_COPY_RECEIVE = 22,

	MACH_MSG_PORT_DESCRIPTOR = 0,
	MACH_MSG_OOL_DESCRIPTOR = 1,
	MACH_MSG_OOL_PORTS_DESCRIPTOR = 2,
	MACH_MSG_OOL_VOLATILE_DESCRIPTOR = 3,

	MACH_MSGH_BITS_COMPLEX = 0x80000000,

	MACH_SEND_MSG = 1,
	MACH_RCV_MSG = 2,
	MACH_RCV_LARGE = 4,

	MACH_SEND_TIMEOUT = 0x10,
	MACH_SEND_INTERRUPT = 0x40,
	MACH_SEND_CANCEL = 0x80,
	MACH_SEND_ALWAYS = 0x10000,
	MACH_SEND_TRAILER = 0x20000,
	MACH_RCV_TIMEOUT = 0x100,
	MACH_RCV_NOTIFY = 0x200,
	MACH_RCV_INTERRUPT = 0x400,
	MACH_RCV_OVERWRITE = 0x1000,
};

mach_port_t mach_task_self(void);
mach_port_t mach_thread_self(void);

#pragma pack on
struct mach_msg_header_t
{
	mach_msg_bits_t bits;
	mach_msg_size_t size;
	mach_port_t remote_port;
	mach_port_t local_port;
	mach_msg_size_t reserved;
	mach_msg_id_t id;
};

struct mach_msg_body_t
{
	uint32 descriptor_count;
};

struct mach_msg_port_descriptor_t
{
	mach_port_t name;
	uint32 pad1;
	uint16 pad2;
	uint8 disposition;
	uint8 type;
};

enum
{
	NDR_PROTOCOL_2_0 = 0,
	NDR_INT_BIG_ENDIAN = 0,
	NDR_INT_LITTLE_ENDIAN = 1,
	NDR_FLOAT_IEEE = 0,
	NDR_CHAR_ASCII = 0
};

struct NDR_record_t
{
	uint8 mig_vers;
	uint8 if_vers;
	uint8 reserved1;
	uint8 mig_encoding;
	uint8 int_rep;
	uint8 char_rep;
	uint8 float_rep;
	uint8 reserved2;
};
#pragma pack off

static NDR_record_t zerondr;

#define MACH_MSGH_BITS(a, b) ((a) | ((b)<<8))

// Mach system calls (in sys_amd64_darwin.s)
kern_return_t mach_msg_trap(mach_msg_header_t*,
	mach_msg_option_t, mach_msg_size_t, mach_msg_size_t,
	mach_port_name_t, mach_msg_timeout_t, mach_port_name_t);
mach_port_t mach_reply_port(void);
mach_port_t mach_task_self(void);
mach_port_t mach_thread_self(void);

static kern_return_t
mach_msg(mach_msg_header_t *h,
	mach_msg_option_t op,
	mach_msg_size_t send_size,
	mach_msg_size_t rcv_size,
	mach_port_name_t rcv_name,
	mach_msg_timeout_t timeout,
	mach_port_name_t notify)
{
	// TODO: Loop on interrupt.
	return mach_msg_trap(h, op, send_size, rcv_size, rcv_name, timeout, notify);
}


// Mach RPC (MIG)
// I'm not using the Mach names anymore.  They're too long.

enum
{
	MinMachMsg = 48,
	Reply = 100,
};

#pragma pack on
typedef struct CodeMsg CodeMsg;
struct CodeMsg
{
	mach_msg_header_t h;
	NDR_record_t NDR;
	kern_return_t code;
};
#pragma pack off

static kern_return_t
machcall(mach_msg_header_t *h, int32 maxsize, int32 rxsize)
{
	uint32 *p;
	int32 i, ret, id;
	mach_port_t port;
	CodeMsg *c;

	if((port = m->machport) == 0){
		port = mach_reply_port();
		m->machport = port;
	}

	h->bits |= MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
	h->local_port = port;
	h->reserved = 0;
	id = h->id;

	if(DebugMach){
		p = (uint32*)h;
		prints("send:\t");
		for(i=0; i<h->size/sizeof(p[0]); i++){
			prints(" ");
			sys·printpointer((void*)p[i]);
			if(i%8 == 7)
				prints("\n\t");
		}
		if(i%8)
			prints("\n");
	}

	ret = mach_msg(h, MACH_SEND_MSG|MACH_RCV_MSG,
		h->size, maxsize, port, 0, 0);
	if(ret != 0){
		if(DebugMach){
			prints("mach_msg error ");
			sys·printint(ret);
			prints("\n");
		}
		return ret;
	}

	if(DebugMach){
		p = (uint32*)h;
		prints("recv:\t");
		for(i=0; i<h->size/sizeof(p[0]); i++){
			prints(" ");
			sys·printpointer((void*)p[i]);
			if(i%8 == 7)
				prints("\n\t");
		}
		if(i%8)
			prints("\n");
	}

	if(h->id != id+Reply){
		if(DebugMach){
			prints("mach_msg reply id mismatch ");
			sys·printint(h->id);
			prints(" != ");
			sys·printint(id+Reply);
			prints("\n");
		}
		return -303;	// MIG_REPLY_MISMATCH
	}

	// Look for a response giving the return value.
	// Any call can send this back with an error,
	// and some calls only have return values so they
	// send it back on success too.  I don't quite see how
	// you know it's one of these and not the full response
	// format, so just look if the message is right.
	c = (CodeMsg*)h;
	if(h->size == sizeof(CodeMsg)
	&& !(h->bits & MACH_MSGH_BITS_COMPLEX)){
		if(DebugMach){
			prints("mig result ");
			sys·printint(c->code);
			prints("\n");
		}
		return c->code;
	}

	if(h->size != rxsize){
		if(DebugMach){
			prints("mach_msg reply size mismatch ");
			sys·printint(h->size);
			prints(" != ");
			sys·printint(rxsize);
			prints("\n");
		}
		return -307;	// MIG_ARRAY_TOO_LARGE
	}

	return 0;
}


// Semaphores!

enum
{
	Tsemcreate = 3418,
	Rsemcreate = Tsemcreate + Reply,

	Tsemdestroy = 3419,
	Rsemdestroy = Tsemdestroy + Reply,
};

typedef struct TsemcreateMsg TsemcreateMsg;
typedef struct RsemcreateMsg RsemcreateMsg;
typedef struct TsemdestroyMsg TsemdestroyMsg;
// RsemdestroyMsg = CodeMsg

#pragma pack on
struct TsemcreateMsg
{
	mach_msg_header_t h;
	NDR_record_t ndr;
	int32 policy;
	int32 value;
};

struct RsemcreateMsg
{
	mach_msg_header_t h;
	mach_msg_body_t body;
	mach_msg_port_descriptor_t semaphore;
};

struct TsemdestroyMsg
{
	mach_msg_header_t h;
	mach_msg_body_t body;
	mach_msg_port_descriptor_t semaphore;
};
#pragma pack off

mach_port_t
semcreate(void)
{
	union {
		TsemcreateMsg tx;
		RsemcreateMsg rx;
		uint8 pad[MinMachMsg];
	} m;
	kern_return_t r;

	m.tx.h.bits = 0;
	m.tx.h.size = sizeof(m.tx);
	m.tx.h.remote_port = mach_task_self();
	m.tx.h.id = Tsemcreate;
	m.tx.ndr = zerondr;

	m.tx.policy = 0;	// 0 = SYNC_POLICY_FIFO
	m.tx.value = 0;

	if((r = machcall(&m.tx.h, sizeof m, sizeof(m.rx))) != 0)
		macherror(r, "semaphore_create");
	if(m.rx.body.descriptor_count != 1)
		unimplemented("semcreate desc count");
	return m.rx.semaphore.name;
}

void
semdestroy(mach_port_t sem)
{
	union {
		TsemdestroyMsg tx;
		uint8 pad[MinMachMsg];
	} m;
	kern_return_t r;

	m.tx.h.bits = MACH_MSGH_BITS_COMPLEX;
	m.tx.h.size = sizeof(m.tx);
	m.tx.h.remote_port = mach_task_self();
	m.tx.h.id = Tsemdestroy;
	m.tx.body.descriptor_count = 1;
	m.tx.semaphore.name = sem;
	m.tx.semaphore.disposition = MACH_MSG_TYPE_MOVE_SEND;
	m.tx.semaphore.type = 0;

	if((r = machcall(&m.tx.h, sizeof m, 0)) != 0)
		macherror(r, "semaphore_destroy");
}

// The other calls have simple system call traps
// in sys_amd64_darwin.s
kern_return_t mach_semaphore_wait(uint32 sema);
kern_return_t mach_semaphore_timedwait(uint32 sema, uint32 sec, uint32 nsec);
kern_return_t mach_semaphore_signal(uint32 sema);
kern_return_t mach_semaphore_signal_all(uint32 sema);

void
semacquire(mach_port_t sem)
{
	kern_return_t r;

	if((r = mach_semaphore_wait(sem)) != 0)
		macherror(r, "semaphore_wait");
}

void
semrelease(mach_port_t sem)
{
	kern_return_t r;

	if((r = mach_semaphore_signal(sem)) != 0)
		macherror(r, "semaphore_signal");
}

