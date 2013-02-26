// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CPU profiling.
// Based on algorithms and data structures used in
// http://code.google.com/p/google-perftools/.
//
// The main difference between this code and the google-perftools
// code is that this code is written to allow copying the profile data
// to an arbitrary io.Writer, while the google-perftools code always
// writes to an operating system file.
//
// The signal handler for the profiling clock tick adds a new stack trace
// to a hash table tracking counts for recent traces.  Most clock ticks
// hit in the cache.  In the event of a cache miss, an entry must be 
// evicted from the hash table, copied to a log that will eventually be
// written as profile data.  The google-perftools code flushed the
// log itself during the signal handler.  This code cannot do that, because
// the io.Writer might block or need system calls or locks that are not
// safe to use from within the signal handler.  Instead, we split the log
// into two halves and let the signal handler fill one half while a goroutine
// is writing out the other half.  When the signal handler fills its half, it
// offers to swap with the goroutine.  If the writer is not done with its half,
// we lose the stack trace for this clock tick (and record that loss).
// The goroutine interacts with the signal handler by calling getprofile() to
// get the next log piece to write, implicitly handing back the last log
// piece it obtained.
//
// The state of this dance between the signal handler and the goroutine
// is encoded in the Profile.handoff field.  If handoff == 0, then the goroutine
// is not using either log half and is waiting (or will soon be waiting) for
// a new piece by calling notesleep(&p->wait).  If the signal handler
// changes handoff from 0 to non-zero, it must call notewakeup(&p->wait)
// to wake the goroutine.  The value indicates the number of entries in the
// log half being handed off.  The goroutine leaves the non-zero value in
// place until it has finished processing the log half and then flips the number
// back to zero.  Setting the high bit in handoff means that the profiling is over, 
// and the goroutine is now in charge of flushing the data left in the hash table
// to the log and returning that data.  
//
// The handoff field is manipulated using atomic operations.
// For the most part, the manipulation of handoff is orderly: if handoff == 0
// then the signal handler owns it and can change it to non-zero.  
// If handoff != 0 then the goroutine owns it and can change it to zero.
// If that were the end of the story then we would not need to manipulate
// handoff using atomic operations.  The operations are needed, however,
// in order to let the log closer set the high bit to indicate "EOF" safely
// in the situation when normally the goroutine "owns" handoff.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

enum
{
	HashSize = 1<<10,
	LogSize = 1<<17,
	Assoc = 4,
	MaxStack = 64,
};

typedef struct Profile Profile;
typedef struct Bucket Bucket;
typedef struct Entry Entry;

struct Entry {
	uintptr count;
	uintptr depth;
	uintptr stack[MaxStack];
};

struct Bucket {
	Entry entry[Assoc];
};

struct Profile {
	bool on;		// profiling is on
	Note wait;		// goroutine waits here
	uintptr count;		// tick count
	uintptr evicts;		// eviction count
	uintptr lost;		// lost ticks that need to be logged
	uintptr totallost;	// total lost ticks

	// Active recent stack traces.
	Bucket hash[HashSize];

	// Log of traces evicted from hash.
	// Signal handler has filled log[toggle][:nlog].
	// Goroutine is writing log[1-toggle][:handoff].
	uintptr log[2][LogSize/2];
	uintptr nlog;
	int32 toggle;
	uint32 handoff;
	
	// Writer state.
	// Writer maintains its own toggle to avoid races
	// looking at signal handler's toggle.
	uint32 wtoggle;
	bool wholding;	// holding & need to release a log half
	bool flushing;	// flushing hash table - profile is over
	bool eod_sent;  // special end-of-data record sent; => flushing
};

static Lock lk;
static Profile *prof;

static void tick(uintptr*, int32);
static void add(Profile*, uintptr*, int32);
static bool evict(Profile*, Entry*);
static bool flushlog(Profile*);

static uintptr eod[3] = {0, 1, 0};

// LostProfileData is a no-op function used in profiles
// to mark the number of profiling stack traces that were
// discarded due to slow data writers.
static void
LostProfileData(void)
{
}

// SetCPUProfileRate sets the CPU profiling rate.
// The user documentation is in debug.go.
void
runtime·SetCPUProfileRate(intgo hz)
{
	uintptr *p;
	uintptr n;
	
	// Call findfunc now so that it won't have to
	// build tables during the signal handler.
	runtime·findfunc(0);

	// Clamp hz to something reasonable.
	if(hz < 0)
		hz = 0;
	if(hz > 1000000)
		hz = 1000000;

	runtime·lock(&lk);
	if(hz > 0) {
		if(prof == nil) {
			prof = runtime·SysAlloc(sizeof *prof);
			if(prof == nil) {
				runtime·printf("runtime: cpu profiling cannot allocate memory\n");
				runtime·unlock(&lk);
				return;
			}
		}
		if(prof->on || prof->handoff != 0) {
			runtime·printf("runtime: cannot set cpu profile rate until previous profile has finished.\n");
			runtime·unlock(&lk);
			return;
		}

		prof->on = true;
		p = prof->log[0];
		// pprof binary header format.
		// http://code.google.com/p/google-perftools/source/browse/trunk/src/profiledata.cc#117
		*p++ = 0;  // count for header
		*p++ = 3;  // depth for header
		*p++ = 0;  // version number
		*p++ = 1000000 / hz;  // period (microseconds)
		*p++ = 0;
		prof->nlog = p - prof->log[0];
		prof->toggle = 0;
		prof->wholding = false;
		prof->wtoggle = 0;
		prof->flushing = false;
		prof->eod_sent = false;
		runtime·noteclear(&prof->wait);

		runtime·setcpuprofilerate(tick, hz);
	} else if(prof->on) {
		runtime·setcpuprofilerate(nil, 0);
		prof->on = false;

		// Now add is not running anymore, and getprofile owns the entire log.
		// Set the high bit in prof->handoff to tell getprofile.
		for(;;) {
			n = prof->handoff;
			if(n&0x80000000)
				runtime·printf("runtime: setcpuprofile(off) twice");
			if(runtime·cas(&prof->handoff, n, n|0x80000000))
				break;
		}
		if(n == 0) {
			// we did the transition from 0 -> nonzero so we wake getprofile
			runtime·notewakeup(&prof->wait);
		}
	}
	runtime·unlock(&lk);
}

static void
tick(uintptr *pc, int32 n)
{
	add(prof, pc, n);
}

// add adds the stack trace to the profile.
// It is called from signal handlers and other limited environments
// and cannot allocate memory or acquire locks that might be
// held at the time of the signal, nor can it use substantial amounts
// of stack.  It is allowed to call evict.
static void
add(Profile *p, uintptr *pc, int32 n)
{
	int32 i, j;
	uintptr h, x;
	Bucket *b;
	Entry *e;

	if(n > MaxStack)
		n = MaxStack;
	
	// Compute hash.
	h = 0;
	for(i=0; i<n; i++) {
		h = h<<8 | (h>>(8*(sizeof(h)-1)));
		x = pc[i];
		h += x*31 + x*7 + x*3;
	}
	p->count++;

	// Add to entry count if already present in table.
	b = &p->hash[h%HashSize];
	for(i=0; i<Assoc; i++) {
		e = &b->entry[i];
		if(e->depth != n)	
			continue;
		for(j=0; j<n; j++)
			if(e->stack[j] != pc[j])
				goto ContinueAssoc;
		e->count++;
		return;
	ContinueAssoc:;
	}

	// Evict entry with smallest count.
	e = &b->entry[0];
	for(i=1; i<Assoc; i++)
		if(b->entry[i].count < e->count)
			e = &b->entry[i];
	if(e->count > 0) {
		if(!evict(p, e)) {
			// Could not evict entry.  Record lost stack.
			p->lost++;
			p->totallost++;
			return;
		}
		p->evicts++;
	}
	
	// Reuse the newly evicted entry.
	e->depth = n;
	e->count = 1;
	for(i=0; i<n; i++)
		e->stack[i] = pc[i];
}

// evict copies the given entry's data into the log, so that
// the entry can be reused.  evict is called from add, which
// is called from the profiling signal handler, so it must not
// allocate memory or block.  It is safe to call flushLog.
// evict returns true if the entry was copied to the log,
// false if there was no room available.
static bool
evict(Profile *p, Entry *e)
{
	int32 i, d, nslot;
	uintptr *log, *q;
	
	d = e->depth;
	nslot = d+2;
	log = p->log[p->toggle];
	if(p->nlog+nslot > nelem(p->log[0])) {
		if(!flushlog(p))
			return false;
		log = p->log[p->toggle];
	}
	
	q = log+p->nlog;
	*q++ = e->count;
	*q++ = d;
	for(i=0; i<d; i++)
		*q++ = e->stack[i];
	p->nlog = q - log;
	e->count = 0;
	return true;
}

// flushlog tries to flush the current log and switch to the other one.
// flushlog is called from evict, called from add, called from the signal handler,
// so it cannot allocate memory or block.  It can try to swap logs with
// the writing goroutine, as explained in the comment at the top of this file.
static bool
flushlog(Profile *p)
{
	uintptr *log, *q;

	if(!runtime·cas(&p->handoff, 0, p->nlog))
		return false;
	runtime·notewakeup(&p->wait);

	p->toggle = 1 - p->toggle;
	log = p->log[p->toggle];
	q = log;
	if(p->lost > 0) {
		*q++ = p->lost;
		*q++ = 1;
		*q++ = (uintptr)LostProfileData;
	}
	p->nlog = q - log;
	return true;
}

// getprofile blocks until the next block of profiling data is available
// and returns it as a []byte.  It is called from the writing goroutine.
Slice
getprofile(Profile *p)
{
	uint32 i, j, n;
	Slice ret;
	Bucket *b;
	Entry *e;

	ret.array = nil;
	ret.len = 0;
	ret.cap = 0;
	
	if(p == nil)	
		return ret;

	if(p->wholding) {
		// Release previous log to signal handling side.
		// Loop because we are racing against setprofile(off).
		for(;;) {
			n = p->handoff;
			if(n == 0) {
				runtime·printf("runtime: phase error during cpu profile handoff\n");
				return ret;
			}
			if(n & 0x80000000) {
				p->wtoggle = 1 - p->wtoggle;
				p->wholding = false;
				p->flushing = true;
				goto flush;
			}
			if(runtime·cas(&p->handoff, n, 0))
				break;
		}
		p->wtoggle = 1 - p->wtoggle;
		p->wholding = false;
	}
	
	if(p->flushing)
		goto flush;
	
	if(!p->on && p->handoff == 0)
		return ret;

	// Wait for new log.
	runtime·entersyscallblock();
	runtime·notesleep(&p->wait);
	runtime·exitsyscall();
	runtime·noteclear(&p->wait);

	n = p->handoff;
	if(n == 0) {
		runtime·printf("runtime: phase error during cpu profile wait\n");
		return ret;
	}
	if(n == 0x80000000) {
		p->flushing = true;
		goto flush;
	}
	n &= ~0x80000000;

	// Return new log to caller.
	p->wholding = true;

	ret.array = (byte*)p->log[p->wtoggle];
	ret.len = n*sizeof(uintptr);
	ret.cap = ret.len;
	return ret;

flush:
	// In flush mode.
	// Add is no longer being called.  We own the log.
	// Also, p->handoff is non-zero, so flushlog will return false.
	// Evict the hash table into the log and return it.
	for(i=0; i<HashSize; i++) {
		b = &p->hash[i];
		for(j=0; j<Assoc; j++) {
			e = &b->entry[j];
			if(e->count > 0 && !evict(p, e)) {
				// Filled the log.  Stop the loop and return what we've got.
				goto breakflush;
			}
		}
	}
breakflush:

	// Return pending log data.
	if(p->nlog > 0) {
		// Note that we're using toggle now, not wtoggle,
		// because we're working on the log directly.
		ret.array = (byte*)p->log[p->toggle];
		ret.len = p->nlog*sizeof(uintptr);
		ret.cap = ret.len;
		p->nlog = 0;
		return ret;
	}

	// Made it through the table without finding anything to log.
	if(!p->eod_sent) {
		// We may not have space to append this to the partial log buf,
		// so we always return a new slice for the end-of-data marker.
		p->eod_sent = true;
		ret.array = (byte*)eod;
		ret.len = sizeof eod;
		ret.cap = ret.len;
		return ret;
	}

	// Finally done.  Clean up and return nil.
	p->flushing = false;
	if(!runtime·cas(&p->handoff, p->handoff, 0))
		runtime·printf("runtime: profile flush racing with something\n");
	return ret;  // set to nil at top of function
}

// CPUProfile returns the next cpu profile block as a []byte.
// The user documentation is in debug.go.
void
runtime·CPUProfile(Slice ret)
{
	ret = getprofile(prof);
	FLUSH(&ret);
}
