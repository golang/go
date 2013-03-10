// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parallel for algorithm.

#include "runtime.h"
#include "arch_GOARCH.h"

struct ParForThread
{
	// the thread's iteration space [32lsb, 32msb)
	uint64 pos;
	// stats
	uint64 nsteal;
	uint64 nstealcnt;
	uint64 nprocyield;
	uint64 nosyield;
	uint64 nsleep;
	byte pad[CacheLineSize];
};

ParFor*
runtime·parforalloc(uint32 nthrmax)
{
	ParFor *desc;

	// The ParFor object is followed by CacheLineSize padding
	// and then nthrmax ParForThread.
	desc = (ParFor*)runtime·malloc(sizeof(ParFor) + CacheLineSize + nthrmax * sizeof(ParForThread));
	desc->thr = (ParForThread*)((byte*)(desc+1) + CacheLineSize);
	desc->nthrmax = nthrmax;
	return desc;
}

// For testing from Go
// func parforalloc2(nthrmax uint32) *ParFor
void
runtime·parforalloc2(uint32 nthrmax, ParFor *desc)
{
	desc = runtime·parforalloc(nthrmax);
	FLUSH(&desc);
}

void
runtime·parforsetup(ParFor *desc, uint32 nthr, uint32 n, void *ctx, bool wait, void (*body)(ParFor*, uint32))
{
	uint32 i, begin, end;
	uint64 *pos;

	if(desc == nil || nthr == 0 || nthr > desc->nthrmax || body == nil) {
		runtime·printf("desc=%p nthr=%d count=%d body=%p\n", desc, nthr, n, body);
		runtime·throw("parfor: invalid args");
	}

	desc->body = body;
	desc->done = 0;
	desc->nthr = nthr;
	desc->thrseq = 0;
	desc->cnt = n;
	desc->ctx = ctx;
	desc->wait = wait;
	desc->nsteal = 0;
	desc->nstealcnt = 0;
	desc->nprocyield = 0;
	desc->nosyield = 0;
	desc->nsleep = 0;
	for(i=0; i<nthr; i++) {
		begin = (uint64)n*i / nthr;
		end = (uint64)n*(i+1) / nthr;
		pos = &desc->thr[i].pos;
		if(((uintptr)pos & 7) != 0)
			runtime·throw("parforsetup: pos is not aligned");
		*pos = (uint64)begin | (((uint64)end)<<32);
	}
}

// For testing from Go
// func parforsetup2(desc *ParFor, nthr, n uint32, ctx *byte, wait bool, body func(*ParFor, uint32))
void
runtime·parforsetup2(ParFor *desc, uint32 nthr, uint32 n, void *ctx, bool wait, void *body)
{
	runtime·parforsetup(desc, nthr, n, ctx, wait, *(void(**)(ParFor*, uint32))body);
}

void
runtime·parfordo(ParFor *desc)
{
	ParForThread *me;
	uint32 tid, begin, end, begin2, try, victim, i;
	uint64 *mypos, *victimpos, pos, newpos;
	void (*body)(ParFor*, uint32);
	bool idle;

	// Obtain 0-based thread index.
	tid = runtime·xadd(&desc->thrseq, 1) - 1;
	if(tid >= desc->nthr) {
		runtime·printf("tid=%d nthr=%d\n", tid, desc->nthr);
		runtime·throw("parfor: invalid tid");
	}

	// If single-threaded, just execute the for serially.
	if(desc->nthr==1) {
		for(i=0; i<desc->cnt; i++)
			desc->body(desc, i);
		return;
	}

	body = desc->body;
	me = &desc->thr[tid];
	mypos = &me->pos;
	for(;;) {
		for(;;) {
			// While there is local work,
			// bump low index and execute the iteration.
			pos = runtime·xadd64(mypos, 1);
			begin = (uint32)pos-1;
			end = (uint32)(pos>>32);
			if(begin < end) {
				body(desc, begin);
				continue;
			}
			break;
		}

		// Out of work, need to steal something.
		idle = false;
		for(try=0;; try++) {
			// If we don't see any work for long enough,
			// increment the done counter...
			if(try > desc->nthr*4 && !idle) {
				idle = true;
				runtime·xadd(&desc->done, 1);
			}
			// ...if all threads have incremented the counter,
			// we are done.
			if(desc->done + !idle == desc->nthr) {
				if(!idle)
					runtime·xadd(&desc->done, 1);
				goto exit;
			}
			// Choose a random victim for stealing.
			victim = runtime·fastrand1() % (desc->nthr-1);
			if(victim >= tid)
				victim++;
			victimpos = &desc->thr[victim].pos;
			pos = runtime·atomicload64(victimpos);
			for(;;) {
				// See if it has any work.
				begin = (uint32)pos;
				end = (uint32)(pos>>32);
				if(begin+1 >= end) {
					begin = end = 0;
					break;
				}
				if(idle) {
					runtime·xadd(&desc->done, -1);
					idle = false;
				}
				begin2 = begin + (end-begin)/2;
				newpos = (uint64)begin | (uint64)begin2<<32;
				if(runtime·cas64(victimpos, &pos, newpos)) {
					begin = begin2;
					break;
				}
			}
			if(begin < end) {
				// Has successfully stolen some work.
				if(idle)
					runtime·throw("parfor: should not be idle");
				runtime·atomicstore64(mypos, (uint64)begin | (uint64)end<<32);
				me->nsteal++;
				me->nstealcnt += end-begin;
				break;
			}
			// Backoff.
			if(try < desc->nthr) {
				// nothing
			} else if (try < 4*desc->nthr) {
				me->nprocyield++;
				runtime·procyield(20);
			// If a caller asked not to wait for the others, exit now
			// (assume that most work is already done at this point).
			} else if (!desc->wait) {
				if(!idle)
					runtime·xadd(&desc->done, 1);
				goto exit;
			} else if (try < 6*desc->nthr) {
				me->nosyield++;
				runtime·osyield();
			} else {
				me->nsleep++;
				runtime·usleep(1);
			}
		}
	}
exit:
	runtime·xadd64(&desc->nsteal, me->nsteal);
	runtime·xadd64(&desc->nstealcnt, me->nstealcnt);
	runtime·xadd64(&desc->nprocyield, me->nprocyield);
	runtime·xadd64(&desc->nosyield, me->nosyield);
	runtime·xadd64(&desc->nsleep, me->nsleep);
	me->nsteal = 0;
	me->nstealcnt = 0;
	me->nprocyield = 0;
	me->nosyield = 0;
	me->nsleep = 0;
}

// For testing from Go
// func parforiters(desc *ParFor, tid uintptr) (uintptr, uintptr)
void
runtime·parforiters(ParFor *desc, uintptr tid, uintptr start, uintptr end)
{
	start = (uint32)desc->thr[tid].pos;
	end = (uint32)(desc->thr[tid].pos>>32);
	FLUSH(&start);
	FLUSH(&end);
}
