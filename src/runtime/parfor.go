// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parallel for algorithm.

package runtime

// A parfor holds state for the parallel for operation.
type parfor struct {
	body   func(*parfor, uint32) // executed for each element
	done   uint32                // number of idle threads
	nthr   uint32                // total number of threads
	thrseq uint32                // thread id sequencer
	cnt    uint32                // iteration space [0, cnt)
	wait   bool                  // if true, wait while all threads finish processing,
	// otherwise parfor may return while other threads are still working

	thr []parforthread // thread descriptors

	// stats
	nsteal     uint64
	nstealcnt  uint64
	nprocyield uint64
	nosyield   uint64
	nsleep     uint64
}

// A parforthread holds state for a single thread in the parallel for.
type parforthread struct {
	// the thread's iteration space [32lsb, 32msb)
	pos uint64
	// stats
	nsteal     uint64
	nstealcnt  uint64
	nprocyield uint64
	nosyield   uint64
	nsleep     uint64
	pad        [_CacheLineSize]byte
}

func parforalloc(nthrmax uint32) *parfor {
	return &parfor{
		thr: make([]parforthread, nthrmax),
	}
}

// Parforsetup initializes desc for a parallel for operation with nthr
// threads executing n jobs.
//
// On return the nthr threads are each expected to call parfordo(desc)
// to run the operation. During those calls, for each i in [0, n), one
// thread will be used invoke body(desc, i).
// If wait is true, no parfordo will return until all work has been completed.
// If wait is false, parfordo may return when there is a small amount
// of work left, under the assumption that another thread has that
// work well in hand.
func parforsetup(desc *parfor, nthr, n uint32, wait bool, body func(*parfor, uint32)) {
	if desc == nil || nthr == 0 || nthr > uint32(len(desc.thr)) || body == nil {
		print("desc=", desc, " nthr=", nthr, " count=", n, " body=", body, "\n")
		throw("parfor: invalid args")
	}

	desc.body = body
	desc.done = 0
	desc.nthr = nthr
	desc.thrseq = 0
	desc.cnt = n
	desc.wait = wait
	desc.nsteal = 0
	desc.nstealcnt = 0
	desc.nprocyield = 0
	desc.nosyield = 0
	desc.nsleep = 0

	for i := range desc.thr {
		begin := uint32(uint64(n) * uint64(i) / uint64(nthr))
		end := uint32(uint64(n) * uint64(i+1) / uint64(nthr))
		desc.thr[i].pos = uint64(begin) | uint64(end)<<32
	}
}

func parfordo(desc *parfor) {
	// Obtain 0-based thread index.
	tid := xadd(&desc.thrseq, 1) - 1
	if tid >= desc.nthr {
		print("tid=", tid, " nthr=", desc.nthr, "\n")
		throw("parfor: invalid tid")
	}

	// If single-threaded, just execute the for serially.
	body := desc.body
	if desc.nthr == 1 {
		for i := uint32(0); i < desc.cnt; i++ {
			body(desc, i)
		}
		return
	}

	me := &desc.thr[tid]
	mypos := &me.pos
	for {
		for {
			// While there is local work,
			// bump low index and execute the iteration.
			pos := xadd64(mypos, 1)
			begin := uint32(pos) - 1
			end := uint32(pos >> 32)
			if begin < end {
				body(desc, begin)
				continue
			}
			break
		}

		// Out of work, need to steal something.
		idle := false
		for try := uint32(0); ; try++ {
			// If we don't see any work for long enough,
			// increment the done counter...
			if try > desc.nthr*4 && !idle {
				idle = true
				xadd(&desc.done, 1)
			}

			// ...if all threads have incremented the counter,
			// we are done.
			extra := uint32(0)
			if !idle {
				extra = 1
			}
			if desc.done+extra == desc.nthr {
				if !idle {
					xadd(&desc.done, 1)
				}
				goto exit
			}

			// Choose a random victim for stealing.
			var begin, end uint32
			victim := fastrand1() % (desc.nthr - 1)
			if victim >= tid {
				victim++
			}
			victimpos := &desc.thr[victim].pos
			for {
				// See if it has any work.
				pos := atomicload64(victimpos)
				begin = uint32(pos)
				end = uint32(pos >> 32)
				if begin+1 >= end {
					end = 0
					begin = end
					break
				}
				if idle {
					xadd(&desc.done, -1)
					idle = false
				}
				begin2 := begin + (end-begin)/2
				newpos := uint64(begin) | uint64(begin2)<<32
				if cas64(victimpos, pos, newpos) {
					begin = begin2
					break
				}
			}
			if begin < end {
				// Has successfully stolen some work.
				if idle {
					throw("parfor: should not be idle")
				}
				atomicstore64(mypos, uint64(begin)|uint64(end)<<32)
				me.nsteal++
				me.nstealcnt += uint64(end) - uint64(begin)
				break
			}

			// Backoff.
			if try < desc.nthr {
				// nothing
			} else if try < 4*desc.nthr {
				me.nprocyield++
				procyield(20)
			} else if !desc.wait {
				// If a caller asked not to wait for the others, exit now
				// (assume that most work is already done at this point).
				if !idle {
					xadd(&desc.done, 1)
				}
				goto exit
			} else if try < 6*desc.nthr {
				me.nosyield++
				osyield()
			} else {
				me.nsleep++
				usleep(1)
			}
		}
	}

exit:
	xadd64(&desc.nsteal, int64(me.nsteal))
	xadd64(&desc.nstealcnt, int64(me.nstealcnt))
	xadd64(&desc.nprocyield, int64(me.nprocyield))
	xadd64(&desc.nosyield, int64(me.nosyield))
	xadd64(&desc.nsleep, int64(me.nsleep))
	me.nsteal = 0
	me.nstealcnt = 0
	me.nprocyield = 0
	me.nosyield = 0
	me.nsleep = 0
}
