// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parallel for algorithm.

package runtime

import "unsafe"

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

func desc_thr_index(desc *parfor, i uint32) *parforthread {
	return (*parforthread)(add(unsafe.Pointer(desc.thr), uintptr(i)*unsafe.Sizeof(*desc.thr)))
}

func parforsetup(desc *parfor, nthr, n uint32, ctx unsafe.Pointer, wait bool, body func(*parfor, uint32)) {
	if desc == nil || nthr == 0 || nthr > desc.nthrmax || body == nil {
		print("desc=", desc, " nthr=", nthr, " count=", n, " body=", body, "\n")
		throw("parfor: invalid args")
	}

	desc.body = *(*unsafe.Pointer)(unsafe.Pointer(&body))
	desc.done = 0
	desc.nthr = nthr
	desc.thrseq = 0
	desc.cnt = n
	desc.ctx = ctx
	desc.wait = wait
	desc.nsteal = 0
	desc.nstealcnt = 0
	desc.nprocyield = 0
	desc.nosyield = 0
	desc.nsleep = 0

	for i := uint32(0); i < nthr; i++ {
		begin := uint32(uint64(n) * uint64(i) / uint64(nthr))
		end := uint32(uint64(n) * uint64(i+1) / uint64(nthr))
		pos := &desc_thr_index(desc, i).pos
		if uintptr(unsafe.Pointer(pos))&7 != 0 {
			throw("parforsetup: pos is not aligned")
		}
		*pos = uint64(begin) | uint64(end)<<32
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
	body := *(*func(*parfor, uint32))(unsafe.Pointer(&desc.body))
	if desc.nthr == 1 {
		for i := uint32(0); i < desc.cnt; i++ {
			body(desc, i)
		}
		return
	}

	me := desc_thr_index(desc, tid)
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
			victimpos := &desc_thr_index(desc, victim).pos
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
