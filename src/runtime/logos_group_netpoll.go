// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1 || windows

package runtime

import (
	"internal/runtime/atomic"
	_ "unsafe"
)

const (
	logosCancelRead  uint32 = 1
	logosCancelWrite uint32 = 2
)

type logosWaiter struct {
	pd    *pollDesc
	gp    *g
	mode  int32
	group uint32
}

var (
	logosWaiters               []logosWaiter
	logosWaiterCount           uint32
	logosWaitAfterRegisterHook func()
)

func logosWaitersSync() {
	atomic.Store(&logosWaiterCount, uint32(len(logosWaiters)))
}

func logosCancelBit(mode int32) uint32 {
	if mode == 'w' {
		return logosCancelWrite
	}
	return logosCancelRead
}

func logosCancelled(pd *pollDesc, mode int32) bool {
	return atomic.Load(&pd.logosCancel)&logosCancelBit(mode) != 0
}

func logosWaitRegister(pd *pollDesc, mode int32) bool {
	gp := getg()
	if gp.logosGroup == 0 {
		return false
	}
	lock(&logosWaitersLock)
	if logosSlotCancelGet(gp.logosGroup) {
		atomic.Or32(&pd.logosCancel, logosCancelBit(mode))
	}
	logosWaiters = append(logosWaiters, logosWaiter{pd: pd, gp: gp, mode: mode, group: gp.logosGroup})
	logosWaitersSync()
	unlock(&logosWaitersLock)
	return true
}

func logosWaitRelease(pd *pollDesc, mode int32) {
	gp := getg()
	owned := false
	lock(&logosWaitersLock)
	for i := range logosWaiters {
		if logosWaiters[i].gp == gp {
			last := len(logosWaiters) - 1
			logosWaiters[i] = logosWaiters[last]
			logosWaiters[last] = logosWaiter{}
			logosWaiters = logosWaiters[:last]
			logosWaitersSync()
			owned = true
			break
		}
	}
	if owned {
		atomic.And32(&pd.logosCancel, ^logosCancelBit(mode))
	}
	unlock(&logosWaitersLock)
}

func logosPollPurge(pd *pollDesc) {
	if atomic.Load(&logosWaiterCount) == 0 {
		atomic.Store(&pd.logosCancel, 0)
		return
	}
	lock(&logosWaitersLock)
	last := len(logosWaiters)
	for i := 0; i < last; {
		if logosWaiters[i].pd == pd {
			last--
			logosWaiters[i] = logosWaiters[last]
			logosWaiters[last] = logosWaiter{}
			continue
		}
		i++
	}
	logosWaiters = logosWaiters[:last]
	logosWaitersSync()
	atomic.Store(&pd.logosCancel, 0)
	unlock(&logosWaitersLock)
}

//go:linkname logosCancelGroup
func logosCancelGroup(slot uint32) {
	var toRun gList
	var delta int32
	lock(&logosWaitersLock)
	logosSlotCancelSet(slot)
	for i := range logosWaiters {
		w := &logosWaiters[i]
		if w.group != slot {
			continue
		}
		pd := w.pd
		lock(&pd.lock)
		atomic.Or32(&pd.logosCancel, logosCancelBit(w.mode))
		if gp := netpollunblock(pd, w.mode, false, &delta); gp != nil {
			toRun.push(gp)
		}
		unlock(&pd.lock)
	}
	unlock(&logosWaitersLock)
	for !toRun.empty() {
		netpollgoready(toRun.pop(), 0)
	}
	netpollAdjustWaiters(delta)
}
