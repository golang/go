// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.staticlockranking

package runtime

// // lockRankStruct is embedded in mutex, but is empty when staticklockranking is
// disabled (the default)
type lockRankStruct struct {
}

func lockInit(l *mutex, rank lockRank) {
}

func getLockRank(l *mutex) lockRank {
	return 0
}

func lockWithRank(l *mutex, rank lockRank) {
	lock2(l)
}

// This function may be called in nosplit context and thus must be nosplit.
//
//go:nosplit
func acquireLockRank(rank lockRank) {
}

func unlockWithRank(l *mutex) {
	unlock2(l)
}

// This function may be called in nosplit context and thus must be nosplit.
//
//go:nosplit
func releaseLockRank(rank lockRank) {
}

func lockWithRankMayAcquire(l *mutex, rank lockRank) {
}

//go:nosplit
func assertLockHeld(l *mutex) {
}

//go:nosplit
func assertRankHeld(r lockRank) {
}

//go:nosplit
func worldStopped() {
}

//go:nosplit
func worldStarted() {
}

//go:nosplit
func assertWorldStopped() {
}

//go:nosplit
func assertWorldStoppedOrLockHeld(l *mutex) {
}
