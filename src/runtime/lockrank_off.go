// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !goexperiment.staticlockranking

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

// The following functions may be called in nosplit context.
// Nosplit is not strictly required for lockWithRank, unlockWithRank
// and lockWithRankMayAcquire, but these nosplit annotations must
// be kept consistent with the equivalent functions in lockrank_on.go.

//go:nosplit
func lockWithRank(l *mutex, rank lockRank) {
	lock2(l)
}

//go:nosplit
func acquireLockRank(rank lockRank) {
}

//go:nosplit
func unlockWithRank(l *mutex) {
	unlock2(l)
}

//go:nosplit
func releaseLockRank(rank lockRank) {
}

//go:nosplit
func lockWithRankMayAcquire(l *mutex, rank lockRank) {
}
