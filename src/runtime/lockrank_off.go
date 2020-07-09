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

func lockWithRank(l *mutex, rank lockRank) {
	lock2(l)
}

func acquireLockRank(rank lockRank) {
}

func unlockWithRank(l *mutex) {
	unlock2(l)
}

func releaseLockRank(rank lockRank) {
}

func lockWithRankMayAcquire(l *mutex, rank lockRank) {
}
