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

func lockRankRelease(l *mutex) {
	unlock2(l)
}

func lockWithRank(l *mutex, rank lockRank) {
	lock2(l)
}

func lockWithRankMayAcquire(l *mutex, rank lockRank) {
}
