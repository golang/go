// errorcheck

package p

import "sync"

type stats struct {
	mu sync.Mutex

	//logos:guardedby mu
	total int
}

type rw struct {
	mu sync.RWMutex

	//logos:guardedby mu
	n int
}

var sink = &stats{}

//logos:guardedby gmu
var gtotal int

var gmu sync.Mutex

type pair struct {
	mu sync.Mutex

	//logos:guardedby mu
	n int
}

var arr [4]pair

func okRead(s *stats) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total
}

func okWrite(s *stats) {
	s.mu.Lock()
	s.total = 1
	s.mu.Unlock()
}

func badRead(s *stats) int {
	return s.total // ERROR "\[concurrency::unguarded_access\]"
}

func badWrite(s *stats) {
	s.total = 2 // ERROR "\[concurrency::unguarded_access\]"
}

func badInc(s *stats) {
	s.total++ // ERROR "\[concurrency::unguarded_access\]"
}

func consistentBranches(s *stats, c bool) {
	if c {
		s.mu.Lock()
	} else {
		s.mu.Lock()
	}
	_ = s.total
	if c {
		s.mu.Unlock()
	} else {
		s.mu.Unlock()
	}
}

func inconsistentBranches(s *stats, c bool) {
	if c {
		s.mu.Lock()
	}
	_ = s.total // ERROR "\[concurrency::unguarded_access\]"
	if c {
		s.mu.Unlock()
	}
}

func loopExitCloses(s *stats) {
	s.mu.Lock()
	for i := 0; i < 3; i++ {
		_ = s.total
	}
	s.mu.Unlock()
}

func loopMayNotLock(s *stats) {
	for i := 0; i < 3; i++ {
		if i == 2 {
			s.mu.Lock()
		}
	}
	_ = s.total // ERROR "\[concurrency::unguarded_access\]"
}

func deferHolds(s *stats) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total
}

func explicitUnlockThenRead(s *stats) int {
	s.mu.Lock()
	s.mu.Unlock()
	return s.total // ERROR "\[concurrency::unguarded_access\]"
}

func closureAfterUnlockDefer(s *stats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	defer func() {
		_ = s.total
	}()
}

func closureBeforeUnlockDefer(s *stats) {
	s.mu.Lock()
	defer func() {
		_ = s.total // ERROR "\[concurrency::unguarded_access\]"
	}()
	defer s.mu.Unlock()
}

func okReadUnderReadLock(r *rw) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.n
}

func badWriteUnderReadLock(r *rw) {
	r.mu.RLock()
	r.n = 1 // ERROR "\[concurrency::unguarded_access\]"
	r.mu.RUnlock()
}

func okWriteUnderWriteLock(r *rw) {
	r.mu.Lock()
	r.n = 1
	r.mu.Unlock()
}

func bumpGlobal() {
	gmu.Lock()
	defer gmu.Unlock()
	gtotal++
}

func badGlobal() {
	gtotal++ // ERROR "\[concurrency::unguarded_access\]"
}

type vault struct {
	mu sync.Mutex

	//logos:guardedby mu
	keys int
}

//logos:acquires v.mu
func (v *vault) seal() {}

//logos:releases v.mu
func (v *vault) open() {}

func usesContracts(v *vault) {
	v.seal()
	_ = v.keys
	v.open()
	_ = v.keys // ERROR "\[concurrency::unguarded_access\]"
}

func buildLocal() *stats {
	t := &stats{}
	t.total = 3
	return t
}

func publishThenAccess() {
	t := &stats{}
	sink = t
	t.total = 4 // ERROR "\[concurrency::unguarded_access\]"
}

func tryLockTaken(s *stats) {
	if s.mu.TryLock() {
		_ = s.total
		s.mu.Unlock()
	}
}

func tryLockNegatedElse(s *stats) {
	if !s.mu.TryLock() {
	} else {
		_ = s.total
		s.mu.Unlock()
	}
}

func tryLockStored(s *stats) {
	ok := s.mu.TryLock()
	if ok {
		_ = s.total // ERROR "\[concurrency::unguarded_access\]"
		s.mu.Unlock()
	}
}

func indexLockSame() {
	arr[0].mu.Lock()
	arr[0].n = 1
	arr[0].mu.Unlock()
}

func indexLockVarSame(i int) {
	arr[i].mu.Lock()
	arr[i].n = 2
	arr[i].mu.Unlock()
}

func indexLockOtherElement() {
	arr[0].mu.Lock()
	arr[1].n = 1 // ERROR "\[concurrency::unguarded_access\]"
	arr[0].mu.Unlock()
}

func indexLockOtherVar(i, j int) {
	arr[i].mu.Lock()
	arr[j].n = 3 // ERROR "\[concurrency::unguarded_access\]"
}

func backwardGotoReentry(s *stats, again bool) {
	s.mu.Lock()
L:
	_ = s.total // ERROR "\[concurrency::unguarded_access\]"
	s.mu.Unlock()
	if again {
		goto L
	}
}

func backwardGotoRelocked(s *stats, again bool) {
	s.mu.Lock()
L2:
	_ = s.total
	s.mu.Unlock()
	if again {
		s.mu.Lock()
		goto L2
	}
}

func unlockDeferredInEarlyClosure(s *stats) {
	s.mu.Lock()
	func() {
		defer s.mu.Unlock()
	}()
	_ = s.total // ERROR "\[concurrency::unguarded_access\]"
}

func use(n int) {}

func goLiteralInheritsNoLock(s *stats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	go func() {
		s.total++ // ERROR "\[concurrency::unguarded_access\]"
	}()
}

func goLiteralTakesOwnLock(s *stats) {
	go func() {
		s.mu.Lock()
		s.total++
		s.mu.Unlock()
	}()
}

func goLiteralLockStaysInChild(s *stats) {
	go func() {
		s.mu.Lock()
	}()
	_ = s.total // ERROR "\[concurrency::unguarded_access\]"
}

func goCallArgumentsRunInParent(s *stats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	go use(s.total)
}

func goCallArgumentsUnlocked(s *stats) {
	go use(s.total) // ERROR "\[concurrency::unguarded_access\]"
}

func goLiteralCallArgumentsRunInParent(s *stats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	go func(n int) {
		s.total++ // ERROR "\[concurrency::unguarded_access\]"
	}(s.total)
}

func backwardGotoInClosure(s *stats) {
	s.mu.Lock()
	func() {
	L:
		_ = s.total // ERROR "\[concurrency::unguarded_access\]"
		s.mu.Unlock()
		goto L
	}()
}

func backwardGotoInClosureRelocked(s *stats) {
	s.mu.Lock()
	func() {
	L:
		_ = s.total
		s.mu.Unlock()
		s.mu.Lock()
		goto L
	}()
}

func fiveLabelClosureUnlock(s *stats) {
	func() {
		s.mu.Lock()
		goto E
	A:
		_ = s.total // ERROR "\[concurrency::unguarded_access\]"
		return
		s.mu.Lock()
	B:
		goto A
		s.mu.Lock()
	C:
		goto B
		s.mu.Lock()
	D:
		goto C
		s.mu.Lock()
	E:
		s.mu.Unlock()
		goto D
	}()
}

func fiveLabelClosureRelocked(s *stats) {
	func() {
		s.mu.Lock()
		goto E
	A:
		_ = s.total
		return
		s.mu.Lock()
	B:
		s.mu.Lock()
		goto A
		s.mu.Lock()
	C:
		s.mu.Lock()
		goto B
		s.mu.Lock()
	D:
		s.mu.Lock()
		goto C
		s.mu.Lock()
	E:
		s.mu.Unlock()
		s.mu.Lock()
		goto D
	}()
}

func backwardGotoDeferredClosure(s *stats, again bool) {
	defer func() {
		_ = s.total // ERROR "\[concurrency::unguarded_access\]"
	}()
L:
	if again {
		goto L
	}
}

//logos:requires s.mu
func requireStatsLock(s *stats) {}

func backwardGotoDeferredRequire(s *stats, again bool) {
	defer requireStatsLock(s) // ERROR "\[concurrency::lock_requirement_unmet\]"
L:
	if again {
		goto L
	}
}
