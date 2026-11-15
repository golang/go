// errorcheck

package p

import "sync"

type stats struct {
	mu sync.Mutex

	//logos:guardedby mu
	total int
}

func movedUnlock(s *stats) {
	s.mu.Lock()
	defer func() {
		_ = s.total // ERROR "\[concurrency::unguarded_access\]"
	}()
	defer s.mu.Unlock()
}
