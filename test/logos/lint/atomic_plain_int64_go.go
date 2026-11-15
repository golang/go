// errorcheck -0 --pedantic

package p

import (
	"sync"
	"sync/atomic"
)

var plain int64 // ERROR "\[portability::atomic_plain_int64\]"

var plainU uint64 // ERROR "\[portability::atomic_plain_int64\]"

var safe atomic.Int64

var small int32

type box struct {
	plain int64 // ERROR "\[portability::atomic_plain_int64\]"
	small int32
}

func reads() int64 {
	return atomic.LoadInt64(&plain)
}

func adds() {
	atomic.AddUint64(&plainU, 1)
}

func swaps() {
	atomic.CompareAndSwapInt64(&plain, 0, 1)
}

func safeRead() int64 {
	return safe.Load()
}

func smallRead() int32 {
	return atomic.LoadInt32(&small)
}

func boxRead(b *box) int64 {
	return atomic.LoadInt64(&b.plain)
}

func mutexes() {
	var mu sync.Mutex
	mu.Lock()
	mu.Unlock()
}
