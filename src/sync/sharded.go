package sync

import (
	"internal/cpu"
)

// Sharded is a container of values of the same type
// have n data of type T.
//
// On a best effort basis, the runtime will strongly associate a given value
// with a CPU core.  That is to say, retrieving a value twice on the same CPU
// core will return the same value with high probablity.  Note that the runtime
// cannot guarantee this fact, and clients must assume that retrieved values
// can be shared between concurrently executing goroutines.
//
// it is safe to call all methods from multiple goroutines.
//
// zero value is unavailable
// must be initialized use [NewSharded]
type Sharded[T any] struct {
	data []value[T]
}

type value[T any] struct {
	_ cpu.CacheLinePad // prevent false sharing
	v T
	_ cpu.CacheLinePad // prevent false sharing
}

// NewSharded make a [Sharded]
//
// must n != 0 , otherwise panic will occur
func NewSharded[T any](n uint) Sharded[T] {
	if n == 0 {
		panic("sync: create zero value Sharded")
	}
	return Sharded[T]{
		data: make([]value[T], n),
	}
}

// Get get one of n data of type T from [Sharded]
func (s *Sharded[T]) Get() *T {
	// If zero value is use, slice access is out of bounds.
	return &s.data[int(pid())%len(s.data)].v
}

// Range Call f for all data of type T in [Sharded]
func (s *Sharded[T]) Range(f func(*T)) {
	if len(s.data) == 0 {
		panic("sync: use no initialized of Sharded")
	}
	for i := range s.data {
		f(&s.data[i].v)
	}
}

// Implemented in runtime/runtime1.go
func pid() uint64
