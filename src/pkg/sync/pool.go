// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

// A Pool is a set of temporary objects that may be individually saved
// and retrieved.
//
// Any item stored in the Pool may be removed automatically by the
// implementation at any time without notification.
// If the Pool holds the only reference when this happens, the item
// might be deallocated.
//
// A Pool is safe for use by multiple goroutines simultaneously.
//
// Pool's intended use is for free lists maintained in global variables,
// typically accessed by multiple goroutines simultaneously. Using a
// Pool instead of a custom free list allows the runtime to reclaim
// entries from the pool when it makes sense to do so. An
// appropriate use of sync.Pool is to create a pool of temporary buffers
// shared between independent clients of a global resource. On the
// other hand, if a free list is maintained as part of an object used
// only by a single client and freed when the client completes,
// implementing that free list as a Pool is not appropriate.
//
// This is an experimental type and might not be released.
type Pool struct {
	next *Pool         // for use by runtime. must be first.
	list []interface{} // offset known to runtime
	mu   Mutex         // guards list

	// New optionally specifies a function to generate
	// a value when Get would otherwise return nil.
	// It may not be changed concurrently with calls to Get.
	New func() interface{}
}

func runtime_registerPool(*Pool)

// Put adds x to the pool.
func (p *Pool) Put(x interface{}) {
	if x == nil {
		return
	}
	p.mu.Lock()
	if p.list == nil {
		runtime_registerPool(p)
	}
	p.list = append(p.list, x)
	p.mu.Unlock()
}

// Get selects an arbitrary item from the Pool, removes it from the
// Pool, and returns it to the caller.
// Get may choose to ignore the pool and treat it as empty.
// Callers should not assume any relation between values passed to Put and
// the values returned by Get.
//
// If Get would otherwise return nil and p.New is non-nil, Get returns
// the result of calling p.New.
func (p *Pool) Get() interface{} {
	p.mu.Lock()
	var x interface{}
	if n := len(p.list); n > 0 {
		x = p.list[n-1]
		p.list[n-1] = nil // Just to be safe
		p.list = p.list[:n-1]
	}
	p.mu.Unlock()
	if x == nil && p.New != nil {
		x = p.New()
	}
	return x
}
