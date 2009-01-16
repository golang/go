// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

func cas(val *int32, old, new int32) bool
func semacquire(*int32)
func semrelease(*int32)

export type Mutex struct {
	key int32;
	sema int32;
}

func xadd(val *int32, delta int32) (new int32) {
	for {
		v := *val;
		if cas(val, v, v+delta) {
			return v+delta;
		}
	}
	panic("unreached")
}

func (m *Mutex) Lock() {
	if xadd(&m.key, 1) == 1 {
		// changed from 0 to 1; we hold lock
		return;
	}
	semacquire(&m.sema);
}

func (m *Mutex) Unlock() {
	if xadd(&m.key, -1) == 0 {
		// changed from 1 to 0; no contention
		return;
	}
	semrelease(&m.sema);
}

