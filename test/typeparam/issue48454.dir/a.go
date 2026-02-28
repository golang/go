// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "sync"

type Val[T any] struct {
	mu  sync.RWMutex
	val T
}

func (v *Val[T]) Has() {
	v.mu.RLock()
}
