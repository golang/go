// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"iter"
)

// Chan adapts ch for `for v := range` in a group child: values flow
// until the channel closes or ctx is done. Either stop ends the range;
// the lowering checks ctx afterwards to report cancellation.
func Chan[T any](ctx context.Context, ch <-chan T) iter.Seq[T] {
	var done <-chan struct{}
	if ctx != nil {
		done = ctx.Done()
	}
	return func(yield func(T) bool) {
		for {
			select {
			case v, ok := <-ch:
				if !ok {
					return
				}
				if !yield(v) {
					return
				}
			case <-done:
				return
			}
		}
	}
}

// ElemZero returns the zero value of the element type of ch, so that
// lowered code can declare a temporary of a type it cannot name.
func ElemZero[T any](ch <-chan T) (zero T) {
	return
}
