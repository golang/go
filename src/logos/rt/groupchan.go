// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"iter"
)

// This file implements the M8-S3 channel helpers (SPEC Sec 2.6, T-206):
// range over a channel in a group child iterates until close or child
// cancellation. Cancellation itself arrives through the lowered select
// arms; this helper only has to stop iteration so the post-loop check
// can return the context error.

// Chan adapts ch for `for v := range` in a group child: values flow
// until the channel closes or ctx is done. Either stop ends the range;
// the lowering checks ctx afterwards to report cancellation.
func Chan[T any](ctx context.Context, ch <-chan T) iter.Seq[T] {
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
			case <-ctx.Done():
				return
			}
		}
	}
}
