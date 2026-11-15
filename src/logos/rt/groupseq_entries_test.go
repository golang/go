// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"iter"
	"testing"
	"testing/synctest"
)

type entryFunc[R any] func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error]

func entries[R any]() map[string]entryFunc[R] {
	return map[string]entryFunc[R]{
		"seq": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupSeq(ctx, limit, intSeq(n), f)
		},
		"seq-unordered": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupSeqUnordered(ctx, limit, intSeq(n), f)
		},
		"chan": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupChan(ctx, limit, intChan(n), f)
		},
		"chan-unordered": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupChanUnordered(ctx, limit, intChan(n), f)
		},
	}
}

func eachEntry[R any](t *testing.T, body func(t *testing.T, run entryFunc[R])) {
	t.Helper()
	for name, run := range entries[R]() {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) { body(t, run) })
		})
	}
}
