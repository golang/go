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

func entries[R any, D Discarder[R]]() map[string]entryFunc[R] {
	return map[string]entryFunc[R]{
		"seq": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupSeq[int, R, D](ctx, limit, intSeq(n), f)
		},
		"seq-unordered": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupSeqUnordered[int, R, D](ctx, limit, intSeq(n), f)
		},
		"chan": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupChan[int, R, D](ctx, limit, intChan(n), f)
		},
		"chan-unordered": func(ctx context.Context, limit, n int, f func(context.Context, int) (R, bool, error)) iter.Seq2[R, error] {
			return GroupChanUnordered[int, R, D](ctx, limit, intChan(n), f)
		},
	}
}

func eachEntry[R any, D Discarder[R]](t *testing.T, body func(t *testing.T, run entryFunc[R])) {
	t.Helper()
	for name, run := range entries[R, D]() {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) { body(t, run) })
		})
	}
}
