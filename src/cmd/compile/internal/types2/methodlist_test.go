// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"testing"
)

func TestLazyMethodList(t *testing.T) {
	l := newLazyMethodList(2)

	if got := l.Len(); got != 2 {
		t.Fatalf("Len() = %d, want 2", got)
	}

	f0 := NewFunc(nopos, nil, "f0", nil)
	f1 := NewFunc(nopos, nil, "f1", nil)

	// Verify that methodList.At is idempotent, by calling it repeatedly with a
	// resolve func that returns different pointer values (f0 or f1).
	steps := []struct {
		index   int
		resolve *Func // the *Func returned by the resolver
		want    *Func // the actual *Func returned by methodList.At
	}{
		{0, f0, f0},
		{0, f1, f0},
		{1, f1, f1},
		{1, f0, f1},
	}

	for i, step := range steps {
		got := l.At(step.index, func() *Func { return step.resolve })
		if got != step.want {
			t.Errorf("step %d: At(%d, ...) = %s, want %s", i, step.index, got.Name(), step.want.Name())
		}
	}
}
