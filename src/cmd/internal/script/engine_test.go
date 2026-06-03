// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"context"
	"slices"
	"testing"
)

func FuzzQuoteArgs(f *testing.F) {
	state, err := NewState(context.Background(), f.TempDir(), nil /* env */)
	if err != nil {
		f.Fatalf("failed to create state: %v", err)
	}

	f.Add("foo")
	f.Add(`"foo"`)
	f.Add(`'foo'`)
	f.Fuzz(func(t *testing.T, s string) {
		give := []string{s}
		quoted := quoteArgs(give)
		cmd, err := parse("file.txt", 42, "cmd "+quoted)
		if err != nil {
			t.Fatalf("quoteArgs(%q) = %q cannot be parsed: %v", give, quoted, err)
		}
		args := expandArgs(state, cmd.rawArgs, nil /* regexpArgs */)

		if !slices.Equal(give, args) {
			t.Fatalf("quoteArgs failed to round-trip.\ninput:\n\t%#q\nquoted:\n\t%q\nparsed:\n\t%#q", give, quoted, args)
		}
	})
}
