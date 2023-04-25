// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug

import (
	"fmt"
	"testing"
)

func resetForTesting() {
	exemplars = nil
	handlers = nil
}

func TestListBugs(t *testing.T) {
	defer resetForTesting()

	Report("bad")

	wantBugs(t, "bad")

	for i := 0; i < 3; i++ {
		Report(fmt.Sprintf("index:%d", i))
	}

	wantBugs(t, "bad", "index:0")
}

func wantBugs(t *testing.T, want ...string) {
	t.Helper()

	bugs := List()
	if got, want := len(bugs), len(want); got != want {
		t.Errorf("List(): got %d bugs, want %d", got, want)
		return
	}

	for i, b := range bugs {
		if got, want := b.Description, want[i]; got != want {
			t.Errorf("bug.List()[%d] = %q, want %q", i, got, want)
		}
	}
}

func TestBugHandler(t *testing.T) {
	defer resetForTesting()

	Report("unseen")

	// Both handlers are called, in order of registration, only once.
	var got string
	Handle(func(b Bug) { got += "1:" + b.Description })
	Handle(func(b Bug) { got += "2:" + b.Description })

	Report("seen")

	Report("again")

	if want := "1:seen2:seen"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
