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
	waiters = nil
}

func TestListBugs(t *testing.T) {
	defer resetForTesting()

	Report("bad", nil)

	wantBugs(t, "bad")

	for i := 0; i < 3; i++ {
		Report(fmt.Sprintf("index:%d", i), nil)
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

func TestBugNotification(t *testing.T) {
	defer resetForTesting()

	Report("unseen", nil)

	notify1 := Notify()
	notify2 := Notify()

	Report("seen", Data{"answer": 42})

	for _, got := range []Bug{<-notify1, <-notify2} {
		if got, want := got.Description, "seen"; got != want {
			t.Errorf("Saw bug %q, want %q", got, want)
		}
		if got, want := got.Data["answer"], 42; got != want {
			t.Errorf(`bug.Data["answer"] = %v, want %v`, got, want)
		}
	}
}
