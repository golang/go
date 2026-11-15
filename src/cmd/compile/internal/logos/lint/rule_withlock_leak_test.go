// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"testing"
)

func TestWithlockLeakRegistered(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	r, ok := Lookup("correctness::withlock_leak")
	if !ok || !r.Go || r.Group != Correctness || r.Default != Deny || r.Check == nil {
		t.Fatalf("registry entry = %+v, %v", r, ok)
	}
	d := DefaultPolicy()
	if got := d.Level("correctness::withlock_leak", KindLgo); got != Deny {
		t.Errorf("default Level(lgo) = %v, want deny", got)
	}
	if got := d.Level("correctness::withlock_leak", KindGo); got != Deny {
		t.Errorf("default Level(go) = %v, want deny", got)
	}
}
