// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import "testing"

// denyInForbidGroup are the rules SPEC Sec 1.2 sets at deny although their
// group, correctness, defaults to forbid.
var denyInForbidGroup = []string{
	"correctness::unclosed_io_closer",
	"correctness::unbounded_channel_op",
}

// TestRuleDefaultSeverity checks the rule-level default: deny for the two
// rules of the forbid group that SPEC sets at deny, forbid for the rest of the
// group, and a group configured forbid raising both.
func TestRuleDefaultSeverity(t *testing.T) {
	d := DefaultPolicy()
	for _, id := range denyInForbidGroup {
		if got := d.Level(id, KindLgo); got != Deny {
			t.Errorf("default Level(%s, lgo) = %v, want deny", id, got)
		}
		if got := d.Level(id, KindGo); got != Deny {
			t.Errorf("default Level(%s, go) = %v, want deny", id, got)
		}
	}
	if got := d.Level("correctness::arena_pointer_escape", KindLgo); got != Forbid {
		t.Errorf("default Level(arena_pointer_escape, lgo) = %v, want forbid", got)
	}
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"correctness":"forbid"}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range denyInForbidGroup {
		if got := p.Level(id, KindLgo); got != Forbid {
			t.Errorf("correctness=forbid Level(%s, lgo) = %v, want forbid", id, got)
		}
		if got := p.Level(id, KindGo); got != Deny {
			t.Errorf("correctness=forbid Level(%s, go) = %v, want deny", id, got)
		}
	}
}
