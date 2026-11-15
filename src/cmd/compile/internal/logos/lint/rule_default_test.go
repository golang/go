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

var specSeverity = map[string]Severity{
	"concurrency::unguarded_access":            Forbid,
	"concurrency::nonatomic_access":            Forbid,
	"concurrency::lock_requirement_unmet":      Forbid,
	"concurrency::concurrent_multiword_race":   Forbid,
	"concurrency::reentrant_withlock_deadlock": Forbid,
	"concurrency::unsupervised_goroutine":      Forbid,
	"correctness::layout_constant_mismatch":    Forbid,
	"correctness::linkname_mismatch":           Forbid,
	"correctness::region_escape":               Forbid,
	"correctness::nil_deref_unproven":          Forbid,
	"correctness::typed_nil_return":            Forbid,
	"correctness::arena_pointer_escape":        Forbid,
	"correctness::defer_erases_errdefer_state": Forbid,
	"correctness::infinite_recursion":          Forbid,
	"correctness::withlock_leak":               Deny,
	"correctness::unclosed_io_closer":          Deny,
	"correctness::unbounded_channel_op":        Deny,
	"slop::err_chain_reset":                    Deny,
	"slop::discarded_error":                    Deny,
	"complexity::cognitive_overflow":           Deny,
	"portability::atomic_plain_int64":          Warn,
	"portability::serialized_unsized_int":      Warn,
	"portability::alias_extension":             Warn,
	"perf::arena_fallback":                     Warn,
	"perf::large_backing_array_pin":            Warn,
	"perf::time_after_in_loop":                 Warn,
	"suspicious::unapplied_struct_default":     Warn,
	"suspicious::side_effect_before_sort":      Warn,
	"pedantic::unsafe_outside_unsafe_file":     Allow,
	"pedantic::exhaustive_union_switch":        Allow,
}

func TestRuleSeverityMatchesSpec(t *testing.T) {
	d := DefaultPolicy()
	for _, r := range Rules() {
		want, ok := specSeverity[r.ID]
		if !ok {
			t.Errorf("rule %s has no SPEC severity: add it to specSeverity", r.ID)
			continue
		}
		if got := d.Level(r.ID, KindLgo); got != want {
			t.Errorf("default Level(%s, lgo) = %v, want %v", r.ID, got, want)
		}
		wantGo := want
		if want == Forbid {
			wantGo = Deny
		}
		if got := d.Level(r.ID, KindGo); got != wantGo {
			t.Errorf("default Level(%s, go) = %v, want %v", r.ID, got, wantGo)
		}
	}
	for id := range specSeverity {
		if _, ok := Lookup(id); !ok {
			t.Errorf("SPEC rule %s is not registered", id)
		}
	}
}
