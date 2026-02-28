// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tracev2_test

import (
	"internal/trace/tracev2"
	"iter"
	"regexp"
	"slices"
	"strings"
	"testing"
)

var argNameRegexp = regexp.MustCompile(`((?P<name>[A-Za-z]+)_)?(?P<type>[A-Za-z]+)`)

func TestSpecs(t *testing.T) {
	if tracev2.NumEvents <= 0 {
		t.Fatalf("no trace events?")
	}
	if tracev2.MaxExperimentalEvent < tracev2.MaxEvent {
		t.Fatalf("max experimental event (%d) is < max event (%d)", tracev2.MaxExperimentalEvent, tracev2.MaxEvent)
	}
	specs := tracev2.Specs()
	for ev := range allEvents() {
		spec := &specs[ev]
		if spec.Name == "" {
			t.Errorf("expected event %d to be defined in specs", ev)
			continue
		}
		if spec.IsTimedEvent && spec.Args[0] != "dt" {
			t.Errorf("%s is a timed event, but its first argument is not 'dt'", spec.Name)
		}
		if spec.HasData && spec.Name != "String" && spec.Name != "ExperimentalBatch" {
			t.Errorf("%s has data, but is not a special kind of event (unsupported, but could be)", spec.Name)
		}
		if spec.IsStack && spec.Name != "Stack" {
			t.Errorf("%s listed as being a stack, but is not the Stack event (unsupported)", spec.Name)
		}
		if spec.IsTimedEvent && len(spec.Args) > tracev2.MaxTimedEventArgs {
			t.Errorf("%s has too many timed event args: have %d, want %d at most", spec.Name, len(spec.Args), tracev2.MaxTimedEventArgs)
		}
		if ev.Experimental() && spec.Experiment == tracev2.NoExperiment {
			t.Errorf("experimental event %s must have an experiment", spec.Name)
		}

		// Check arg types.
		for _, arg := range spec.Args {
			matches := argNameRegexp.FindStringSubmatch(arg)
			if len(matches) == 0 {
				t.Errorf("malformed argument %s for event %s", arg, spec.Name)
			}
		}

		// Check stacks.
		for _, i := range spec.StackIDs {
			if !strings.HasSuffix(spec.Args[i], "stack") {
				t.Errorf("stack argument listed at %d in %s, but argument name %s does not imply stack type", i, spec.Name, spec.Args[i])
			}
		}
		for i, arg := range spec.Args {
			if !strings.HasSuffix(spec.Args[i], "stack") {
				continue
			}
			if !slices.Contains(spec.StackIDs, i) {
				t.Errorf("found stack argument %s in %s at index %d not listed in StackIDs", arg, spec.Name, i)
			}
		}

		// Check strings.
		for _, i := range spec.StringIDs {
			if !strings.HasSuffix(spec.Args[i], "string") {
				t.Errorf("string argument listed at %d in %s, but argument name %s does not imply string type", i, spec.Name, spec.Args[i])
			}
		}
		for i, arg := range spec.Args {
			if !strings.HasSuffix(spec.Args[i], "string") {
				continue
			}
			if !slices.Contains(spec.StringIDs, i) {
				t.Errorf("found string argument %s in %s at index %d not listed in StringIDs", arg, spec.Name, i)
			}
		}
	}
}

func allEvents() iter.Seq[tracev2.EventType] {
	return func(yield func(tracev2.EventType) bool) {
		for ev := tracev2.EvNone + 1; ev < tracev2.NumEvents; ev++ {
			if !yield(ev) {
				return
			}
		}
		for ev := tracev2.MaxEvent + 1; ev < tracev2.NumExperimentalEvents; ev++ {
			if !yield(ev) {
				return
			}
		}
	}
}
