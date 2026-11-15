// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"strconv"
	"sync"
	"testing"
)

func activeUnionTrials() int {
	unionTrials.mu.Lock()
	defer unionTrials.mu.Unlock()
	return len(unionTrials.active)
}

func TestUnionTrialMemoMiss(t *testing.T) {
	raw := []byte(`{"a":1}`)
	trial := EnterUnionTrial(raw)
	defer trial.Exit()
	hit, out := trial.Lookup("trial-memo-test-miss", raw)
	if hit {
		t.Fatalf("unknown key must miss, got %+v", out)
	}
	if out.Kind != UnionTrialMiss || out.Winner != -1 {
		t.Fatalf("miss must report Miss/-1, got %+v", out)
	}
}

func TestUnionTrialMemoCapKeepsRecent(t *testing.T) {
	const id = "trial-memo-test-cap"
	trial := EnterUnionTrial([]byte(`[]`))
	defer trial.Exit()
	for i := range 1100 {
		raw := []byte(`{"n":` + strconv.Itoa(i) + `}`)
		trial.Store(id, raw, UnionTrialOutcome{Kind: UnionTrialNoMatch, Winner: -1})
	}
	if hit, _ := trial.Lookup(id, []byte(`{"n":0}`)); hit {
		t.Fatalf("entries past the cap must be evicted")
	}
	hit, out := trial.Lookup(id, []byte(`{"n":1099}`))
	if !hit || out.Kind != UnionTrialNoMatch {
		t.Fatalf("recent entries must survive the cap reset, got %v %+v", hit, out)
	}
}

func TestUnionTrialNestedValueSharesRootOutcomes(t *testing.T) {
	const id = "trial-memo-test-nested"
	root := []byte(`{"a":{"k":1},"b":{"k":1}}`)
	first, second := root[5:12], root[17:24]
	outer := EnterUnionTrial(root)
	inner := EnterUnionTrial(first)
	inner.Store(id, first, UnionTrialOutcome{Kind: UnionTrialSingle, Winner: 2})
	inner.Exit()
	sibling := EnterUnionTrial(second)
	hit, out := sibling.Lookup(id, second)
	sibling.Exit()
	outer.Exit()
	if !hit || out != (UnionTrialOutcome{Kind: UnionTrialSingle, Winner: 2}) {
		t.Fatalf("equal bytes inside one root value must replay the stored outcome, got %v %+v", hit, out)
	}
}

func TestUnionTrialSeparateBufferSharesNothing(t *testing.T) {
	const id = "trial-memo-test-separate"
	raw := []byte(`{"k":1}`)
	inFlight := EnterUnionTrial(raw)
	defer inFlight.Exit()
	inFlight.Store(id, raw, UnionTrialOutcome{Kind: UnionTrialSingle, Winner: 1})
	var wg sync.WaitGroup
	wg.Go(func() {
		other := []byte(string(raw))
		trial := EnterUnionTrial(other)
		defer trial.Exit()
		if hit, out := trial.Lookup(id, other); hit {
			t.Errorf("a decode of another buffer with equal bytes replayed %+v", out)
		}
	})
	wg.Wait()
}

func TestUnionTrialOutcomesEndWithOutermostTrial(t *testing.T) {
	const id = "trial-memo-test-lifetime"
	raw := []byte(`{"k":1}`)
	before := activeUnionTrials()
	trial := EnterUnionTrial(raw)
	trial.Store(id, raw, UnionTrialOutcome{Kind: UnionTrialSingle, Winner: 1})
	trial.Exit()
	if n := activeUnionTrials(); n != before {
		t.Fatalf("%d root values in flight after the outermost trial returned, want %d", n, before)
	}
	next := EnterUnionTrial(raw)
	defer next.Exit()
	if hit, out := next.Lookup(id, raw); hit {
		t.Fatalf("a later decode of the same bytes replayed %+v", out)
	}
}

func TestUnionTrialConcurrentRootsKeepTheirOwnOutcomes(t *testing.T) {
	const id = "trial-memo-test-concurrent"
	var wg sync.WaitGroup
	for winner := range 64 {
		wg.Go(func() {
			root := []byte(`{"v":{"k":1}}`)
			nested := root[5:12]
			for range 200 {
				outer := EnterUnionTrial(root)
				outer.Store(id, nested, UnionTrialOutcome{Kind: UnionTrialSingle, Winner: winner})
				inner := EnterUnionTrial(nested)
				hit, out := inner.Lookup(id, nested)
				inner.Exit()
				outer.Exit()
				if !hit || out.Winner != winner {
					t.Errorf("root %d read %v %+v", winner, hit, out)
					return
				}
			}
		})
	}
	wg.Wait()
}
