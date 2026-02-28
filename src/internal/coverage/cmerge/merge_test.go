// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmerge_test

import (
	"fmt"
	"internal/coverage"
	"internal/coverage/cmerge"
	"testing"
)

func TestClash(t *testing.T) {
	m := &cmerge.Merger{}
	err := m.SetModeAndGranularity("mdf1.data", coverage.CtrModeSet, coverage.CtrGranularityPerBlock)
	if err != nil {
		t.Fatalf("unexpected clash: %v", err)
	}
	err = m.SetModeAndGranularity("mdf1.data", coverage.CtrModeSet, coverage.CtrGranularityPerBlock)
	if err != nil {
		t.Fatalf("unexpected clash: %v", err)
	}
	err = m.SetModeAndGranularity("mdf1.data", coverage.CtrModeCount, coverage.CtrGranularityPerBlock)
	if err == nil {
		t.Fatalf("expected mode clash, not found")
	}
	err = m.SetModeAndGranularity("mdf1.data", coverage.CtrModeSet, coverage.CtrGranularityPerFunc)
	if err == nil {
		t.Fatalf("expected granularity clash, not found")
	}
	m.SetModeMergePolicy(cmerge.ModeMergeRelaxed)
	err = m.SetModeAndGranularity("mdf1.data", coverage.CtrModeCount, coverage.CtrGranularityPerBlock)
	if err != nil {
		t.Fatalf("unexpected clash: %v", err)
	}
	err = m.SetModeAndGranularity("mdf1.data", coverage.CtrModeSet, coverage.CtrGranularityPerBlock)
	if err != nil {
		t.Fatalf("unexpected clash: %v", err)
	}
	err = m.SetModeAndGranularity("mdf1.data", coverage.CtrModeAtomic, coverage.CtrGranularityPerBlock)
	if err != nil {
		t.Fatalf("unexpected clash: %v", err)
	}
	m.ResetModeAndGranularity()
	err = m.SetModeAndGranularity("mdf1.data", coverage.CtrModeCount, coverage.CtrGranularityPerFunc)
	if err != nil {
		t.Fatalf("unexpected clash after reset: %v", err)
	}
}

func TestBasic(t *testing.T) {
	scenarios := []struct {
		cmode         coverage.CounterMode
		cgran         coverage.CounterGranularity
		src, dst, res []uint32
		iters         int
		merr          bool
		overflow      bool
	}{
		{
			cmode:    coverage.CtrModeSet,
			cgran:    coverage.CtrGranularityPerBlock,
			src:      []uint32{1, 0, 1},
			dst:      []uint32{1, 1, 0},
			res:      []uint32{1, 1, 1},
			iters:    2,
			overflow: false,
		},
		{
			cmode:    coverage.CtrModeCount,
			cgran:    coverage.CtrGranularityPerBlock,
			src:      []uint32{1, 0, 3},
			dst:      []uint32{5, 7, 0},
			res:      []uint32{6, 7, 3},
			iters:    1,
			overflow: false,
		},
		{
			cmode:    coverage.CtrModeCount,
			cgran:    coverage.CtrGranularityPerBlock,
			src:      []uint32{4294967200, 0, 3},
			dst:      []uint32{4294967001, 7, 0},
			res:      []uint32{4294967295, 7, 3},
			iters:    1,
			overflow: true,
		},
	}

	for k, scenario := range scenarios {
		var err error
		var ovf bool
		m := &cmerge.Merger{}
		mdf := fmt.Sprintf("file%d", k)
		err = m.SetModeAndGranularity(mdf, scenario.cmode, scenario.cgran)
		if err != nil {
			t.Fatalf("case %d SetModeAndGranularity failed: %v", k, err)
		}
		for i := 0; i < scenario.iters; i++ {
			err, ovf = m.MergeCounters(scenario.dst, scenario.src)
			if ovf != scenario.overflow {
				t.Fatalf("case %d overflow mismatch: got %v want %v", k, ovf, scenario.overflow)
			}
			if !scenario.merr && err != nil {
				t.Fatalf("case %d unexpected err %v", k, err)
			}
			if scenario.merr && err == nil {
				t.Fatalf("case %d expected err, not received", k)
			}
			for i := range scenario.dst {
				if scenario.dst[i] != scenario.res[i] {
					t.Fatalf("case %d: bad merge at %d got %d want %d",
						k, i, scenario.dst[i], scenario.res[i])
				}
			}
		}
	}
}
