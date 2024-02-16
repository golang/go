// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmerge

// package cmerge provides a few small utility APIs for helping
// with merging of counter data for a given function.

import (
	"fmt"
	"internal/coverage"
	"math"
)

type ModeMergePolicy uint8

const (
	ModeMergeStrict ModeMergePolicy = iota
	ModeMergeRelaxed
)

// Merger provides state and methods to help manage the process of
// merging together coverage counter data for a given function, for
// tools that need to implicitly merge counter as they read multiple
// coverage counter data files.
type Merger struct {
	cmode    coverage.CounterMode
	cgran    coverage.CounterGranularity
	policy   ModeMergePolicy
	overflow bool
}

func (cm *Merger) SetModeMergePolicy(policy ModeMergePolicy) {
	cm.policy = policy
}

// MergeCounters takes the counter values in 'src' and merges them
// into 'dst' according to the correct counter mode.
func (m *Merger) MergeCounters(dst, src []uint32) (error, bool) {
	if len(src) != len(dst) {
		return fmt.Errorf("merging counters: len(dst)=%d len(src)=%d", len(dst), len(src)), false
	}
	if m.cmode == coverage.CtrModeSet {
		for i := 0; i < len(src); i++ {
			if src[i] != 0 {
				dst[i] = 1
			}
		}
	} else {
		for i := 0; i < len(src); i++ {
			dst[i] = m.SaturatingAdd(dst[i], src[i])
		}
	}
	ovf := m.overflow
	m.overflow = false
	return nil, ovf
}

// Saturating add does a saturating addition of 'dst' and 'src',
// returning added value or math.MaxUint32 if there is an overflow.
// Overflows are recorded in case the client needs to track them.
func (m *Merger) SaturatingAdd(dst, src uint32) uint32 {
	result, overflow := SaturatingAdd(dst, src)
	if overflow {
		m.overflow = true
	}
	return result
}

// Saturating add does a saturating addition of 'dst' and 'src',
// returning added value or math.MaxUint32 plus an overflow flag.
func SaturatingAdd(dst, src uint32) (uint32, bool) {
	d, s := uint64(dst), uint64(src)
	sum := d + s
	overflow := false
	if uint64(uint32(sum)) != sum {
		overflow = true
		sum = math.MaxUint32
	}
	return uint32(sum), overflow
}

// SetModeAndGranularity records the counter mode and granularity for
// the current merge. In the specific case of merging across coverage
// data files from different binaries, where we're combining data from
// more than one meta-data file, we need to check for and resolve
// mode/granularity clashes.
func (cm *Merger) SetModeAndGranularity(mdf string, cmode coverage.CounterMode, cgran coverage.CounterGranularity) error {
	if cm.cmode == coverage.CtrModeInvalid {
		// Set merger mode based on what we're seeing here.
		cm.cmode = cmode
		cm.cgran = cgran
	} else {
		// Granularity clashes are always errors.
		if cm.cgran != cgran {
			return fmt.Errorf("counter granularity clash while reading meta-data file %s: previous file had %s, new file has %s", mdf, cm.cgran.String(), cgran.String())
		}
		// Mode clashes are treated as errors if we're using the
		// default strict policy.
		if cm.cmode != cmode {
			if cm.policy == ModeMergeStrict {
				return fmt.Errorf("counter mode clash while reading meta-data file %s: previous file had %s, new file has %s", mdf, cm.cmode.String(), cmode.String())
			}
			// In the case of a relaxed mode merge policy, upgrade
			// mode if needed.
			if cm.cmode < cmode {
				cm.cmode = cmode
			}
		}
	}
	return nil
}

func (cm *Merger) ResetModeAndGranularity() {
	cm.cmode = coverage.CtrModeInvalid
	cm.cgran = coverage.CtrGranularityInvalid
	cm.overflow = false
}

func (cm *Merger) Mode() coverage.CounterMode {
	return cm.cmode
}

func (cm *Merger) Granularity() coverage.CounterGranularity {
	return cm.cgran
}
