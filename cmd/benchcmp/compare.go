// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

// BenchCmp is a pair of benchmarks.
type BenchCmp struct {
	Before *Bench
	After  *Bench
}

// Correlate correlates benchmarks from two BenchSets.
func Correlate(before, after BenchSet) (cmps []BenchCmp, warnings []string) {
	cmps = make([]BenchCmp, 0, len(after))
	for name, beforebb := range before {
		afterbb := after[name]
		if len(beforebb) != len(afterbb) {
			warnings = append(warnings, fmt.Sprintf("ignoring %s: before has %d instances, after has %d", name, len(beforebb), len(afterbb)))
			continue
		}
		for i, beforeb := range beforebb {
			afterb := afterbb[i]
			cmps = append(cmps, BenchCmp{beforeb, afterb})
		}
	}
	return
}

func (c BenchCmp) Name() string           { return c.Before.Name }
func (c BenchCmp) String() string         { return fmt.Sprintf("<%s, %s>", c.Before, c.After) }
func (c BenchCmp) Measured(flag int) bool { return c.Before.Measured&c.After.Measured&flag != 0 }
func (c BenchCmp) DeltaNsOp() Delta       { return Delta{c.Before.NsOp, c.After.NsOp} }
func (c BenchCmp) DeltaMbS() Delta        { return Delta{c.Before.MbS, c.After.MbS} }
func (c BenchCmp) DeltaBOp() Delta        { return Delta{float64(c.Before.BOp), float64(c.After.BOp)} }
func (c BenchCmp) DeltaAllocsOp() Delta {
	return Delta{float64(c.Before.AllocsOp), float64(c.After.AllocsOp)}
}

// Delta is the before and after value for a benchmark measurement.
// Both must be non-negative.
type Delta struct {
	Before float64
	After  float64
}

// mag calculates the magnitude of a change, regardless of the direction of
// the change. mag is intended for sorting and has no independent meaning.
func (d Delta) mag() float64 {
	switch {
	case d.Before != 0 && d.After != 0 && d.Before >= d.After:
		return d.After / d.Before
	case d.Before != 0 && d.After != 0 && d.Before < d.After:
		return d.Before / d.After
	case d.Before == 0 && d.After == 0:
		return 1
	default:
		// 0 -> 1 or 1 -> 0
		// These are significant changes and worth surfacing.
		return math.Inf(1)
	}
}

// Changed reports whether the benchmark quantities are different.
func (d Delta) Changed() bool { return d.Before != d.After }

// Float64 returns After / Before. If Before is 0, Float64 returns
// 1 if After is also 0, and +Inf otherwise.
func (d Delta) Float64() float64 {
	switch {
	case d.Before != 0:
		return d.After / d.Before
	case d.After == 0:
		return 1
	default:
		return math.Inf(1)
	}
}

// Percent formats a Delta as a percent change, ranging from -100% up.
func (d Delta) Percent() string {
	return fmt.Sprintf("%+.2f%%", 100*d.Float64()-100)
}

// Multiple formats a Delta as a multiplier, ranging from 0.00x up.
func (d Delta) Multiple() string {
	return fmt.Sprintf("%.2fx", d.Float64())
}

func (d Delta) String() string {
	return fmt.Sprintf("Δ(%f, %f)", d.Before, d.After)
}

// ByParseOrder sorts BenchCmps to match the order in
// which the Before benchmarks were presented to Parse.
type ByParseOrder []BenchCmp

func (x ByParseOrder) Len() int           { return len(x) }
func (x ByParseOrder) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x ByParseOrder) Less(i, j int) bool { return x[i].Before.ord < x[j].Before.ord }

// lessByDelta provides lexicographic ordering:
//   * largest delta by magnitude
//   * alphabetic by name
func lessByDelta(i, j BenchCmp, calcDelta func(BenchCmp) Delta) bool {
	iDelta, jDelta := calcDelta(i).mag(), calcDelta(j).mag()
	if iDelta != jDelta {
		return iDelta < jDelta
	}
	return i.Name() < j.Name()
}

// ByDeltaNsOp sorts BenchCmps lexicographically by change
// in ns/op, descending, then by benchmark name.
type ByDeltaNsOp []BenchCmp

func (x ByDeltaNsOp) Len() int           { return len(x) }
func (x ByDeltaNsOp) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x ByDeltaNsOp) Less(i, j int) bool { return lessByDelta(x[i], x[j], BenchCmp.DeltaNsOp) }

// ByDeltaMbS sorts BenchCmps lexicographically by change
// in MB/s, descending, then by benchmark name.
type ByDeltaMbS []BenchCmp

func (x ByDeltaMbS) Len() int           { return len(x) }
func (x ByDeltaMbS) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x ByDeltaMbS) Less(i, j int) bool { return lessByDelta(x[i], x[j], BenchCmp.DeltaMbS) }

// ByDeltaBOp sorts BenchCmps lexicographically by change
// in B/op, descending, then by benchmark name.
type ByDeltaBOp []BenchCmp

func (x ByDeltaBOp) Len() int           { return len(x) }
func (x ByDeltaBOp) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x ByDeltaBOp) Less(i, j int) bool { return lessByDelta(x[i], x[j], BenchCmp.DeltaBOp) }

// ByDeltaAllocsOp sorts BenchCmps lexicographically by change
// in allocs/op, descending, then by benchmark name.
type ByDeltaAllocsOp []BenchCmp

func (x ByDeltaAllocsOp) Len() int           { return len(x) }
func (x ByDeltaAllocsOp) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x ByDeltaAllocsOp) Less(i, j int) bool { return lessByDelta(x[i], x[j], BenchCmp.DeltaAllocsOp) }
