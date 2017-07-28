// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"container/heap"
	"math"
	"sort"
	"strings"
	"time"
)

// MutatorUtil is a change in mutator utilization at a particular
// time. Mutator utilization functions are represented as a
// time-ordered []MutatorUtil.
type MutatorUtil struct {
	Time int64
	// Util is the mean mutator utilization starting at Time. This
	// is in the range [0, 1].
	Util float64
}

// UtilFlags controls the behavior of MutatorUtilization.
type UtilFlags int

const (
	// UtilSTW means utilization should account for STW events.
	UtilSTW UtilFlags = 1 << iota
	// UtilBackground means utilization should account for
	// background mark workers.
	UtilBackground
	// UtilAssist means utilization should account for mark
	// assists.
	UtilAssist
	// UtilSweep means utilization should account for sweeping.
	UtilSweep
)

// MutatorUtilization returns the mutator utilization function for the
// given trace. This function will always end with 0 utilization. The
// bounds of the function are implicit in the first and last event;
// outside of these bounds the function is undefined.
func (p *Parsed) MutatorUtilization(flags UtilFlags) []MutatorUtil {
	events := p.Events
	if len(events) == 0 {
		return nil
	}

	gomaxprocs, gcPs, stw := 1, 0, 0
	out := []MutatorUtil{{events[0].Ts, 1}}
	assists := map[uint64]bool{}
	block := map[uint64]*Event{}
	bgMark := map[uint64]bool{}
	for _, ev := range events {
		switch ev.Type {
		case EvGomaxprocs:
			gomaxprocs = int(ev.Args[0])
		case EvGCSTWStart:
			if flags&UtilSTW != 0 {
				stw++
			}
		case EvGCSTWDone:
			if flags&UtilSTW != 0 {
				stw--
			}
		case EvGCMarkAssistStart:
			if flags&UtilAssist != 0 {
				gcPs++
				assists[ev.G] = true
			}
		case EvGCMarkAssistDone:
			if flags&UtilAssist != 0 {
				gcPs--
				delete(assists, ev.G)
			}
		case EvGCSweepStart:
			if flags&UtilSweep != 0 {
				gcPs++
			}
		case EvGCSweepDone:
			if flags&UtilSweep != 0 {
				gcPs--
			}
		case EvGoStartLabel:
			if flags&UtilBackground != 0 && strings.HasPrefix(ev.SArgs[0], "GC ") && ev.SArgs[0] != "GC (idle)" {
				// Background mark worker.
				bgMark[ev.G] = true
				gcPs++
			}
			fallthrough
		case EvGoStart:
			if assists[ev.G] {
				// Unblocked during assist.
				gcPs++
			}
			block[ev.G] = ev.Link
		default:
			if ev != block[ev.G] {
				continue
			}

			if assists[ev.G] {
				// Blocked during assist.
				gcPs--
			}
			if bgMark[ev.G] {
				// Background mark worker done.
				gcPs--
				delete(bgMark, ev.G)
			}
			delete(block, ev.G)
		}

		ps := gcPs
		if stw > 0 {
			ps = gomaxprocs
		}
		mu := MutatorUtil{ev.Ts, 1 - float64(ps)/float64(gomaxprocs)}
		if mu.Util == out[len(out)-1].Util {
			// No change.
			continue
		}
		if mu.Time == out[len(out)-1].Time {
			// Take the lowest utilization at a time stamp.
			if mu.Util < out[len(out)-1].Util {
				out[len(out)-1] = mu
			}
		} else {
			out = append(out, mu)
		}
	}

	// Add final 0 utilization event. This is important to mark
	// the end of the trace. The exact value shouldn't matter
	// since no window should extend beyond this, but using 0 is
	// symmetric with the start of the trace.
	endTime := events[len(events)-1].Ts
	if out[len(out)-1].Time == endTime {
		out[len(out)-1].Util = 0
	} else {
		out = append(out, MutatorUtil{endTime, 0})
	}

	return out
}

// totalUtil is total utilization, measured in nanoseconds. This is a
// separate type primarily to distinguish it from mean utilization,
// which is also a float64.
type totalUtil float64

func totalUtilOf(meanUtil float64, dur int64) totalUtil {
	return totalUtil(meanUtil * float64(dur))
}

// mean returns the mean utilization over dur.
func (u totalUtil) mean(dur time.Duration) float64 {
	return float64(u) / float64(dur)
}

// An MMUCurve is the minimum mutator utilization curve across
// multiple window sizes.
type MMUCurve struct {
	util []MutatorUtil
	// sums[j] is the cumulative sum of util[:j].
	sums []totalUtil
	// bands summarizes util in non-overlapping bands of duration
	// bandDur.
	bands []mmuBand
	// bandDur is the duration of each band.
	bandDur int64
}

type mmuBand struct {
	// minUtil is the minimum instantaneous mutator utilization in
	// this band.
	minUtil float64
	// cumUtil is the cumulative total mutator utilization between
	// time 0 and the left edge of this band.
	cumUtil totalUtil

	// integrator is the integrator for the left edge of this
	// band.
	integrator integrator
}

// NewMMUCurve returns an MMU curve for the given mutator utilization
// function.
func NewMMUCurve(util []MutatorUtil) *MMUCurve {
	// Compute cumulative sum.
	sums := make([]totalUtil, len(util))
	var prev MutatorUtil
	var sum totalUtil
	for j, u := range util {
		sum += totalUtilOf(prev.Util, u.Time-prev.Time)
		sums[j] = sum
		prev = u
	}

	// Divide the utilization curve up into equal size
	// non-overlapping "bands" and compute a summary for each of
	// these bands.
	//
	// Compute the duration of each band.
	numBands := 1000
	if numBands > len(util) {
		// There's no point in having lots of bands if there
		// aren't many events.
		numBands = len(util)
	}
	dur := util[len(util)-1].Time - util[0].Time
	bandDur := (dur + int64(numBands) - 1) / int64(numBands)
	if bandDur < 1 {
		bandDur = 1
	}
	// Compute the bands. There are numBands+1 bands in order to
	// record the final cumulative sum.
	bands := make([]mmuBand, numBands+1)
	c := MMUCurve{util, sums, bands, bandDur}
	leftSum := integrator{&c, 0}
	for i := range bands {
		startTime, endTime := c.bandTime(i)
		cumUtil := leftSum.advance(startTime)
		predIdx := leftSum.pos
		minUtil := 1.0
		for i := predIdx; i < len(util) && util[i].Time < endTime; i++ {
			minUtil = math.Min(minUtil, util[i].Util)
		}
		bands[i] = mmuBand{minUtil, cumUtil, leftSum}
	}

	return &c
}

func (c *MMUCurve) bandTime(i int) (start, end int64) {
	start = int64(i)*c.bandDur + c.util[0].Time
	end = start + c.bandDur
	return
}

type bandUtil struct {
	// Band index
	i int
	// Lower bound of mutator utilization for all windows
	// with a left edge in this band.
	utilBound float64
}

type bandUtilHeap []bandUtil

func (h bandUtilHeap) Len() int {
	return len(h)
}

func (h bandUtilHeap) Less(i, j int) bool {
	return h[i].utilBound < h[j].utilBound
}

func (h bandUtilHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *bandUtilHeap) Push(x interface{}) {
	*h = append(*h, x.(bandUtil))
}

func (h *bandUtilHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}

// UtilWindow is a specific window at Time.
type UtilWindow struct {
	Time int64
	// MutatorUtil is the mean mutator utilization in this window.
	MutatorUtil float64
}

type utilHeap []UtilWindow

func (h utilHeap) Len() int {
	return len(h)
}

func (h utilHeap) Less(i, j int) bool {
	if h[i].MutatorUtil != h[j].MutatorUtil {
		return h[i].MutatorUtil > h[j].MutatorUtil
	}
	return h[i].Time > h[j].Time
}

func (h utilHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *utilHeap) Push(x interface{}) {
	*h = append(*h, x.(UtilWindow))
}

func (h *utilHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}

// An accumulator collects different MMU-related statistics depending
// on what's desired.
type accumulator struct {
	mmu float64

	// bound is the mutator utilization bound where adding any
	// mutator utilization above this bound cannot affect the
	// accumulated statistics.
	bound float64

	// Worst N window tracking
	nWorst int
	wHeap  utilHeap
}

// addMU records mutator utilization mu over the given window starting
// at time.
//
// It returns true if further calls to addMU would be pointless.
func (acc *accumulator) addMU(time int64, mu float64, window time.Duration) bool {
	if mu < acc.mmu {
		acc.mmu = mu
	}
	acc.bound = acc.mmu

	if acc.nWorst == 0 {
		// If the minimum has reached zero, it can't go any
		// lower, so we can stop early.
		return mu == 0
	}

	// Consider adding this window to the n worst.
	if len(acc.wHeap) < acc.nWorst || mu < acc.wHeap[0].MutatorUtil {
		// This window is lower than the K'th worst window.
		//
		// Check if there's any overlapping window
		// already in the heap and keep whichever is
		// worse.
		for i, ui := range acc.wHeap {
			if time+int64(window) > ui.Time && ui.Time+int64(window) > time {
				if ui.MutatorUtil <= mu {
					// Keep the first window.
					goto keep
				} else {
					// Replace it with this window.
					heap.Remove(&acc.wHeap, i)
					break
				}
			}
		}

		heap.Push(&acc.wHeap, UtilWindow{time, mu})
		if len(acc.wHeap) > acc.nWorst {
			heap.Pop(&acc.wHeap)
		}
	keep:
	}
	if len(acc.wHeap) < acc.nWorst {
		// We don't have N windows yet, so keep accumulating.
		acc.bound = 1.0
	} else {
		// Anything above the least worst window has no effect.
		acc.bound = math.Max(acc.bound, acc.wHeap[0].MutatorUtil)
	}

	// If we've found enough 0 utilizations, we can stop immediately.
	return len(acc.wHeap) == acc.nWorst && acc.wHeap[0].MutatorUtil == 0
}

// MMU returns the minimum mutator utilization for the given time
// window. This is the minimum utilization for all windows of this
// duration across the execution. The returned value is in the range
// [0, 1].
func (c *MMUCurve) MMU(window time.Duration) (mmu float64) {
	acc := accumulator{mmu: 1.0, bound: 1.0}
	c.mmu(window, &acc)
	return acc.mmu
}

// Examples returns n specific examples of the lowest mutator
// utilization for the given window size. The returned windows will be
// disjoint (otherwise there would be a huge number of
// mostly-overlapping windows at the single lowest point). There are
// no guarantees on which set of disjoint windows this returns.
func (c *MMUCurve) Examples(window time.Duration, n int) (worst []UtilWindow) {
	acc := accumulator{mmu: 1.0, bound: 1.0, nWorst: n}
	c.mmu(window, &acc)
	sort.Sort(sort.Reverse(acc.wHeap))
	return ([]UtilWindow)(acc.wHeap)
}

func (c *MMUCurve) mmu(window time.Duration, acc *accumulator) {
	if window <= 0 {
		acc.mmu = 0
		return
	}
	util := c.util
	if max := time.Duration(util[len(util)-1].Time - util[0].Time); window > max {
		window = max
	}

	bandU := bandUtilHeap(c.mkBandUtil(window))

	// Process bands from lowest utilization bound to highest.
	heap.Init(&bandU)

	// Refine each band into a precise window and MMU until
	// refining the next lowest band can no longer affect the MMU
	// or windows.
	for len(bandU) > 0 && bandU[0].utilBound < acc.bound {
		c.bandMMU(bandU[0].i, window, acc)
		heap.Pop(&bandU)
	}
}

func (c *MMUCurve) mkBandUtil(window time.Duration) []bandUtil {
	// For each band, compute the worst-possible total mutator
	// utilization for all windows that start in that band.

	// minBands is the minimum number of bands a window can span
	// and maxBands is the maximum number of bands a window can
	// span in any alignment.
	minBands := int((int64(window) + c.bandDur - 1) / c.bandDur)
	maxBands := int((int64(window) + 2*(c.bandDur-1)) / c.bandDur)
	if window > 1 && maxBands < 2 {
		panic("maxBands < 2")
	}
	tailDur := int64(window) % c.bandDur
	nUtil := len(c.bands) - maxBands + 1
	if nUtil < 0 {
		nUtil = 0
	}
	bandU := make([]bandUtil, nUtil)
	for i := range bandU {
		// To compute the worst-case MU, we assume the minimum
		// for any bands that are only partially overlapped by
		// some window and the mean for any bands that are
		// completely covered by all windows.
		var util totalUtil

		// Find the lowest and second lowest of the partial
		// bands.
		l := c.bands[i].minUtil
		r1 := c.bands[i+minBands-1].minUtil
		r2 := c.bands[i+maxBands-1].minUtil
		minBand := math.Min(l, math.Min(r1, r2))
		// Assume the worst window maximally overlaps the
		// worst minimum and then the rest overlaps the second
		// worst minimum.
		if minBands == 1 {
			util += totalUtilOf(minBand, int64(window))
		} else {
			util += totalUtilOf(minBand, c.bandDur)
			midBand := 0.0
			switch {
			case minBand == l:
				midBand = math.Min(r1, r2)
			case minBand == r1:
				midBand = math.Min(l, r2)
			case minBand == r2:
				midBand = math.Min(l, r1)
			}
			util += totalUtilOf(midBand, tailDur)
		}

		// Add the total mean MU of bands that are completely
		// overlapped by all windows.
		if minBands > 2 {
			util += c.bands[i+minBands-1].cumUtil - c.bands[i+1].cumUtil
		}

		bandU[i] = bandUtil{i, util.mean(window)}
	}

	return bandU
}

// bandMMU computes the precise minimum mutator utilization for
// windows with a left edge in band bandIdx.
func (c *MMUCurve) bandMMU(bandIdx int, window time.Duration, acc *accumulator) {
	util := c.util

	// We think of the mutator utilization over time as the
	// box-filtered utilization function, which we call the
	// "windowed mutator utilization function". The resulting
	// function is continuous and piecewise linear (unless
	// window==0, which we handle elsewhere), where the boundaries
	// between segments occur when either edge of the window
	// encounters a change in the instantaneous mutator
	// utilization function. Hence, the minimum of this function
	// will always occur when one of the edges of the window
	// aligns with a utilization change, so these are the only
	// points we need to consider.
	//
	// We compute the mutator utilization function incrementally
	// by tracking the integral from t=0 to the left edge of the
	// window and to the right edge of the window.
	left := c.bands[bandIdx].integrator
	right := left
	time, endTime := c.bandTime(bandIdx)
	if utilEnd := util[len(util)-1].Time - int64(window); utilEnd < endTime {
		endTime = utilEnd
	}
	for {
		// Advance edges to time and time+window.
		mu := (right.advance(time+int64(window)) - left.advance(time)).mean(window)
		if acc.addMU(time, mu, window) {
			break
		}

		// The maximum slope of the windowed mutator
		// utilization function is 1/window, so we can always
		// advance the time by at least (mu - mmu) * window
		// without dropping below mmu.
		minTime := time + int64((mu-acc.bound)*float64(window))

		// Advance the window to the next time where either
		// the left or right edge of the window encounters a
		// change in the utilization curve.
		if t1, t2 := left.next(time), right.next(time+int64(window))-int64(window); t1 < t2 {
			time = t1
		} else {
			time = t2
		}
		if time < minTime {
			time = minTime
		}
		if time >= endTime {
			break
		}
	}
}

// An integrator tracks a position in a utilization function and
// integrates it.
type integrator struct {
	u *MMUCurve
	// pos is the index in u.util of the current time's non-strict
	// predecessor.
	pos int
}

// advance returns the integral of the utilization function from 0 to
// time. advance must be called on monotonically increasing values of
// times.
func (in *integrator) advance(time int64) totalUtil {
	util, pos := in.u.util, in.pos
	// Advance pos until pos+1 is time's strict successor (making
	// pos time's non-strict predecessor).
	//
	// Very often, this will be nearby, so we optimize that case,
	// but it may be arbitrarily far away, so we handled that
	// efficiently, too.
	const maxSeq = 8
	if pos+maxSeq < len(util) && util[pos+maxSeq].Time > time {
		// Nearby. Use a linear scan.
		for pos+1 < len(util) && util[pos+1].Time <= time {
			pos++
		}
	} else {
		// Far. Binary search for time's strict successor.
		l, r := pos, len(util)
		for l < r {
			h := int(uint(l+r) >> 1)
			if util[h].Time <= time {
				l = h + 1
			} else {
				r = h
			}
		}
		pos = l - 1 // Non-strict predecessor.
	}
	in.pos = pos
	var partial totalUtil
	if time != util[pos].Time {
		partial = totalUtilOf(util[pos].Util, time-util[pos].Time)
	}
	return in.u.sums[pos] + partial
}

// next returns the smallest time t' > time of a change in the
// utilization function.
func (in *integrator) next(time int64) int64 {
	for _, u := range in.u.util[in.pos:] {
		if u.Time > time {
			return u.Time
		}
	}
	return 1<<63 - 1
}
