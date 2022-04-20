// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

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

	// UtilPerProc means each P should be given a separate
	// utilization function. Otherwise, there is a single function
	// and each P is given a fraction of the utilization.
	UtilPerProc
)

// MutatorUtilization returns a set of mutator utilization functions
// for the given trace. Each function will always end with 0
// utilization. The bounds of each function are implicit in the first
// and last event; outside of these bounds each function is undefined.
//
// If the UtilPerProc flag is not given, this always returns a single
// utilization function. Otherwise, it returns one function per P.
func MutatorUtilization(events []*Event, flags UtilFlags) [][]MutatorUtil {
	if len(events) == 0 {
		return nil
	}

	type perP struct {
		// gc > 0 indicates that GC is active on this P.
		gc int
		// series the logical series number for this P. This
		// is necessary because Ps may be removed and then
		// re-added, and then the new P needs a new series.
		series int
	}
	ps := []perP{}
	stw := 0

	out := [][]MutatorUtil{}
	assists := map[uint64]bool{}
	block := map[uint64]*Event{}
	bgMark := map[uint64]bool{}

	for _, ev := range events {
		switch ev.Type {
		case EvGomaxprocs:
			gomaxprocs := int(ev.Args[0])
			if len(ps) > gomaxprocs {
				if flags&UtilPerProc != 0 {
					// End each P's series.
					for _, p := range ps[gomaxprocs:] {
						out[p.series] = addUtil(out[p.series], MutatorUtil{ev.Ts, 0})
					}
				}
				ps = ps[:gomaxprocs]
			}
			for len(ps) < gomaxprocs {
				// Start new P's series.
				series := 0
				if flags&UtilPerProc != 0 || len(out) == 0 {
					series = len(out)
					out = append(out, []MutatorUtil{{ev.Ts, 1}})
				}
				ps = append(ps, perP{series: series})
			}
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
				ps[ev.P].gc++
				assists[ev.G] = true
			}
		case EvGCMarkAssistDone:
			if flags&UtilAssist != 0 {
				ps[ev.P].gc--
				delete(assists, ev.G)
			}
		case EvGCSweepStart:
			if flags&UtilSweep != 0 {
				ps[ev.P].gc++
			}
		case EvGCSweepDone:
			if flags&UtilSweep != 0 {
				ps[ev.P].gc--
			}
		case EvGoStartLabel:
			if flags&UtilBackground != 0 && strings.HasPrefix(ev.SArgs[0], "GC ") && ev.SArgs[0] != "GC (idle)" {
				// Background mark worker.
				//
				// If we're in per-proc mode, we don't
				// count dedicated workers because
				// they kick all of the goroutines off
				// that P, so don't directly
				// contribute to goroutine latency.
				if !(flags&UtilPerProc != 0 && ev.SArgs[0] == "GC (dedicated)") {
					bgMark[ev.G] = true
					ps[ev.P].gc++
				}
			}
			fallthrough
		case EvGoStart:
			if assists[ev.G] {
				// Unblocked during assist.
				ps[ev.P].gc++
			}
			block[ev.G] = ev.Link
		default:
			if ev != block[ev.G] {
				continue
			}

			if assists[ev.G] {
				// Blocked during assist.
				ps[ev.P].gc--
			}
			if bgMark[ev.G] {
				// Background mark worker done.
				ps[ev.P].gc--
				delete(bgMark, ev.G)
			}
			delete(block, ev.G)
		}

		if flags&UtilPerProc == 0 {
			// Compute the current average utilization.
			if len(ps) == 0 {
				continue
			}
			gcPs := 0
			if stw > 0 {
				gcPs = len(ps)
			} else {
				for i := range ps {
					if ps[i].gc > 0 {
						gcPs++
					}
				}
			}
			mu := MutatorUtil{ev.Ts, 1 - float64(gcPs)/float64(len(ps))}

			// Record the utilization change. (Since
			// len(ps) == len(out), we know len(out) > 0.)
			out[0] = addUtil(out[0], mu)
		} else {
			// Check for per-P utilization changes.
			for i := range ps {
				p := &ps[i]
				util := 1.0
				if stw > 0 || p.gc > 0 {
					util = 0.0
				}
				out[p.series] = addUtil(out[p.series], MutatorUtil{ev.Ts, util})
			}
		}
	}

	// Add final 0 utilization event to any remaining series. This
	// is important to mark the end of the trace. The exact value
	// shouldn't matter since no window should extend beyond this,
	// but using 0 is symmetric with the start of the trace.
	mu := MutatorUtil{events[len(events)-1].Ts, 0}
	for i := range ps {
		out[ps[i].series] = addUtil(out[ps[i].series], mu)
	}
	return out
}

func addUtil(util []MutatorUtil, mu MutatorUtil) []MutatorUtil {
	if len(util) > 0 {
		if mu.Util == util[len(util)-1].Util {
			// No change.
			return util
		}
		if mu.Time == util[len(util)-1].Time {
			// Take the lowest utilization at a time stamp.
			if mu.Util < util[len(util)-1].Util {
				util[len(util)-1] = mu
			}
			return util
		}
	}
	return append(util, mu)
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
	series []mmuSeries
}

type mmuSeries struct {
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
func NewMMUCurve(utils [][]MutatorUtil) *MMUCurve {
	series := make([]mmuSeries, len(utils))
	for i, util := range utils {
		series[i] = newMMUSeries(util)
	}
	return &MMUCurve{series}
}

// bandsPerSeries is the number of bands to divide each series into.
// This is only changed by tests.
var bandsPerSeries = 1000

func newMMUSeries(util []MutatorUtil) mmuSeries {
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
	numBands := bandsPerSeries
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
	s := mmuSeries{util, sums, bands, bandDur}
	leftSum := integrator{&s, 0}
	for i := range bands {
		startTime, endTime := s.bandTime(i)
		cumUtil := leftSum.advance(startTime)
		predIdx := leftSum.pos
		minUtil := 1.0
		for i := predIdx; i < len(util) && util[i].Time < endTime; i++ {
			minUtil = math.Min(minUtil, util[i].Util)
		}
		bands[i] = mmuBand{minUtil, cumUtil, leftSum}
	}

	return s
}

func (s *mmuSeries) bandTime(i int) (start, end int64) {
	start = int64(i)*s.bandDur + s.util[0].Time
	end = start + s.bandDur
	return
}

type bandUtil struct {
	// Utilization series index
	series int
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

func (h *bandUtilHeap) Push(x any) {
	*h = append(*h, x.(bandUtil))
}

func (h *bandUtilHeap) Pop() any {
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

func (h *utilHeap) Push(x any) {
	*h = append(*h, x.(UtilWindow))
}

func (h *utilHeap) Pop() any {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}

// An accumulator takes a windowed mutator utilization function and
// tracks various statistics for that function.
type accumulator struct {
	mmu float64

	// bound is the mutator utilization bound where adding any
	// mutator utilization above this bound cannot affect the
	// accumulated statistics.
	bound float64

	// Worst N window tracking
	nWorst int
	wHeap  utilHeap

	// Mutator utilization distribution tracking
	mud *mud
	// preciseMass is the distribution mass that must be precise
	// before accumulation is stopped.
	preciseMass float64
	// lastTime and lastMU are the previous point added to the
	// windowed mutator utilization function.
	lastTime int64
	lastMU   float64
}

// resetTime declares a discontinuity in the windowed mutator
// utilization function by resetting the current time.
func (acc *accumulator) resetTime() {
	// This only matters for distribution collection, since that's
	// the only thing that depends on the progression of the
	// windowed mutator utilization function.
	acc.lastTime = math.MaxInt64
}

// addMU adds a point to the windowed mutator utilization function at
// (time, mu). This must be called for monotonically increasing values
// of time.
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

	if acc.mud != nil {
		if acc.lastTime != math.MaxInt64 {
			// Update distribution.
			acc.mud.add(acc.lastMU, mu, float64(time-acc.lastTime))
		}
		acc.lastTime, acc.lastMU = time, mu
		if _, mudBound, ok := acc.mud.approxInvCumulativeSum(); ok {
			acc.bound = math.Max(acc.bound, mudBound)
		} else {
			// We haven't accumulated enough total precise
			// mass yet to even reach our goal, so keep
			// accumulating.
			acc.bound = 1
		}
		// It's not worth checking percentiles every time, so
		// just keep accumulating this band.
		return false
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

// MUD returns mutator utilization distribution quantiles for the
// given window size.
//
// The mutator utilization distribution is the distribution of mean
// mutator utilization across all windows of the given window size in
// the trace.
//
// The minimum mutator utilization is the minimum (0th percentile) of
// this distribution. (However, if only the minimum is desired, it's
// more efficient to use the MMU method.)
func (c *MMUCurve) MUD(window time.Duration, quantiles []float64) []float64 {
	if len(quantiles) == 0 {
		return []float64{}
	}

	// Each unrefined band contributes a known total mass to the
	// distribution (bandDur except at the end), but in an unknown
	// way. However, we know that all the mass it contributes must
	// be at or above its worst-case mean mutator utilization.
	//
	// Hence, we refine bands until the highest desired
	// distribution quantile is less than the next worst-case mean
	// mutator utilization. At this point, all further
	// contributions to the distribution must be beyond the
	// desired quantile and hence cannot affect it.
	//
	// First, find the highest desired distribution quantile.
	maxQ := quantiles[0]
	for _, q := range quantiles {
		if q > maxQ {
			maxQ = q
		}
	}
	// The distribution's mass is in units of time (it's not
	// normalized because this would make it more annoying to
	// account for future contributions of unrefined bands). The
	// total final mass will be the duration of the trace itself
	// minus the window size. Using this, we can compute the mass
	// corresponding to quantile maxQ.
	var duration int64
	for _, s := range c.series {
		duration1 := s.util[len(s.util)-1].Time - s.util[0].Time
		if duration1 >= int64(window) {
			duration += duration1 - int64(window)
		}
	}
	qMass := float64(duration) * maxQ

	// Accumulate the MUD until we have precise information for
	// everything to the left of qMass.
	acc := accumulator{mmu: 1.0, bound: 1.0, preciseMass: qMass, mud: new(mud)}
	acc.mud.setTrackMass(qMass)
	c.mmu(window, &acc)

	// Evaluate the quantiles on the accumulated MUD.
	out := make([]float64, len(quantiles))
	for i := range out {
		mu, _ := acc.mud.invCumulativeSum(float64(duration) * quantiles[i])
		if math.IsNaN(mu) {
			// There are a few legitimate ways this can
			// happen:
			//
			// 1. If the window is the full trace
			// duration, then the windowed MU function is
			// only defined at a single point, so the MU
			// distribution is not well-defined.
			//
			// 2. If there are no events, then the MU
			// distribution has no mass.
			//
			// Either way, all of the quantiles will have
			// converged toward the MMU at this point.
			mu = acc.mmu
		}
		out[i] = mu
	}
	return out
}

func (c *MMUCurve) mmu(window time.Duration, acc *accumulator) {
	if window <= 0 {
		acc.mmu = 0
		return
	}

	var bandU bandUtilHeap
	windows := make([]time.Duration, len(c.series))
	for i, s := range c.series {
		windows[i] = window
		if max := time.Duration(s.util[len(s.util)-1].Time - s.util[0].Time); window > max {
			windows[i] = max
		}

		bandU1 := bandUtilHeap(s.mkBandUtil(i, windows[i]))
		if bandU == nil {
			bandU = bandU1
		} else {
			bandU = append(bandU, bandU1...)
		}
	}

	// Process bands from lowest utilization bound to highest.
	heap.Init(&bandU)

	// Refine each band into a precise window and MMU until
	// refining the next lowest band can no longer affect the MMU
	// or windows.
	for len(bandU) > 0 && bandU[0].utilBound < acc.bound {
		i := bandU[0].series
		c.series[i].bandMMU(bandU[0].i, windows[i], acc)
		heap.Pop(&bandU)
	}
}

func (c *mmuSeries) mkBandUtil(series int, window time.Duration) []bandUtil {
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

		bandU[i] = bandUtil{series, i, util.mean(window)}
	}

	return bandU
}

// bandMMU computes the precise minimum mutator utilization for
// windows with a left edge in band bandIdx.
func (c *mmuSeries) bandMMU(bandIdx int, window time.Duration, acc *accumulator) {
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
	acc.resetTime()
	for {
		// Advance edges to time and time+window.
		mu := (right.advance(time+int64(window)) - left.advance(time)).mean(window)
		if acc.addMU(time, mu, window) {
			break
		}
		if time == endTime {
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
			// For MMUs we could stop here, but for MUDs
			// it's important that we span the entire
			// band.
			time = endTime
		}
	}
}

// An integrator tracks a position in a utilization function and
// integrates it.
type integrator struct {
	u *mmuSeries
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
