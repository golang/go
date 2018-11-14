// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"math"
	"sort"
)

// mud is an updatable mutator utilization distribution.
//
// This is a continuous distribution of duration over mutator
// utilization. For example, the integral from mutator utilization a
// to b is the total duration during which the mutator utilization was
// in the range [a, b].
//
// This distribution is *not* normalized (it is not a probability
// distribution). This makes it easier to work with as it's being
// updated.
//
// It is represented as the sum of scaled uniform distribution
// functions and Dirac delta functions (which are treated as
// degenerate uniform distributions).
type mud struct {
	sorted, unsorted []edge

	// trackMass is the inverse cumulative sum to track as the
	// distribution is updated.
	trackMass float64
	// trackBucket is the bucket in which trackMass falls. If the
	// total mass of the distribution is < trackMass, this is
	// len(hist).
	trackBucket int
	// trackSum is the cumulative sum of hist[:trackBucket]. Once
	// trackSum >= trackMass, trackBucket must be recomputed.
	trackSum float64

	// hist is a hierarchical histogram of distribution mass.
	hist [mudDegree]float64
}

const (
	// mudDegree is the number of buckets in the MUD summary
	// histogram.
	mudDegree = 1024
)

type edge struct {
	// At x, the function increases by y.
	x, delta float64
	// Additionally at x is a Dirac delta function with area dirac.
	dirac float64
}

// add adds a uniform function over [l, r] scaled so the total weight
// of the uniform is area. If l==r, this adds a Dirac delta function.
func (d *mud) add(l, r, area float64) {
	if area == 0 {
		return
	}

	if r < l {
		l, r = r, l
	}

	// Add the edges.
	if l == r {
		d.unsorted = append(d.unsorted, edge{l, 0, area})
	} else {
		delta := area / (r - l)
		d.unsorted = append(d.unsorted, edge{l, delta, 0}, edge{r, -delta, 0})
	}

	// Update the histogram.
	h := &d.hist
	lbFloat, lf := math.Modf(l * mudDegree)
	lb := int(lbFloat)
	if lb >= mudDegree {
		lb, lf = mudDegree-1, 1
	}
	if l == r {
		h[lb] += area
	} else {
		rbFloat, rf := math.Modf(r * mudDegree)
		rb := int(rbFloat)
		if rb >= mudDegree {
			rb, rf = mudDegree-1, 1
		}
		if lb == rb {
			h[lb] += area
		} else {
			perBucket := area / (r - l) / mudDegree
			h[lb] += perBucket * (1 - lf)
			h[rb] += perBucket * rf
			for i := lb + 1; i < rb; i++ {
				h[i] += perBucket
			}
		}
	}

	// Update mass tracking.
	if thresh := float64(d.trackBucket) / mudDegree; l < thresh {
		if r < thresh {
			d.trackSum += area
		} else {
			d.trackSum += area * (thresh - l) / (r - l)
		}
		if d.trackSum >= d.trackMass {
			// The tracked mass now falls in a different
			// bucket. Recompute the inverse cumulative sum.
			d.setTrackMass(d.trackMass)
		}
	}
}

// setTrackMass sets the mass to track the inverse cumulative sum for.
//
// Specifically, mass is a cumulative duration, and the mutator
// utilization bounds for this duration can be queried using
// approxInvCumulativeSum.
func (d *mud) setTrackMass(mass float64) {
	d.trackMass = mass

	// Find the bucket currently containing trackMass by computing
	// the cumulative sum.
	sum := 0.0
	for i, val := range d.hist[:] {
		newSum := sum + val
		if newSum > mass {
			// mass falls in bucket i.
			d.trackBucket = i
			d.trackSum = sum
			return
		}
		sum = newSum
	}
	d.trackBucket = len(d.hist)
	d.trackSum = sum
}

// approxInvCumulativeSum is like invCumulativeSum, but specifically
// operates on the tracked mass and returns an upper and lower bound
// approximation of the inverse cumulative sum.
//
// The true inverse cumulative sum will be in the range [lower, upper).
func (d *mud) approxInvCumulativeSum() (float64, float64, bool) {
	if d.trackBucket == len(d.hist) {
		return math.NaN(), math.NaN(), false
	}
	return float64(d.trackBucket) / mudDegree, float64(d.trackBucket+1) / mudDegree, true
}

// invCumulativeSum returns x such that the integral of d from -∞ to x
// is y. If the total weight of d is less than y, it returns the
// maximum of the distribution and false.
//
// Specifically, y is a cumulative duration, and invCumulativeSum
// returns the mutator utilization x such that at least y time has
// been spent with mutator utilization <= x.
func (d *mud) invCumulativeSum(y float64) (float64, bool) {
	if len(d.sorted) == 0 && len(d.unsorted) == 0 {
		return math.NaN(), false
	}

	// Sort edges.
	edges := d.unsorted
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].x < edges[j].x
	})
	// Merge with sorted edges.
	d.unsorted = nil
	if d.sorted == nil {
		d.sorted = edges
	} else {
		oldSorted := d.sorted
		newSorted := make([]edge, len(oldSorted)+len(edges))
		i, j := 0, 0
		for o := range newSorted {
			if i >= len(oldSorted) {
				copy(newSorted[o:], edges[j:])
				break
			} else if j >= len(edges) {
				copy(newSorted[o:], oldSorted[i:])
				break
			} else if oldSorted[i].x < edges[j].x {
				newSorted[o] = oldSorted[i]
				i++
			} else {
				newSorted[o] = edges[j]
				j++
			}
		}
		d.sorted = newSorted
	}

	// Traverse edges in order computing a cumulative sum.
	csum, rate, prevX := 0.0, 0.0, 0.0
	for _, e := range d.sorted {
		newCsum := csum + (e.x-prevX)*rate
		if newCsum >= y {
			// y was exceeded between the previous edge
			// and this one.
			if rate == 0 {
				// Anywhere between prevX and
				// e.x will do. We return e.x
				// because that takes care of
				// the y==0 case naturally.
				return e.x, true
			}
			return (y-csum)/rate + prevX, true
		}
		newCsum += e.dirac
		if newCsum >= y {
			// y was exceeded by the Dirac delta at e.x.
			return e.x, true
		}
		csum, prevX = newCsum, e.x
		rate += e.delta
	}
	return prevX, false
}
