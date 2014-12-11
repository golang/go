// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"appengine"
	"appengine/datastore"
)

var knownTags = map[string]string{
	"go1":   "0051c7442fed9c888de6617fa9239a913904d96e",
	"go1.1": "d29da2ced72ba2cf48ed6a8f1ec4abc01e4c5bf1",
	"go1.2": "b1edf8faa5d6cbc50c6515785df9df9c19296564",
	"go1.3": "f153208c0a0e306bfca14f71ef11f09859ccabc8",
	"go1.4": "faa3ed1dc30e42771a68b6337dcf8be9518d5c07",
}

var lastRelease = "go1.4"

func splitBench(benchProcs string) (string, int) {
	ss := strings.Split(benchProcs, "-")
	procs, _ := strconv.Atoi(ss[1])
	return ss[0], procs
}

func dashPerfCommits(c appengine.Context, page int) ([]*Commit, error) {
	q := datastore.NewQuery("Commit").
		Ancestor((&Package{}).Key(c)).
		Order("-Num").
		Filter("NeedsBenchmarking =", true).
		Limit(commitsPerPage).
		Offset(page * commitsPerPage)
	var commits []*Commit
	_, err := q.GetAll(c, &commits)
	if err == nil && len(commits) == 0 {
		err = fmt.Errorf("no commits")
	}
	return commits, err
}

func perfChangeStyle(pc *PerfConfig, v float64, builder, benchmark, metric string) string {
	noise := pc.NoiseLevel(builder, benchmark, metric)
	if isNoise(v, noise) {
		return "noise"
	}
	if v > 0 {
		return "bad"
	}
	return "good"
}

func isNoise(diff, noise float64) bool {
	rnoise := -100 * noise / (noise + 100)
	return diff < noise && diff > rnoise
}

func perfDiff(old, new uint64) float64 {
	return 100*float64(new)/float64(old) - 100
}

func isPerfFailed(res *PerfResult, builder string) bool {
	data := res.ParseData()[builder]
	return data != nil && data["meta-done"] != nil && !data["meta-done"].OK
}

// PerfResultCache caches a set of PerfResults so that it's easy to access them
// without lots of duplicate accesses to datastore.
// It allows to iterate over newer or older results for some base commit.
type PerfResultCache struct {
	c       appengine.Context
	newer   bool
	iter    *datastore.Iterator
	results map[int]*PerfResult
}

func MakePerfResultCache(c appengine.Context, com *Commit, newer bool) *PerfResultCache {
	p := &Package{}
	q := datastore.NewQuery("PerfResult").Ancestor(p.Key(c)).Limit(100)
	if newer {
		q = q.Filter("CommitNum >=", com.Num).Order("CommitNum")
	} else {
		q = q.Filter("CommitNum <=", com.Num).Order("-CommitNum")
	}
	rc := &PerfResultCache{c: c, newer: newer, iter: q.Run(c), results: make(map[int]*PerfResult)}
	return rc
}

func (rc *PerfResultCache) Get(commitNum int) *PerfResult {
	rc.Next(commitNum) // fetch the commit, if necessary
	return rc.results[commitNum]
}

// Next returns the next PerfResult for the commit commitNum.
// It does not care whether the result has any data, failed or whatever.
func (rc *PerfResultCache) Next(commitNum int) (*PerfResult, error) {
	// See if we have next result in the cache.
	next := -1
	for ci := range rc.results {
		if rc.newer {
			if ci > commitNum && (next == -1 || ci < next) {
				next = ci
			}
		} else {
			if ci < commitNum && (next == -1 || ci > next) {
				next = ci
			}
		}
	}
	if next != -1 {
		return rc.results[next], nil
	}
	// Fetch next result from datastore.
	res := new(PerfResult)
	_, err := rc.iter.Next(res)
	if err == datastore.Done {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("fetching perf results: %v", err)
	}
	if (rc.newer && res.CommitNum < commitNum) || (!rc.newer && res.CommitNum > commitNum) {
		rc.c.Errorf("PerfResultCache.Next: bad commit num")
	}
	rc.results[res.CommitNum] = res
	return res, nil
}

// NextForComparison returns PerfResult which we need to use for performance comprison.
// It skips failed results, but does not skip results with no data.
func (rc *PerfResultCache) NextForComparison(commitNum int, builder string) (*PerfResult, error) {
	for {
		res, err := rc.Next(commitNum)
		if err != nil {
			return nil, err
		}
		if res == nil {
			return nil, nil
		}
		if res.CommitNum == commitNum {
			continue
		}
		parsed := res.ParseData()
		if builder != "" {
			// Comparing for a particular builder.
			// This is used in perf_changes and in email notifications.
			b := parsed[builder]
			if b == nil || b["meta-done"] == nil {
				// No results yet, must not do the comparison.
				return nil, nil
			}
			if b["meta-done"].OK {
				// Have complete results, compare.
				return res, nil
			}
		} else {
			// Comparing for all builders, find a result with at least
			// one successful meta-done.
			// This is used in perf_detail.
			for _, benchs := range parsed {
				if data := benchs["meta-done"]; data != nil && data.OK {
					return res, nil
				}
			}
		}
		// Failed, try next result.
		commitNum = res.CommitNum
	}
}

type PerfChange struct {
	Builder string
	Bench   string
	Metric  string
	Old     uint64
	New     uint64
	Diff    float64
}

func significantPerfChanges(pc *PerfConfig, builder string, prevRes, res *PerfResult) (changes []*PerfChange) {
	// First, collect all significant changes.
	for builder1, benchmarks1 := range res.ParseData() {
		if builder != "" && builder != builder1 {
			// This is not the builder you're looking for, Luke.
			continue
		}
		benchmarks0 := prevRes.ParseData()[builder1]
		if benchmarks0 == nil {
			continue
		}
		for benchmark, data1 := range benchmarks1 {
			data0 := benchmarks0[benchmark]
			if data0 == nil {
				continue
			}
			for metric, val := range data1.Metrics {
				val0 := data0.Metrics[metric]
				if val0 == 0 {
					continue
				}
				diff := perfDiff(val0, val)
				noise := pc.NoiseLevel(builder, benchmark, metric)
				if isNoise(diff, noise) {
					continue
				}
				ch := &PerfChange{Builder: builder, Bench: benchmark, Metric: metric, Old: val0, New: val, Diff: diff}
				changes = append(changes, ch)
			}
		}
	}
	// Then, strip non-repeatable changes (flakes).
	// The hypothesis is that a real change must show up with the majority of GOMAXPROCS values.
	majority := len(pc.ProcList(builder))/2 + 1
	cnt := make(map[string]int)
	for _, ch := range changes {
		b, _ := splitBench(ch.Bench)
		name := b + "|" + ch.Metric
		if ch.Diff < 0 {
			name += "--"
		}
		cnt[name] = cnt[name] + 1
	}
	for i := 0; i < len(changes); i++ {
		ch := changes[i]
		b, _ := splitBench(ch.Bench)
		name := b + "|" + ch.Metric
		if cnt[name] >= majority {
			continue
		}
		if cnt[name+"--"] >= majority {
			continue
		}
		// Remove flake.
		last := len(changes) - 1
		changes[i] = changes[last]
		changes = changes[:last]
		i--
	}
	return changes
}

// orderPerfTodo reorders commit nums for benchmarking todo.
// The resulting order is somewhat tricky. We want 2 things:
// 1. benchmark sequentially backwards (this provides information about most
// recent changes, and allows to estimate noise levels)
// 2. benchmark old commits in "scatter" order (this allows to quickly gather
// brief information about thousands of old commits)
// So this function interleaves the two orders.
func orderPerfTodo(nums []int) []int {
	sort.Ints(nums)
	n := len(nums)
	pow2 := uint32(0) // next power-of-two that is >= n
	npow2 := 0
	for npow2 <= n {
		pow2++
		npow2 = 1 << pow2
	}
	res := make([]int, n)
	resPos := n - 1            // result array is filled backwards
	present := make([]bool, n) // denotes values that already present in result array
	for i0, i1 := n-1, 0; i0 >= 0 || i1 < npow2; {
		// i0 represents "benchmark sequentially backwards" sequence
		// find the next commit that is not yet present and add it
		for cnt := 0; cnt < 2; cnt++ {
			for ; i0 >= 0; i0-- {
				if !present[i0] {
					present[i0] = true
					res[resPos] = nums[i0]
					resPos--
					i0--
					break
				}
			}
		}
		// i1 represents "scatter order" sequence
		// find the next commit that is not yet present and add it
		for ; i1 < npow2; i1++ {
			// do the "recursive split-ordering" trick
			idx := 0 // bitwise reverse of i1
			for j := uint32(0); j <= pow2; j++ {
				if (i1 & (1 << j)) != 0 {
					idx = idx | (1 << (pow2 - j - 1))
				}
			}
			if idx < n && !present[idx] {
				present[idx] = true
				res[resPos] = nums[idx]
				resPos--
				i1++
				break
			}
		}
	}
	// The above can't possibly be correct. Do dump check.
	res2 := make([]int, n)
	copy(res2, res)
	sort.Ints(res2)
	for i := range res2 {
		if res2[i] != nums[i] {
			panic(fmt.Sprintf("diff at %v: expect %v, want %v\nwas: %v\n become: %v",
				i, nums[i], res2[i], nums, res2))
		}
	}
	return res
}
