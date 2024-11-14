// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package liveness

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"testing"
)

func TestMain(m *testing.M) {
	flag.Parse()
	os.Exit(m.Run())
}

func TestMakeAndPrint(t *testing.T) {
	testcases := []struct {
		inp []int
		exp string
		err bool
	}{
		{
			inp: []int{0, 1, 2, 3},
			exp: "[0,1) [2,3)",
		},
		{ // degenerate but legal
			inp: []int{0, 1, 1, 2},
			exp: "[0,1) [1,2)",
		},
		{ // odd number of elems
			inp: []int{0},
			err: true,
			exp: "odd number of elems 1",
		},
		{
			// bad range element
			inp: []int{0, 0},
			err: true,
			exp: "bad range elem 0:0, en<=st",
		},
		{
			// overlap w/ previous
			inp: []int{0, 9, 3, 12},
			err: true,
			exp: "bad range elem 3:12 overlaps prev 0:9",
		},
		{
			// range starts not ordered
			inp: []int{10, 11, 3, 4},
			err: true,
			exp: "range start not ordered 3:4 less than prev 10:11",
		},
	}

	for k, tc := range testcases {
		is, err := makeIntervals(tc.inp...)
		want := tc.exp
		if err != nil {
			if !tc.err {
				t.Fatalf("unexpected error on tc:%d %+v -> %v", k, tc.inp, err)
			} else {
				got := fmt.Sprintf("%v", err)
				if got != want {
					t.Fatalf("bad error on tc:%d %+v got %q want %q", k, tc.inp, got, want)
				}
			}
			continue
		} else if tc.err {
			t.Fatalf("missing error on tc:%d %+v return was %q", k, tc.inp, is.String())
		}
		got := is.String()
		if got != want {
			t.Fatalf("exp mismatch on tc:%d %+v got %q want %q", k, tc.inp, got, want)
		}
	}
}

func TestIntervalOverlap(t *testing.T) {
	testcases := []struct {
		i1, i2 Interval
		exp    bool
	}{
		{
			i1:  Interval{st: 0, en: 1},
			i2:  Interval{st: 0, en: 1},
			exp: true,
		},
		{
			i1:  Interval{st: 0, en: 1},
			i2:  Interval{st: 1, en: 2},
			exp: false,
		},
		{
			i1:  Interval{st: 9, en: 10},
			i2:  Interval{st: 1, en: 2},
			exp: false,
		},
		{
			i1:  Interval{st: 0, en: 10},
			i2:  Interval{st: 5, en: 6},
			exp: true,
		},
	}

	for _, tc := range testcases {
		want := tc.exp
		got := tc.i1.Overlaps(tc.i2)
		if want != got {
			t.Fatalf("Overlaps([%d,%d), [%d,%d)): got %v want %v",
				tc.i1.st, tc.i1.en, tc.i2.st, tc.i2.en, got, want)
		}
	}
}

func TestIntervalAdjacent(t *testing.T) {
	testcases := []struct {
		i1, i2 Interval
		exp    bool
	}{
		{
			i1:  Interval{st: 0, en: 1},
			i2:  Interval{st: 0, en: 1},
			exp: false,
		},
		{
			i1:  Interval{st: 0, en: 1},
			i2:  Interval{st: 1, en: 2},
			exp: true,
		},
		{
			i1:  Interval{st: 1, en: 2},
			i2:  Interval{st: 0, en: 1},
			exp: true,
		},
		{
			i1:  Interval{st: 0, en: 10},
			i2:  Interval{st: 0, en: 3},
			exp: false,
		},
	}

	for k, tc := range testcases {
		want := tc.exp
		got := tc.i1.adjacent(tc.i2)
		if want != got {
			t.Fatalf("tc=%d adjacent([%d,%d), [%d,%d)): got %v want %v",
				k, tc.i1.st, tc.i1.en, tc.i2.st, tc.i2.en, got, want)
		}
	}
}

func TestIntervalMerge(t *testing.T) {
	testcases := []struct {
		i1, i2 Interval
		exp    Interval
		err    bool
	}{
		{
			// error case
			i1:  Interval{st: 0, en: 1},
			i2:  Interval{st: 2, en: 3},
			err: true,
		},
		{
			// same
			i1:  Interval{st: 0, en: 1},
			i2:  Interval{st: 0, en: 1},
			exp: Interval{st: 0, en: 1},
			err: false,
		},
		{
			// adjacent
			i1:  Interval{st: 0, en: 1},
			i2:  Interval{st: 1, en: 2},
			exp: Interval{st: 0, en: 2},
			err: false,
		},
		{
			// overlapping 1
			i1:  Interval{st: 0, en: 5},
			i2:  Interval{st: 3, en: 10},
			exp: Interval{st: 0, en: 10},
			err: false,
		},
		{
			// overlapping 2
			i1:  Interval{st: 9, en: 15},
			i2:  Interval{st: 3, en: 11},
			exp: Interval{st: 3, en: 15},
			err: false,
		},
	}

	for k, tc := range testcases {
		var dst Interval
		dstp := &dst
		dst = tc.i1
		err := dstp.MergeInto(tc.i2)
		if (err != nil) != tc.err {
			t.Fatalf("tc=%d MergeInto([%d,%d) <= [%d,%d)): got err=%v want err=%v", k, tc.i1.st, tc.i1.en, tc.i2.st, tc.i2.en, err, tc.err)
		}
		if err != nil {
			continue
		}
		want := tc.exp.String()
		got := dst.String()
		if want != got {
			t.Fatalf("tc=%d MergeInto([%d,%d) <= [%d,%d)): got %v want %v",
				k, tc.i1.st, tc.i1.en, tc.i2.st, tc.i2.en, got, want)
		}
	}
}

func TestIntervalsOverlap(t *testing.T) {
	testcases := []struct {
		inp1, inp2 []int
		exp        bool
	}{
		{
			// first empty
			inp1: []int{},
			inp2: []int{1, 2},
			exp:  false,
		},
		{
			// second empty
			inp1: []int{9, 10},
			inp2: []int{},
			exp:  false,
		},
		{
			// disjoint 1
			inp1: []int{1, 2},
			inp2: []int{2, 3},
			exp:  false,
		},
		{
			// disjoint 2
			inp1: []int{2, 3},
			inp2: []int{1, 2},
			exp:  false,
		},
		{
			// interleaved 1
			inp1: []int{1, 2, 3, 4},
			inp2: []int{2, 3, 5, 6},
			exp:  false,
		},
		{
			// interleaved 2
			inp1: []int{2, 3, 5, 6},
			inp2: []int{1, 2, 3, 4},
			exp:  false,
		},
		{
			// overlap 1
			inp1: []int{1, 3},
			inp2: []int{2, 9, 10, 11},
			exp:  true,
		},
		{
			// overlap 2
			inp1: []int{18, 29},
			inp2: []int{2, 9, 10, 19},
			exp:  true,
		},
	}

	for k, tc := range testcases {
		is1, err1 := makeIntervals(tc.inp1...)
		if err1 != nil {
			t.Fatalf("unexpected error on tc:%d %+v: %v", k, tc.inp1, err1)
		}
		is2, err2 := makeIntervals(tc.inp2...)
		if err2 != nil {
			t.Fatalf("unexpected error on tc:%d %+v: %v", k, tc.inp2, err2)
		}
		got := is1.Overlaps(is2)
		want := tc.exp
		if got != want {
			t.Fatalf("overlaps mismatch on tc:%d %+v %+v got %v want %v", k, tc.inp1, tc.inp2, got, want)
		}
	}
}

var seedflag = flag.Int64("seed", 101, "Random seed")
var trialsflag = flag.Int64("trials", 10000, "Number of trials")
var segsflag = flag.Int64("segs", 4, "Max segments within interval")
var limitflag = flag.Int64("limit", 20, "Limit of interval max end")

// NB: consider turning this into a fuzz test if the interval data
// structures or code get any more complicated.

func TestRandomIntervalsOverlap(t *testing.T) {
	rand.Seed(*seedflag)

	// Return a pseudo-random intervals object with 0-3 segments within
	// the range of 0 to limit
	mk := func {
		vals := rand.Perm(int(*limitflag))
		// decide how many segments
		segs := rand.Intn(int(*segsflag))
		picked := vals[:(segs * 2)]
		sort.Ints(picked)
		ii, err := makeIntervals(picked...)
		if err != nil {
			t.Fatalf("makeIntervals(%+v) returns err %v", picked, err)
		}
		return ii
	}

	brute := func(i1, i2 Intervals) bool {
		for i := range i1 {
			for j := range i2 {
				if i1[i].Overlaps(i2[j]) {
					return true
				}
			}
		}
		return false
	}

	for k := range *trialsflag {
		// Create two interval ranges and test if they overlap. Then
		// compare the overlap with a brute-force overlap calculation.
		i1, i2 := mk(), mk()
		got := i1.Overlaps(i2)
		want := brute(i1, i2)
		if got != want {
			t.Fatalf("overlap mismatch on t:%d %v %v got %v want %v",
				k, i1, i2, got, want)
		}
	}
}

func TestIntervalsMerge(t *testing.T) {
	testcases := []struct {
		inp1, inp2 []int
		exp        []int
	}{
		{
			// first empty
			inp1: []int{},
			inp2: []int{1, 2},
			exp:  []int{1, 2},
		},
		{
			// second empty
			inp1: []int{1, 2},
			inp2: []int{},
			exp:  []int{1, 2},
		},
		{
			// overlap 1
			inp1: []int{1, 2},
			inp2: []int{2, 3},
			exp:  []int{1, 3},
		},
		{
			// overlap 2
			inp1: []int{1, 5},
			inp2: []int{2, 10},
			exp:  []int{1, 10},
		},
		{
			// non-overlap 1
			inp1: []int{1, 2},
			inp2: []int{11, 12},
			exp:  []int{1, 2, 11, 12},
		},
		{
			// non-overlap 2
			inp1: []int{1, 2, 3, 4, 5, 6},
			inp2: []int{2, 3, 4, 5, 6, 7},
			exp:  []int{1, 7},
		},
	}

	for k, tc := range testcases {
		is1, err1 := makeIntervals(tc.inp1...)
		if err1 != nil {
			t.Fatalf("unexpected error on tc:%d %+v: %v", k, tc.inp1, err1)
		}
		is2, err2 := makeIntervals(tc.inp2...)
		if err2 != nil {
			t.Fatalf("unexpected error on tc:%d %+v: %v", k, tc.inp2, err2)
		}
		m := is1.Merge(is2)
		wis, werr := makeIntervals(tc.exp...)
		if werr != nil {
			t.Fatalf("unexpected error on tc:%d %+v: %v", k, tc.exp, werr)
		}
		want := wis.String()
		got := m.String()
		if want != got {
			t.Fatalf("k=%d Merge(%s, %s): got %v want %v",
				k, is1, is2, m, want)
		}
	}
}

func TestBuilder(t *testing.T) {
	type posLiveKill struct {
		pos                 int
		becomesLive, isKill bool // what to pass to IntervalsBuilder
	}
	testcases := []struct {
		inp        []posLiveKill
		exp        []int
		aerr, ferr bool
	}{
		// error case, position non-decreasing
		{
			inp: []posLiveKill{
				{pos: 10, becomesLive: true},
				{pos: 18, isKill: true},
			},
			aerr: true,
		},
		// error case, position negative
		{
			inp: []posLiveKill{
				{pos: -1, becomesLive: true},
			},
			aerr: true,
		},
		// empty
		{
			exp: nil,
		},
		// single BB
		{
			inp: []posLiveKill{
				{pos: 10, becomesLive: true},
				{pos: 9, isKill: true},
			},
			exp: []int{10, 11},
		},
		// couple of BBs
		{
			inp: []posLiveKill{
				{pos: 11, becomesLive: true},
				{pos: 10, becomesLive: true},
				{pos: 9, isKill: true},
				{pos: 4, becomesLive: true},
				{pos: 1, isKill: true},
			},
			exp: []int{2, 5, 10, 12},
		},
		// couple of BBs
		{
			inp: []posLiveKill{
				{pos: 20, isKill: true},
				{pos: 19, isKill: true},
				{pos: 17, becomesLive: true},
				{pos: 14, becomesLive: true},
				{pos: 10, isKill: true},
				{pos: 4, becomesLive: true},
				{pos: 0, isKill: true},
			},
			exp: []int{1, 5, 11, 18},
		},
	}

	for k, tc := range testcases {
		var c IntervalsBuilder
		var aerr error
		for _, event := range tc.inp {
			if event.becomesLive {
				if err := c.Live(event.pos); err != nil {
					aerr = err
					break
				}
			}
			if event.isKill {
				if err := c.Kill(event.pos); err != nil {
					aerr = err
					break
				}
			}
		}
		if (aerr != nil) != tc.aerr {
			t.Fatalf("k=%d add err mismatch: tc.aerr:%v aerr!=nil:%v",
				k, tc.aerr, (aerr != nil))
		}
		if tc.aerr {
			continue
		}
		ii, ferr := c.Finish()
		if ferr != nil {
			if tc.ferr {
				continue
			}
			t.Fatalf("h=%d finish err mismatch: tc.ferr:%v ferr!=nil:%v", k, tc.ferr, ferr != nil)
		}
		got := ii.String()
		wis, werr := makeIntervals(tc.exp...)
		if werr != nil {
			t.Fatalf("unexpected error on tc:%d %+v: %v", k, tc.exp, werr)
		}
		want := wis.String()
		if want != got {
			t.Fatalf("k=%d Ctor test: got %v want %v", k, got, want)
		}
	}
}

// makeIntervals constructs an Intervals object from the start/end
// sequence in nums, expected to be of the form
// s1,en1,st2,en2,...,stk,enk. Used only for unit testing.
func makeIntervals(nums ...int) (Intervals, error) {
	var r Intervals
	if len(nums)&1 != 0 {
		return r, fmt.Errorf("odd number of elems %d", len(nums))
	}
	for i := 0; i < len(nums); i += 2 {
		st := nums[i]
		en := nums[i+1]
		r = append(r, Interval{st: st, en: en})
	}
	return r, check(r)
}
