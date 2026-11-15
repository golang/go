// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"slices"
	"testing"
)

// unboundedSpanSrc holds one channel operation of each operand form the rule
// reports: a call, a selector, an index, a slice, a call as a send value, and
// a range over a channel with and without iteration variables.
const unboundedSpanSrc = `package p

import "time"

func f(int) int { return 0 }

func a(d time.Duration, t *time.Timer, chs []chan int, buf []byte, ch chan []byte, c chan int, v any) {
	<-time.After(time.Duration(2) * d)
	<-t.C
	<-chs[0]
	<-chs[len("]")]
	<-v.(chan int)
	<-(c)
	ch <- buf[:10]
	c <- f(1)
}

func b(c chan int) {
	for v := range c {
		_ = v
	}
	for range c {
	}
}
`

// TestUnboundedChannelOpSpan checks that the span of a finding covers the
// whole operation, through the closing token of its last operand: the call,
// index, or slice bracket that syntax.EndPos leaves out. A range over a
// channel starts at the range keyword. Tools splice by this byte range.
func TestUnboundedChannelOpSpan(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	want := []string{
		"<-time.After(time.Duration(2) * d)",
		"<-t.C",
		"<-chs[0]",
		`<-chs[len("]")]`,
		"<-v.(chan int)",
		"<-(c)",
		"ch <- buf[:10]",
		"c <- f(1)",
		"range c",
		"range c",
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		var got []string
		for _, r := range Run(check(t, kind, name, unboundedSpanSrc)) {
			if r.Rule == "correctness::unbounded_channel_op" {
				got = append(got, spanText(unboundedSpanSrc, r))
			}
		}
		if !slices.Equal(got, want) {
			t.Errorf("%s spans = %q, want %q", name, got, want)
		}
	}
}

const unboundedSegmentExemptSrc = `package p

import "context"

func segmentSend(ctx context.Context, xs []int, out chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).map(x => {
		out <- x
		return x
	})
}

func segmentRecv(ctx context.Context, xs []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).map(x => x + <-in)
}

func segmentFilter(ctx context.Context, xs []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).filter(x => x == <-in)
}

func segmentUnordered(ctx context.Context, xs []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).unordered().map(x => x + <-in)
}

func segmentSecondLambda(ctx context.Context, xs []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).filter(x => x > 0).map(x => x + <-in)
}

func segmentChannelSource(ctx context.Context, src <-chan int, in chan int) ([]int, error) {
	return src.group(ctx, limit = 2).map(x => x + <-in)
}

func segmentRange(ctx context.Context, xs []int, in chan int) error {
	for y := range xs.group(ctx, limit = 2).map(x => x + <-in) {
		_ = y
	}
	return nil
}

func segmentNestedStage(ctx context.Context, xs, ys []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).map(x => ys.group(ctx, limit = 3).map(y => y + <-in).sum()?)
}
`

const unboundedSegmentReportedSrc = `package p

import "context"

type store struct{}

func (store) group(n int) []int { return nil }

func plainSend(xs []int, out chan int) []int {
	return xs.map(x => {
		out <- x // want
		return x
	})
}

func plainRecv(xs []int, in chan int) []int {
	return xs.map(x => x + <-in) // want
}

func plainFilter(xs []int, in chan int) []int {
	return xs.filter(x => x == <-in) // want
}

func plainSecondLambda(xs []int, in chan int) []int {
	return xs.filter(x => x > 0).map(x => x + <-in) // want
}

func afterSegment(ctx context.Context, xs []int, in chan int) (bool, error) {
	return xs.group(ctx, limit = 2).map(x => x).any(x => x == <-in) // want
}

func mapAfterTerminalStage(ctx context.Context, xs []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).map(x => x).sortBy(x => x).map(x => x + <-in) // want
}

func methodNamedGroup(s store, in chan int) []int {
	return s.group(1).map(x => x + <-in) // want
}

func closureInSegment(ctx context.Context, xs []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).map(x => func() int {
		return <-in // want
	}())
}

func plainChainInSegment(ctx context.Context, xs, ys []int, in chan int) ([]int, error) {
	return xs.group(ctx, limit = 2).map(x => ys.map(y => y + <-in).sum()) // want
}
`

func TestUnboundedChannelOpSegmentLambdas(t *testing.T) {
	rule := finding{"correctness::unbounded_channel_op", Deny}
	expect(t, KindLgo, nil, testSource{"exempt.lgo", unboundedSegmentExemptSrc}, rule)
	expect(t, KindLgo, nil, testSource{"reported.lgo", unboundedSegmentReportedSrc}, rule)
}
