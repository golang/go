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
// reports: a call, a selector, an index, a slice, and a call as a send value.
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
`

// TestUnboundedChannelOpSpan checks that the span of a finding covers the
// whole operation, through the closing token of its last operand: the call,
// index, or slice bracket that syntax.EndPos leaves out. Tools splice by this
// byte range.
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
