// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"strings"
	"testing"
	. "time"
)

func TestHasMonotonicClock(t *testing.T) {
	yes := func(expr string, tt Time) {
		if GetMono(&tt) == 0 {
			t.Errorf("%s: missing monotonic clock reading", expr)
		}
	}
	no := func(expr string, tt Time) {
		if GetMono(&tt) != 0 {
			t.Errorf("%s: unexpected monotonic clock reading", expr)
		}
	}

	yes("<-After(1)", <-After(1))
	ticker := NewTicker(1)
	yes("<-Tick(1)", <-ticker.C)
	ticker.Stop()
	no("Date(2009, 11, 23, 0, 0, 0, 0, UTC)", Date(2009, 11, 23, 0, 0, 0, 0, UTC))
	tp, _ := Parse(UnixDate, "Sat Mar  7 11:06:39 PST 2015")
	no(`Parse(UnixDate, "Sat Mar  7 11:06:39 PST 2015")`, tp)
	no("Unix(1486057371, 0)", Unix(1486057371, 0))

	yes("Now()", Now())

	tu := Unix(1486057371, 0)
	tm := tu
	SetMono(&tm, 123456)
	no("tu", tu)
	yes("tm", tm)

	no("tu.Add(1)", tu.Add(1))
	no("tu.In(UTC)", tu.In(UTC))
	no("tu.AddDate(1, 1, 1)", tu.AddDate(1, 1, 1))
	no("tu.AddDate(0, 0, 0)", tu.AddDate(0, 0, 0))
	no("tu.Local()", tu.Local())
	no("tu.UTC()", tu.UTC())
	no("tu.Round(2)", tu.Round(2))
	no("tu.Truncate(2)", tu.Truncate(2))

	yes("tm.Add(1)", tm.Add(1))
	no("tm.AddDate(1, 1, 1)", tm.AddDate(1, 1, 1))
	no("tm.AddDate(0, 0, 0)", tm.AddDate(0, 0, 0))
	no("tm.In(UTC)", tm.In(UTC))
	no("tm.Local()", tm.Local())
	no("tm.UTC()", tm.UTC())
	no("tm.Round(2)", tm.Round(2))
	no("tm.Truncate(2)", tm.Truncate(2))
}

func TestMonotonicAdd(t *testing.T) {
	tm := Unix(1486057371, 123456)
	SetMono(&tm, 123456789012345)

	t2 := tm.Add(1e8)
	if t2.Nanosecond() != 100123456 {
		t.Errorf("t2.Nanosecond() = %d, want 100123456", t2.Nanosecond())
	}
	if GetMono(&t2) != 123456889012345 {
		t.Errorf("t2.mono = %d, want 123456889012345", GetMono(&t2))
	}

	t3 := tm.Add(-9e18) // wall now out of range
	if t3.Nanosecond() != 123456 {
		t.Errorf("t3.Nanosecond() = %d, want 123456", t3.Nanosecond())
	}
	if GetMono(&t3) != 0 {
		t.Errorf("t3.mono = %d, want 0 (wall time out of range for monotonic reading)", GetMono(&t3))
	}

	t4 := tm.Add(+9e18) // wall now out of range
	if t4.Nanosecond() != 123456 {
		t.Errorf("t4.Nanosecond() = %d, want 123456", t4.Nanosecond())
	}
	if GetMono(&t4) != 0 {
		t.Errorf("t4.mono = %d, want 0 (wall time out of range for monotonic reading)", GetMono(&t4))
	}

	tn := Now()
	tn1 := tn.Add(1 * Hour)
	Sleep(100 * Millisecond)
	d := Until(tn1)
	if d < 59*Minute {
		t.Errorf("Until(Now().Add(1*Hour)) = %v, wanted at least 59m", d)
	}
	now := Now()
	if now.After(tn1) {
		t.Errorf("Now().After(Now().Add(1*Hour)) = true, want false")
	}
	if !tn1.After(now) {
		t.Errorf("Now().Add(1*Hour).After(now) = false, want true")
	}
	if tn1.Before(now) {
		t.Errorf("Now().Add(1*Hour).Before(Now()) = true, want false")
	}
	if !now.Before(tn1) {
		t.Errorf("Now().Before(Now().Add(1*Hour)) = false, want true")
	}
	if got, want := now.Compare(tn1), -1; got != want {
		t.Errorf("Now().Compare(Now().Add(1*Hour)) = %d, want %d", got, want)
	}
	if got, want := tn1.Compare(now), 1; got != want {
		t.Errorf("Now().Add(1*Hour).Compare(Now()) = %d, want %d", got, want)
	}
}

func TestMonotonicSub(t *testing.T) {
	t1 := Unix(1483228799, 995e6)
	SetMono(&t1, 123456789012345)

	t2 := Unix(1483228799, 5e6)
	SetMono(&t2, 123456789012345+10e6)

	t3 := Unix(1483228799, 995e6)
	SetMono(&t3, 123456789012345+1e9)

	t1w := t1.AddDate(0, 0, 0)
	if GetMono(&t1w) != 0 {
		t.Fatalf("AddDate didn't strip monotonic clock reading")
	}
	t2w := t2.AddDate(0, 0, 0)
	if GetMono(&t2w) != 0 {
		t.Fatalf("AddDate didn't strip monotonic clock reading")
	}
	t3w := t3.AddDate(0, 0, 0)
	if GetMono(&t3w) != 0 {
		t.Fatalf("AddDate didn't strip monotonic clock reading")
	}

	sub := func(txs, tys string, tx, txw, ty, tyw Time, d, dw Duration) {
		check := func(expr string, d, want Duration) {
			if d != want {
				t.Errorf("%s = %v, want %v", expr, d, want)
			}
		}
		check(txs+".Sub("+tys+")", tx.Sub(ty), d)
		check(txs+"w.Sub("+tys+")", txw.Sub(ty), dw)
		check(txs+".Sub("+tys+"w)", tx.Sub(tyw), dw)
		check(txs+"w.Sub("+tys+"w)", txw.Sub(tyw), dw)
	}
	sub("t1", "t1", t1, t1w, t1, t1w, 0, 0)
	sub("t1", "t2", t1, t1w, t2, t2w, -10*Millisecond, 990*Millisecond)
	sub("t1", "t3", t1, t1w, t3, t3w, -1000*Millisecond, 0)

	sub("t2", "t1", t2, t2w, t1, t1w, 10*Millisecond, -990*Millisecond)
	sub("t2", "t2", t2, t2w, t2, t2w, 0, 0)
	sub("t2", "t3", t2, t2w, t3, t3w, -990*Millisecond, -990*Millisecond)

	sub("t3", "t1", t3, t3w, t1, t1w, 1000*Millisecond, 0)
	sub("t3", "t2", t3, t3w, t2, t2w, 990*Millisecond, 990*Millisecond)
	sub("t3", "t3", t3, t3w, t3, t3w, 0, 0)

	cmp := func(txs, tys string, tx, txw, ty, tyw Time, c, cw int) {
		check := func(expr string, b, want any) {
			if b != want {
				t.Errorf("%s = %v, want %v", expr, b, want)
			}
		}
		check(txs+".After("+tys+")", tx.After(ty), c > 0)
		check(txs+"w.After("+tys+")", txw.After(ty), cw > 0)
		check(txs+".After("+tys+"w)", tx.After(tyw), cw > 0)
		check(txs+"w.After("+tys+"w)", txw.After(tyw), cw > 0)

		check(txs+".Before("+tys+")", tx.Before(ty), c < 0)
		check(txs+"w.Before("+tys+")", txw.Before(ty), cw < 0)
		check(txs+".Before("+tys+"w)", tx.Before(tyw), cw < 0)
		check(txs+"w.Before("+tys+"w)", txw.Before(tyw), cw < 0)

		check(txs+".Equal("+tys+")", tx.Equal(ty), c == 0)
		check(txs+"w.Equal("+tys+")", txw.Equal(ty), cw == 0)
		check(txs+".Equal("+tys+"w)", tx.Equal(tyw), cw == 0)
		check(txs+"w.Equal("+tys+"w)", txw.Equal(tyw), cw == 0)

		check(txs+".Compare("+tys+")", tx.Compare(ty), c)
		check(txs+"w.Compare("+tys+")", txw.Compare(ty), cw)
		check(txs+".Compare("+tys+"w)", tx.Compare(tyw), cw)
		check(txs+"w.Compare("+tys+"w)", txw.Compare(tyw), cw)
	}

	cmp("t1", "t1", t1, t1w, t1, t1w, 0, 0)
	cmp("t1", "t2", t1, t1w, t2, t2w, -1, +1)
	cmp("t1", "t3", t1, t1w, t3, t3w, -1, 0)

	cmp("t2", "t1", t2, t2w, t1, t1w, +1, -1)
	cmp("t2", "t2", t2, t2w, t2, t2w, 0, 0)
	cmp("t2", "t3", t2, t2w, t3, t3w, -1, -1)

	cmp("t3", "t1", t3, t3w, t1, t1w, +1, 0)
	cmp("t3", "t2", t3, t3w, t2, t2w, +1, +1)
	cmp("t3", "t3", t3, t3w, t3, t3w, 0, 0)
}

func TestMonotonicOverflow(t *testing.T) {
	t1 := Now().Add(-30 * Second)
	d := Until(t1)
	if d < -35*Second || -30*Second < d {
		t.Errorf("Until(Now().Add(-30s)) = %v, want roughly -30s (-35s to -30s)", d)
	}

	t1 = Now().Add(30 * Second)
	d = Until(t1)
	if d < 25*Second || 30*Second < d {
		t.Errorf("Until(Now().Add(-30s)) = %v, want roughly 30s (25s to 30s)", d)
	}

	t0 := Now()
	t1 = t0.Add(Duration(1<<63 - 1))
	if GetMono(&t1) != 0 {
		t.Errorf("Now().Add(maxDuration) has monotonic clock reading (%v => %v %d %d)", t0.String(), t1.String(), t0.Unix(), t1.Unix())
	}
	t2 := t1.Add(-Duration(1<<63 - 1))
	d = Since(t2)
	if d < -10*Second || 10*Second < d {
		t.Errorf("Since(Now().Add(max).Add(-max)) = %v, want [-10s, 10s]", d)
	}

	t0 = Now()
	t1 = t0.Add(1 * Hour)
	Sleep(100 * Millisecond)
	t2 = Now().Add(-5 * Second)
	if !t1.After(t2) {
		t.Errorf("Now().Add(1*Hour).After(Now().Add(-5*Second)) = false, want true\nt1=%v\nt2=%v", t1, t2)
	}
	if t2.After(t1) {
		t.Errorf("Now().Add(-5*Second).After(Now().Add(1*Hour)) = true, want false\nt1=%v\nt2=%v", t1, t2)
	}
	if t1.Before(t2) {
		t.Errorf("Now().Add(1*Hour).Before(Now().Add(-5*Second)) = true, want false\nt1=%v\nt2=%v", t1, t2)
	}
	if !t2.Before(t1) {
		t.Errorf("Now().Add(-5*Second).Before(Now().Add(1*Hour)) = false, want true\nt1=%v\nt2=%v", t1, t2)
	}
	if got, want := t1.Compare(t2), 1; got != want {
		t.Errorf("Now().Add(1*Hour).Compare(Now().Add(-5*Second)) = %d, want %d\nt1=%v\nt2=%v", got, want, t1, t2)
	}
	if got, want := t2.Compare(t1), -1; got != want {
		t.Errorf("Now().Add(-5*Second).Before(Now().Add(1*Hour)) = %d, want %d\nt1=%v\nt2=%v", got, want, t1, t2)
	}
}

var monotonicStringTests = []struct {
	mono int64
	want string
}{
	{0, "m=+0.000000000"},
	{123456789, "m=+0.123456789"},
	{-123456789, "m=-0.123456789"},
	{123456789000, "m=+123.456789000"},
	{-123456789000, "m=-123.456789000"},
	{9e18, "m=+9000000000.000000000"},
	{-9e18, "m=-9000000000.000000000"},
	{-1 << 63, "m=-9223372036.854775808"},
}

func TestMonotonicString(t *testing.T) {
	t1 := Now()
	t.Logf("Now() = %v", t1)

	for _, tt := range monotonicStringTests {
		t1 := Now()
		SetMono(&t1, tt.mono)
		s := t1.String()
		got := s[strings.LastIndex(s, " ")+1:]
		if got != tt.want {
			t.Errorf("with mono=%d: got %q; want %q", tt.mono, got, tt.want)
		}
	}
}
