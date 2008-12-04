// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"testing";
	"time";
)

export func TestTick(t *testing.T) {
	const (
		Delta uint64 = 10*1e6;
		Count uint64 = 10;
	);
	c := Tick(Delta);
	t0, err := Nanoseconds();
	for i := 0; i < Count; i++ {
		<-c;
	}
	t1, err1 := Nanoseconds();
	ns := t1 - t0;
	target := int64(Delta*Count);
	slop := target*2/10;
	if ns < target - slop || ns > target + slop {
		t.Fatalf("%d ticks of %d ns took %d ns, expected %d", Count, Delta, ns, target);
	}
}
