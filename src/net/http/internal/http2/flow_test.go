// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import "testing"

func TestInFlowTake(t *testing.T) {
	var f inflow
	f.init(100)
	if !f.take(40) {
		t.Fatalf("f.take(40) from 100: got false, want true")
	}
	if !f.take(40) {
		t.Fatalf("f.take(40) from 60: got false, want true")
	}
	if f.take(40) {
		t.Fatalf("f.take(40) from 20: got true, want false")
	}
	if !f.take(20) {
		t.Fatalf("f.take(20) from 20: got false, want true")
	}
}

func TestInflowAddSmall(t *testing.T) {
	var f inflow
	f.init(0)
	// Adding even a small amount when there is no flow causes an immediate send.
	if got, want := f.add(1), int32(1); got != want {
		t.Fatalf("f.add(1) to 1 = %v, want %v", got, want)
	}
}

func TestInflowAdd(t *testing.T) {
	var f inflow
	f.init(10 * inflowMinRefresh)
	if got, want := f.add(inflowMinRefresh-1), int32(0); got != want {
		t.Fatalf("f.add(minRefresh - 1) = %v, want %v", got, want)
	}
	if got, want := f.add(1), int32(inflowMinRefresh); got != want {
		t.Fatalf("f.add(minRefresh) = %v, want %v", got, want)
	}
}

func TestTakeInflows(t *testing.T) {
	var a, b inflow
	a.init(10)
	b.init(20)
	if !takeInflows(&a, &b, 5) {
		t.Fatalf("takeInflows(a, b, 5) from 10, 20: got false, want true")
	}
	if takeInflows(&a, &b, 6) {
		t.Fatalf("takeInflows(a, b, 6) from 5, 15: got true, want false")
	}
	if !takeInflows(&a, &b, 5) {
		t.Fatalf("takeInflows(a, b, 5) from 5, 15: got false, want true")
	}
}

func TestOutFlow(t *testing.T) {
	var st outflow
	var conn outflow
	st.add(3)
	conn.add(2)

	if got, want := st.available(), int32(3); got != want {
		t.Errorf("available = %d; want %d", got, want)
	}
	st.setConnFlow(&conn)
	if got, want := st.available(), int32(2); got != want {
		t.Errorf("after parent setup, available = %d; want %d", got, want)
	}

	st.take(2)
	if got, want := conn.available(), int32(0); got != want {
		t.Errorf("after taking 2, conn = %d; want %d", got, want)
	}
	if got, want := st.available(), int32(0); got != want {
		t.Errorf("after taking 2, stream = %d; want %d", got, want)
	}
}

func TestOutFlowAdd(t *testing.T) {
	var f outflow
	if !f.add(1) {
		t.Fatal("failed to add 1")
	}
	if !f.add(-1) {
		t.Fatal("failed to add -1")
	}
	if got, want := f.available(), int32(0); got != want {
		t.Fatalf("size = %d; want %d", got, want)
	}
	if !f.add(1<<31 - 1) {
		t.Fatal("failed to add 2^31-1")
	}
	if got, want := f.available(), int32(1<<31-1); got != want {
		t.Fatalf("size = %d; want %d", got, want)
	}
	if f.add(1) {
		t.Fatal("adding 1 to max shouldn't be allowed")
	}
}

func TestOutFlowAddOverflow(t *testing.T) {
	var f outflow
	if !f.add(0) {
		t.Fatal("failed to add 0")
	}
	if !f.add(-1) {
		t.Fatal("failed to add -1")
	}
	if !f.add(0) {
		t.Fatal("failed to add 0")
	}
	if !f.add(1) {
		t.Fatal("failed to add 1")
	}
	if !f.add(1) {
		t.Fatal("failed to add 1")
	}
	if !f.add(0) {
		t.Fatal("failed to add 0")
	}
	if !f.add(-3) {
		t.Fatal("failed to add -3")
	}
	if got, want := f.available(), int32(-2); got != want {
		t.Fatalf("size = %d; want %d", got, want)
	}
	if !f.add(1<<31 - 1) {
		t.Fatal("failed to add 2^31-1")
	}
	if got, want := f.available(), int32(1+-3+(1<<31-1)); got != want {
		t.Fatalf("size = %d; want %d", got, want)
	}

}
