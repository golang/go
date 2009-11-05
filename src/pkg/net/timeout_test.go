// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"testing";
	"time";
)

func testTimeout(t *testing.T, network, addr string) {
	fd, err := Dial(network, "", addr);
	defer fd.Close();
	if err != nil {
		t.Errorf("dial %s %s failed: %v", network, addr, err);
	}
	t0 := time.Nanoseconds();
	fd.SetReadTimeout(1e8);	// 100ms
	var b [100]byte;
	n, err1 := fd.Read(&b);
	t1 := time.Nanoseconds();
	if n != 0 || !isEAGAIN(err1) {
		t.Errorf("fd.Read on %s %s did not return 0, EAGAIN: %v, %v", network, addr, n, err1);
	}
	if t1-t0 < 0.5e8 || t1-t0 > 1.5e8 {
		t.Errorf("fd.Read on %s %s took %f seconds, expected 0.1", network, addr, float64(t1-t0)/1e9);
	}
}

func TestTimeoutUDP(t *testing.T) {
	testTimeout(t, "udp", "127.0.0.1:53");
}

func TestTimeoutTCP(t *testing.T) {
	// 74.125.19.99 is www.google.com.
	// could use dns, but dns depends on
	// timeouts and this is the timeout test.
	testTimeout(t, "tcp", "74.125.19.99:80");
}
