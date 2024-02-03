// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"time"
)

var (
	// if non-nil, overrides dialTCP.
	testHookDialTCP func(ctx context.Context, net string, laddr, raddr *TCPAddr) (*TCPConn, error)

	testHookLookupIP = func(
		ctx context.Context,
		fn func(context.Context, string, string) ([]IPAddr, error),
		network string,
		host string,
	) ([]IPAddr, error) {
		return fn(ctx, network, host)
	}
	testHookSetKeepAlive = func(time.Duration) {}

	// testHookStepTime sleeps until time has moved forward by a nonzero amount.
	// This helps to avoid flakes in timeout tests by ensuring that an implausibly
	// short deadline (such as 1ns in the future) is always expired by the time
	// a relevant system call occurs.
	testHookStepTime = func() {}
)
