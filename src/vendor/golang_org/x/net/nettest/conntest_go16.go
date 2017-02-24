// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.7

package nettest

import "testing"

func testConn(t *testing.T, mp MakePipe) {
	// Avoid using subtests on Go 1.6 and below.
	timeoutWrapper(t, mp, testBasicIO)
	timeoutWrapper(t, mp, testPingPong)
	timeoutWrapper(t, mp, testRacyRead)
	timeoutWrapper(t, mp, testRacyWrite)
	timeoutWrapper(t, mp, testReadTimeout)
	timeoutWrapper(t, mp, testWriteTimeout)
	timeoutWrapper(t, mp, testPastTimeout)
	timeoutWrapper(t, mp, testPresentTimeout)
	timeoutWrapper(t, mp, testFutureTimeout)
	timeoutWrapper(t, mp, testCloseTimeout)
	timeoutWrapper(t, mp, testConcurrentMethods)
}
