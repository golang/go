// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"syscall";
	"time"
)

// TODO(rsc): This implementation of time.Tick is a
// simple placeholder.  Eventually, there will need to be
// a single central time server no matter how many tickers
// are active.  There also needs to be a way to cancel a ticker.
//
// Also, if timeouts become part of the select statement,
// perhaps the Ticker is just:
//
//	func Ticker(ns int64, c *chan int64) {
//		for {
//			select { timeout ns: }
//			nsec, err := time.Nanoseconds();
//			c <- nsec;
//		}

func Ticker(ns int64, c *chan int64) {
	var tv syscall.Timeval;
	for {
		syscall.nstotimeval(ns, &tv);
		syscall.Syscall6(syscall.SYS_SELECT, 0, 0, 0, 0, syscall.TimevalPtr(&tv), 0);
		nsec, err := time.Nanoseconds();
		c <- nsec;
	}
}

export func Tick(ns int64) *chan int64 {
	c := new(chan int64);
	go Ticker(ns, c);
	return c;
}

