// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

// TODO(rsc): This implementation of Tick is a
// simple placeholder.  Eventually, there will need to be
// a single central time server no matter how many tickers
// are active.
//
// Also, if timeouts become part of the select statement,
// perhaps the Ticker is just:
//
//	func Ticker(ns int64, c chan int64) {
//		for {
//			select { timeout ns: }
//			nsec, err := Nanoseconds();
//			c <- nsec;
//		}


// A Ticker holds a synchronous channel that delivers `ticks' of a clock
// at intervals.
type Ticker struct {
	C        <-chan int64 // The channel on which the ticks are delivered.
	done     chan bool
	ns       int64
	shutdown bool
}

// Stop turns off a ticker.  After Stop, no more ticks will be sent.
func (t *Ticker) Stop() {
	t.shutdown = true
	go t.drain()
}

func (t *Ticker) drain() {
	for {
		select {
		case <-t.C:
		case <-t.done:
			return
		}
	}
}

func (t *Ticker) ticker(c chan<- int64) {
	now := Nanoseconds()
	when := now
	for !t.shutdown {
		when += t.ns // next alarm

		// if c <- now took too long, skip ahead
		if when < now {
			// one big step
			when += (now - when) / t.ns * t.ns
		}
		for when <= now {
			// little steps until when > now
			when += t.ns
		}

		for !t.shutdown && when > now {
			// limit individual sleeps so that stopped
			// long-term tickers don't pile up.
			const maxSleep = 1e9
			if when-now > maxSleep {
				Sleep(maxSleep)
			} else {
				Sleep(when - now)
			}
			now = Nanoseconds()
		}
		if t.shutdown {
			break
		}
		c <- now
	}
	t.done <- true
}

// Tick is a convenience wrapper for NewTicker providing access to the ticking
// channel only.  Useful for clients that have no need to shut down the ticker.
func Tick(ns int64) <-chan int64 {
	if ns <= 0 {
		return nil
	}
	return NewTicker(ns).C
}

// Ticker returns a new Ticker containing a synchronous channel that will
// send the time, in nanoseconds, every ns nanoseconds.  It adjusts the
// intervals to make up for pauses in delivery of the ticks.
func NewTicker(ns int64) *Ticker {
	if ns <= 0 {
		return nil
	}
	c := make(chan int64)
	t := &Ticker{c, make(chan bool), ns, false}
	go t.ticker(c)
	return t
}
