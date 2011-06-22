// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os"
	"sync"
)

// A Ticker holds a synchronous channel that delivers `ticks' of a clock
// at intervals.
type Ticker struct {
	C        <-chan int64 // The channel on which the ticks are delivered.
	c        chan<- int64 // The same channel, but the end we use.
	ns       int64
	shutdown chan bool // Buffered channel used to signal shutdown.
	nextTick int64
	next     *Ticker
}

// Stop turns off a ticker.  After Stop, no more ticks will be sent.
func (t *Ticker) Stop() {
	select {
	case t.shutdown <- true:
		// ok
	default:
		// Stop in progress already
	}
}

// Tick is a convenience wrapper for NewTicker providing access to the ticking
// channel only.  Useful for clients that have no need to shut down the ticker.
func Tick(ns int64) <-chan int64 {
	if ns <= 0 {
		return nil
	}
	return NewTicker(ns).C
}

type alarmer struct {
	wakeUp   chan bool // wakeup signals sent/received here
	wakeMeAt chan int64
	wakeTime int64
}

// Set alarm to go off at time ns, if not already set earlier.
func (a *alarmer) set(ns int64) {
	switch {
	case a.wakeTime > ns:
		// Next tick we expect is too late; shut down the late runner
		// and (after fallthrough) start a new wakeLoop.
		close(a.wakeMeAt)
		fallthrough
	case a.wakeMeAt == nil:
		// There's no wakeLoop, start one.
		a.wakeMeAt = make(chan int64)
		a.wakeUp = make(chan bool, 1)
		go wakeLoop(a.wakeMeAt, a.wakeUp)
		fallthrough
	case a.wakeTime == 0:
		// Nobody else is waiting; it's just us.
		a.wakeTime = ns
		a.wakeMeAt <- ns
	default:
		// There's already someone scheduled.
	}
}

// Channel to notify tickerLoop of new Tickers being created.
var newTicker chan *Ticker

func startTickerLoop() {
	newTicker = make(chan *Ticker)
	go tickerLoop()
}

// wakeLoop delivers ticks at scheduled times, sleeping until the right moment.
// If another, earlier Ticker is created while it sleeps, tickerLoop() will start a new
// wakeLoop and signal that this one is done by closing the wakeMeAt channel.
func wakeLoop(wakeMeAt chan int64, wakeUp chan bool) {
	for wakeAt := range wakeMeAt {
		Sleep(wakeAt - Nanoseconds())
		wakeUp <- true
	}
}

// A single tickerLoop serves all ticks to Tickers.  It waits for two events:
// either the creation of a new Ticker or a tick from the alarm,
// signaling a time to wake up one or more Tickers.
func tickerLoop() {
	// Represents the next alarm to be delivered.
	var alarm alarmer
	var now, wakeTime int64
	var tickers *Ticker
	for {
		select {
		case t := <-newTicker:
			// Add Ticker to list
			t.next = tickers
			tickers = t
			// Arrange for a new alarm if this one precedes the existing one.
			alarm.set(t.nextTick)
		case <-alarm.wakeUp:
			now = Nanoseconds()
			wakeTime = now + 1e15 // very long in the future
			var prev *Ticker = nil
			// Scan list of tickers, delivering updates to those
			// that need it and determining the next wake time.
			// TODO(r): list should be sorted in time order.
			for t := tickers; t != nil; t = t.next {
				select {
				case <-t.shutdown:
					// Ticker is done; remove it from list.
					if prev == nil {
						tickers = t.next
					} else {
						prev.next = t.next
					}
					continue
				default:
				}
				if t.nextTick <= now {
					if len(t.c) == 0 {
						// Only send if there's room.  We must not block.
						// The channel is allocated with a one-element
						// buffer, which is sufficient: if he hasn't picked
						// up the last tick, no point in sending more.
						t.c <- now
					}
					t.nextTick += t.ns
					if t.nextTick <= now {
						// Still behind; advance in one big step.
						t.nextTick += (now - t.nextTick + t.ns) / t.ns * t.ns
					}
				}
				if t.nextTick < wakeTime {
					wakeTime = t.nextTick
				}
				prev = t
			}
			if tickers != nil {
				// Please send wakeup at earliest required time.
				// If there are no tickers, don't bother.
				alarm.wakeTime = wakeTime
				alarm.wakeMeAt <- wakeTime
			} else {
				alarm.wakeTime = 0
			}
		}
	}
}

var onceStartTickerLoop sync.Once

// NewTicker returns a new Ticker containing a channel that will
// send the time, in nanoseconds, every ns nanoseconds.  It adjusts the
// intervals to make up for pauses in delivery of the ticks. The value of
// ns must be greater than zero; if not, NewTicker will panic.
func NewTicker(ns int64) *Ticker {
	if ns <= 0 {
		panic(os.NewError("non-positive interval for NewTicker"))
	}
	c := make(chan int64, 1) //  See comment on send in tickerLoop
	t := &Ticker{
		C:        c,
		c:        c,
		ns:       ns,
		shutdown: make(chan bool, 1),
		nextTick: Nanoseconds() + ns,
	}
	onceStartTickerLoop.Do(startTickerLoop)
	// must be run in background so global Tickers can be created
	go func() { newTicker <- t }()
	return t
}
