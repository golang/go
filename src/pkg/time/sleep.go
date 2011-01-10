// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os"
	"syscall"
	"sync"
	"container/heap"
)

// The event type represents a single After or AfterFunc event.
type event struct {
	t        int64       // The absolute time that the event should fire.
	f        func(int64) // The function to call when the event fires.
	sleeping bool        // A sleeper is sleeping for this event.
}

type eventHeap []*event

var events eventHeap
var eventMutex sync.Mutex

func init() {
	events.Push(&event{1 << 62, nil, true}) // sentinel
}

// Sleep pauses the current goroutine for at least ns nanoseconds.
// Higher resolution sleeping may be provided by syscall.Nanosleep 
// on some operating systems.
func Sleep(ns int64) os.Error {
	_, err := sleep(Nanoseconds(), ns)
	return err
}

// sleep takes the current time and a duration,
// pauses for at least ns nanoseconds, and
// returns the current time and an error.
func sleep(t, ns int64) (int64, os.Error) {
	// TODO(cw): use monotonic-time once it's available
	end := t + ns
	for t < end {
		errno := syscall.Sleep(end - t)
		if errno != 0 && errno != syscall.EINTR {
			return 0, os.NewSyscallError("sleep", errno)
		}
		t = Nanoseconds()
	}
	return t, nil
}

// After waits at least ns nanoseconds before sending the current time
// on the returned channel.
func After(ns int64) <-chan int64 {
	c := make(chan int64, 1)
	after(ns, func(t int64) { c <- t })
	return c
}

// AfterFunc waits at least ns nanoseconds before calling f
// in its own goroutine.
func AfterFunc(ns int64, f func()) {
	after(ns, func(_ int64) {
		go f()
	})
}

// after is the implementation of After and AfterFunc.
// When the current time is after ns, it calls f with the current time.
// It assumes that f will not block.
func after(ns int64, f func(int64)) {
	t := Nanoseconds() + ns
	eventMutex.Lock()
	t0 := events[0].t
	heap.Push(events, &event{t, f, false})
	if t < t0 {
		go sleeper()
	}
	eventMutex.Unlock()
}

// sleeper continually looks at the earliest event in the queue, marks it
// as sleeping, waits until it happens, then removes any events
// in the queue that are due. It stops when it finds an event that is
// already marked as sleeping. When an event is inserted before the first item,
// a new sleeper is started.
//
// Scheduling vagaries mean that sleepers may not wake up in
// exactly the order of the events that they are waiting for,
// but this does not matter as long as there are at least as
// many sleepers as events marked sleeping (invariant). This ensures that
// there is always a sleeper to service the remaining events.
//
// A sleeper will remove at least the event it has been waiting for
// unless the event has already been removed by another sleeper.  Both
// cases preserve the invariant described above.
func sleeper() {
	eventMutex.Lock()
	e := events[0]
	for !e.sleeping {
		t := Nanoseconds()
		if dt := e.t - t; dt > 0 {
			e.sleeping = true
			eventMutex.Unlock()
			if nt, err := sleep(t, dt); err != nil {
				// If sleep has encountered an error,
				// there's not much we can do. We pretend
				// that time really has advanced by the required
				// amount and lie to the rest of the system.
				t = e.t
			} else {
				t = nt
			}
			eventMutex.Lock()
			e = events[0]
		}
		for t >= e.t {
			e.f(t)
			heap.Pop(events)
			e = events[0]
		}
	}
	eventMutex.Unlock()
}

func (eventHeap) Len() int {
	return len(events)
}

func (eventHeap) Less(i, j int) bool {
	return events[i].t < events[j].t
}

func (eventHeap) Swap(i, j int) {
	events[i], events[j] = events[j], events[i]
}

func (eventHeap) Push(x interface{}) {
	events = append(events, x.(*event))
}

func (eventHeap) Pop() interface{} {
	// TODO: possibly shrink array.
	n := len(events) - 1
	e := events[n]
	events[n] = nil
	events = events[0:n]
	return e
}
