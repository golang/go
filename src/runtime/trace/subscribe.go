// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"internal/trace/tracev2"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	_ "unsafe"
)

var tracing traceMultiplexer

type traceMultiplexer struct {
	sync.Mutex
	enabled     atomic.Bool
	subscribers int

	subscribersMu    sync.Mutex
	traceStartWriter io.Writer
	flightRecorder   *recorder
}

func (t *traceMultiplexer) subscribeFlightRecorder(r *recorder) error {
	t.Lock()
	defer t.Unlock()

	t.subscribersMu.Lock()
	if t.flightRecorder != nil {
		t.subscribersMu.Unlock()
		return fmt.Errorf("flight recorder already enabled")
	}
	t.flightRecorder = r
	t.subscribersMu.Unlock()

	if err := t.addedSubscriber(); err != nil {
		t.subscribersMu.Lock()
		t.flightRecorder = nil
		t.subscribersMu.Unlock()
		return err
	}
	return nil
}

func (t *traceMultiplexer) unsubscribeFlightRecorder() error {
	t.Lock()
	defer t.Unlock()

	t.removingSubscriber()

	t.subscribersMu.Lock()
	if t.flightRecorder == nil {
		t.subscribersMu.Unlock()
		return fmt.Errorf("attempt to unsubscribe missing flight recorder")
	}
	t.flightRecorder = nil
	t.subscribersMu.Unlock()

	t.removedSubscriber()
	return nil
}

func (t *traceMultiplexer) subscribeTraceStartWriter(w io.Writer) error {
	t.Lock()
	defer t.Unlock()

	t.subscribersMu.Lock()
	if t.traceStartWriter != nil {
		t.subscribersMu.Unlock()
		return fmt.Errorf("execution tracer already enabled")
	}
	t.traceStartWriter = w
	t.subscribersMu.Unlock()

	if err := t.addedSubscriber(); err != nil {
		t.subscribersMu.Lock()
		t.traceStartWriter = nil
		t.subscribersMu.Unlock()
		return err
	}
	return nil
}

func (t *traceMultiplexer) unsubscribeTraceStartWriter() {
	t.Lock()
	defer t.Unlock()

	t.removingSubscriber()

	t.subscribersMu.Lock()
	if t.traceStartWriter == nil {
		t.subscribersMu.Unlock()
		return
	}
	t.traceStartWriter = nil
	t.subscribersMu.Unlock()

	t.removedSubscriber()
	return
}

func (t *traceMultiplexer) addedSubscriber() error {
	if t.enabled.Load() {
		// This is necessary for the trace reader goroutine to pick up on the new subscriber.
		runtime_traceAdvance(false)
	} else {
		if err := t.startLocked(); err != nil {
			return err
		}
	}
	t.subscribers++
	return nil
}

func (t *traceMultiplexer) removingSubscriber() {
	if t.subscribers == 0 {
		return
	}
	t.subscribers--
	if t.subscribers == 0 {
		runtime.StopTrace()
		t.enabled.Store(false)
	} else {
		// This is necessary to avoid missing trace data when the system is under high load.
		runtime_traceAdvance(false)
	}
}

func (t *traceMultiplexer) removedSubscriber() {
	if t.subscribers > 0 {
		// This is necessary for the trace reader goroutine to pick up on the new subscriber.
		runtime_traceAdvance(false)
	}
}

func (t *traceMultiplexer) startLocked() error {
	if err := runtime.StartTrace(); err != nil {
		return err
	}

	// Grab the trace reader goroutine's subscribers.
	//
	// We only update our subscribers if we see an end-of-generation
	// signal from the runtime after this, so any new subscriptions
	// or unsubscriptions must call traceAdvance to ensure the reader
	// goroutine sees an end-of-generation signal.
	t.subscribersMu.Lock()
	flightRecorder := t.flightRecorder
	traceStartWriter := t.traceStartWriter
	t.subscribersMu.Unlock()

	go func() {
		for {
			data := runtime_readTrace()
			if data == nil {
				break
			}
			if len(data) == 1 && tracev2.EventType(data[0]) == tracev2.EvEndOfGeneration {
				if flightRecorder != nil {
					flightRecorder.endGeneration()
				}

				// Pick up any changes.
				t.subscribersMu.Lock()
				flightRecorder = t.flightRecorder
				traceStartWriter = t.traceStartWriter
				t.subscribersMu.Unlock()
			} else {
				if traceStartWriter != nil {
					traceStartWriter.Write(data)
				}
				if flightRecorder != nil {
					flightRecorder.Write(data)
				}
			}
		}
	}()
	t.enabled.Store(true)
	return nil
}

//go:linkname runtime_readTrace
func runtime_readTrace() (buf []byte)
