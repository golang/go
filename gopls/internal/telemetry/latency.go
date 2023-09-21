// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"golang.org/x/telemetry/counter"
)

// latencyKey is used for looking up latency counters.
type latencyKey struct {
	operation, bucket string
	isError           bool
}

var (
	latencyBuckets = []struct {
		end  time.Duration
		name string
	}{
		{10 * time.Millisecond, "<10ms"},
		{50 * time.Millisecond, "<50ms"},
		{100 * time.Millisecond, "<100ms"},
		{200 * time.Millisecond, "<200ms"},
		{500 * time.Millisecond, "<500ms"},
		{1 * time.Second, "<1s"},
		{5 * time.Second, "<5s"},
		{24 * time.Hour, "<24h"},
	}

	latencyCounterMu sync.Mutex
	latencyCounters  = make(map[latencyKey]*counter.Counter) // lazily populated
)

// ForEachLatencyCounter runs the provided function for each current latency
// counter measuring the given operation.
//
// Exported for testing.
func ForEachLatencyCounter(operation string, isError bool, f func(*counter.Counter)) {
	latencyCounterMu.Lock()
	defer latencyCounterMu.Unlock()

	for k, v := range latencyCounters {
		if k.operation == operation && k.isError == isError {
			f(v)
		}
	}
}

// getLatencyCounter returns the counter used to record latency of the given
// operation in the given bucket.
func getLatencyCounter(operation, bucket string, isError bool) *counter.Counter {
	latencyCounterMu.Lock()
	defer latencyCounterMu.Unlock()

	key := latencyKey{operation, bucket, isError}
	c, ok := latencyCounters[key]
	if !ok {
		var name string
		if isError {
			name = fmt.Sprintf("gopls/%s/error-latency:%s", operation, bucket)
		} else {
			name = fmt.Sprintf("gopls/%s/latency:%s", operation, bucket)
		}
		c = counter.New(name)
		latencyCounters[key] = c
	}
	return c
}

// StartLatencyTimer starts a timer for the gopls operation with the given
// name, and returns a func to stop the timer and record the latency sample.
//
// If the context provided to the the resulting func is done, no observation is
// recorded.
func StartLatencyTimer(operation string) func(context.Context, error) {
	start := time.Now()
	return func(ctx context.Context, err error) {
		if errors.Is(ctx.Err(), context.Canceled) {
			// Ignore timing where the operation is cancelled, it may be influenced
			// by client behavior.
			return
		}
		latency := time.Since(start)
		bucketIdx := sort.Search(len(latencyBuckets), func(i int) bool {
			bucket := latencyBuckets[i]
			return latency < bucket.end
		})
		if bucketIdx < len(latencyBuckets) { // ignore latency longer than a day :)
			bucketName := latencyBuckets[bucketIdx].name
			getLatencyCounter(operation, bucketName, err != nil).Inc()
		}
	}
}
