// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"internal/race"
	"io"
	"os"
	"os/signal"
	"reflect"
	"strconv"
	"testing"
	"time"
)

var benchmarkWorkerFlag = flag.Bool("benchmarkworker", false, "")

func TestMain(m *testing.M) {
	flag.Parse()
	if *benchmarkWorkerFlag {
		runBenchmarkWorker()
		return
	}
	os.Exit(m.Run())
}

func BenchmarkWorkerFuzzOverhead(b *testing.B) {
	if race.Enabled {
		b.Skip("TODO(48504): fix and re-enable")
	}
	origEnv := os.Getenv("GODEBUG")
	defer func() { os.Setenv("GODEBUG", origEnv) }()
	os.Setenv("GODEBUG", fmt.Sprintf("%s,fuzzseed=123", origEnv))

	ws := &workerServer{
		fuzzFn:     func(_ CorpusEntry) (time.Duration, error) { return time.Second, nil },
		workerComm: workerComm{memMu: make(chan *sharedMem, 1)},
	}

	mem, err := sharedMemTempFile(workerSharedMemSize)
	if err != nil {
		b.Fatalf("failed to create temporary shared memory file: %s", err)
	}
	defer func() {
		if err := mem.Close(); err != nil {
			b.Error(err)
		}
	}()

	initialVal := []any{make([]byte, 32)}
	encodedVals := marshalCorpusFile(initialVal...)
	mem.setValue(encodedVals)

	ws.memMu <- mem

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ws.m = newMutator()
		mem.setValue(encodedVals)
		mem.header().count = 0

		ws.fuzz(context.Background(), fuzzArgs{Limit: 1})
	}
}

// BenchmarkWorkerPing acts as the coordinator and measures the time it takes
// a worker to respond to N pings. This is a rough measure of our RPC latency.
func BenchmarkWorkerPing(b *testing.B) {
	if race.Enabled {
		b.Skip("TODO(48504): fix and re-enable")
	}
	b.SetParallelism(1)
	w := newWorkerForTest(b)
	for i := 0; i < b.N; i++ {
		if err := w.client.ping(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWorkerFuzz acts as the coordinator and measures the time it takes
// a worker to mutate a given input and call a trivial fuzz function N times.
func BenchmarkWorkerFuzz(b *testing.B) {
	if race.Enabled {
		b.Skip("TODO(48504): fix and re-enable")
	}
	b.SetParallelism(1)
	w := newWorkerForTest(b)
	entry := CorpusEntry{Values: []any{[]byte(nil)}}
	entry.Data = marshalCorpusFile(entry.Values...)
	for i := int64(0); i < int64(b.N); {
		args := fuzzArgs{
			Limit:   int64(b.N) - i,
			Timeout: workerFuzzDuration,
		}
		_, resp, _, err := w.client.fuzz(context.Background(), entry, args)
		if err != nil {
			b.Fatal(err)
		}
		if resp.Err != "" {
			b.Fatal(resp.Err)
		}
		if resp.Count == 0 {
			b.Fatal("worker did not make progress")
		}
		i += resp.Count
	}
}

// newWorkerForTest creates and starts a worker process for testing or
// benchmarking. The worker process calls RunFuzzWorker, which responds to
// RPC messages until it's stopped. The process is stopped and cleaned up
// automatically when the test is done.
func newWorkerForTest(tb testing.TB) *worker {
	tb.Helper()
	c, err := newCoordinator(CoordinateFuzzingOpts{
		Types: []reflect.Type{reflect.TypeOf([]byte(nil))},
		Log:   io.Discard,
	})
	if err != nil {
		tb.Fatal(err)
	}
	dir := ""             // same as self
	binPath := os.Args[0] // same as self
	args := append(os.Args[1:], "-benchmarkworker")
	env := os.Environ() // same as self
	w, err := newWorker(c, dir, binPath, args, env)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := w.cleanup(); err != nil {
			tb.Error(err)
		}
	})
	if err := w.startAndPing(context.Background()); err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := w.stop(); err != nil {
			tb.Error(err)
		}
	})
	return w
}

func runBenchmarkWorker() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	fn := func(CorpusEntry) error { return nil }
	if err := RunFuzzWorker(ctx, fn); err != nil && err != ctx.Err() {
		panic(err)
	}
}

func BenchmarkWorkerMinimize(b *testing.B) {
	if race.Enabled {
		b.Skip("TODO(48504): fix and re-enable")
	}

	ws := &workerServer{
		workerComm: workerComm{memMu: make(chan *sharedMem, 1)},
	}

	mem, err := sharedMemTempFile(workerSharedMemSize)
	if err != nil {
		b.Fatalf("failed to create temporary shared memory file: %s", err)
	}
	defer func() {
		if err := mem.Close(); err != nil {
			b.Error(err)
		}
	}()
	ws.memMu <- mem

	bytes := make([]byte, 1024)
	ctx := context.Background()
	for sz := 1; sz <= len(bytes); sz <<= 1 {
		sz := sz
		input := []any{bytes[:sz]}
		encodedVals := marshalCorpusFile(input...)
		mem = <-ws.memMu
		mem.setValue(encodedVals)
		ws.memMu <- mem
		b.Run(strconv.Itoa(sz), func(b *testing.B) {
			i := 0
			ws.fuzzFn = func(_ CorpusEntry) (time.Duration, error) {
				if i == 0 {
					i++
					return time.Second, errors.New("initial failure for deflake")
				}
				return time.Second, nil
			}
			for i := 0; i < b.N; i++ {
				b.SetBytes(int64(sz))
				ws.minimize(ctx, minimizeArgs{})
			}
		})
	}
}
