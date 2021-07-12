// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"context"
	"fmt"
	"os"
	"testing"
)

func BenchmarkWorkerFuzzOverhead(b *testing.B) {
	origEnv := os.Getenv("GODEBUG")
	defer func() { os.Setenv("GODEBUG", origEnv) }()
	os.Setenv("GODEBUG", fmt.Sprintf("%s,fuzzseed=123", origEnv))

	ws := &workerServer{
		fuzzFn:     func(_ CorpusEntry) error { return nil },
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

	initialVal := []interface{}{make([]byte, 32)}
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
