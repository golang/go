// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag adds support for telemetry tracins.
package trace

import (
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
)

type TraceID [16]byte
type SpanID [8]byte

func (t TraceID) String() string {
	return fmt.Sprintf("%02x", t[:])
}

func (s SpanID) String() string {
	return fmt.Sprintf("%02x", s[:])
}

func (s SpanID) IsValid() bool {
	return s != SpanID{}
}

var (
	generationMu sync.Mutex
	nextSpanID   uint64
	spanIDInc    uint64

	traceIDAdd  [2]uint64
	traceIDRand *rand.Rand
)

func initGenerator() {
	var rngSeed int64
	for _, p := range []interface{}{
		&rngSeed, &traceIDAdd, &nextSpanID, &spanIDInc,
	} {
		binary.Read(crand.Reader, binary.LittleEndian, p)
	}
	traceIDRand = rand.New(rand.NewSource(rngSeed))
	spanIDInc |= 1
}

func newTraceID() TraceID {
	generationMu.Lock()
	defer generationMu.Unlock()
	if traceIDRand == nil {
		initGenerator()
	}
	var tid [16]byte
	binary.LittleEndian.PutUint64(tid[0:8], traceIDRand.Uint64()+traceIDAdd[0])
	binary.LittleEndian.PutUint64(tid[8:16], traceIDRand.Uint64()+traceIDAdd[1])
	return tid
}

func newSpanID() SpanID {
	var id uint64
	for id == 0 {
		id = atomic.AddUint64(&nextSpanID, spanIDInc)
	}
	var sid [8]byte
	binary.LittleEndian.PutUint64(sid[:], id)
	return sid
}
