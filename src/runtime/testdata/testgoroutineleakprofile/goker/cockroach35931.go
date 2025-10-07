// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Cockroach35931", Cockroach35931)
}

type RowReceiver_cockroach35931 interface {
	Push()
}

type inboundStreamInfo_cockroach35931 struct {
	receiver RowReceiver_cockroach35931
}

type RowChannel_cockroach35931 struct {
	dataChan chan struct{}
}

func (rc *RowChannel_cockroach35931) Push() {
	// The buffer size can be either 0 or 1 when this function is entered.
	// We need context sensitivity or a path-condition on the buffer size
	// to find this bug.
	rc.dataChan <- struct{}{}
}

func (rc *RowChannel_cockroach35931) initWithBufSizeAndNumSenders(chanBufSize int) {
	rc.dataChan = make(chan struct{}, chanBufSize)
}

type flowEntry_cockroach35931 struct {
	flow           *Flow_cockroach35931
	inboundStreams map[int]*inboundStreamInfo_cockroach35931
}

type flowRegistry_cockroach35931 struct {
	sync.Mutex
	flows map[int]*flowEntry_cockroach35931
}

func (fr *flowRegistry_cockroach35931) getEntryLocked(id int) *flowEntry_cockroach35931 {
	entry, ok := fr.flows[id]
	if !ok {
		entry = &flowEntry_cockroach35931{}
		fr.flows[id] = entry
	}
	return entry
}

func (fr *flowRegistry_cockroach35931) cancelPendingStreamsLocked(id int) []RowReceiver_cockroach35931 {
	entry := fr.flows[id]
	pendingReceivers := make([]RowReceiver_cockroach35931, 0)
	for _, is := range entry.inboundStreams {
		pendingReceivers = append(pendingReceivers, is.receiver)
	}
	return pendingReceivers
}

type Flow_cockroach35931 struct {
	id             int
	flowRegistry   *flowRegistry_cockroach35931
	inboundStreams map[int]*inboundStreamInfo_cockroach35931
}

func (f *Flow_cockroach35931) cancel() {
	f.flowRegistry.Lock()
	timedOutReceivers := f.flowRegistry.cancelPendingStreamsLocked(f.id)
	f.flowRegistry.Unlock()

	for _, receiver := range timedOutReceivers {
		receiver.Push()
	}
}

func (fr *flowRegistry_cockroach35931) RegisterFlow(f *Flow_cockroach35931, inboundStreams map[int]*inboundStreamInfo_cockroach35931) {
	entry := fr.getEntryLocked(f.id)
	entry.flow = f
	entry.inboundStreams = inboundStreams
}

func makeFlowRegistry_cockroach35931() *flowRegistry_cockroach35931 {
	return &flowRegistry_cockroach35931{
		flows: make(map[int]*flowEntry_cockroach35931),
	}
}

func Cockroach35931() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		fr := makeFlowRegistry_cockroach35931()

		left := &RowChannel_cockroach35931{}
		left.initWithBufSizeAndNumSenders(1)
		right := &RowChannel_cockroach35931{}
		right.initWithBufSizeAndNumSenders(1)

		inboundStreams := map[int]*inboundStreamInfo_cockroach35931{
			0: {
				receiver: left,
			},
			1: {
				receiver: right,
			},
		}

		left.Push()

		flow := &Flow_cockroach35931{
			id:             0,
			flowRegistry:   fr,
			inboundStreams: inboundStreams,
		}

		fr.RegisterFlow(flow, inboundStreams)

		flow.cancel()
	}()
}
