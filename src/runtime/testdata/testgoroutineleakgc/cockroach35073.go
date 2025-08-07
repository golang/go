package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"time"
)

func init() {
	register("Cockroach35073", Cockroach35073)
}

type ConsumerStatus_cockroach35073 uint32

const (
	NeedMoreRows_cockroach35073 ConsumerStatus_cockroach35073 = iota
	DrainRequested_cockroach35073
	ConsumerClosed_cockroach35073
)

const rowChannelBufSize_cockroach35073 = 16
const outboxBufRows_cockroach35073 = 16

type rowSourceBase_cockroach35073 struct {
	consumerStatus ConsumerStatus_cockroach35073
}

func (rb *rowSourceBase_cockroach35073) consumerClosed() {
	atomic.StoreUint32((*uint32)(&rb.consumerStatus), uint32(ConsumerClosed_cockroach35073))
}

type RowChannelMsg_cockroach35073 int

type RowChannel_cockroach35073 struct {
	rowSourceBase_cockroach35073
	dataChan chan RowChannelMsg_cockroach35073
}

func (rc *RowChannel_cockroach35073) ConsumerClosed() {
	rc.consumerClosed()
	select {
	case <-rc.dataChan:
	default:
	}
}

func (rc *RowChannel_cockroach35073) Push() ConsumerStatus_cockroach35073 {
	consumerStatus := ConsumerStatus_cockroach35073(
		atomic.LoadUint32((*uint32)(&rc.consumerStatus)))
	switch consumerStatus {
	case NeedMoreRows_cockroach35073:
		rc.dataChan <- RowChannelMsg_cockroach35073(0)
	case DrainRequested_cockroach35073:
	case ConsumerClosed_cockroach35073:
	}
	return consumerStatus
}

func (rc *RowChannel_cockroach35073) InitWithNumSenders() {
	rc.initWithBufSizeAndNumSenders(rowChannelBufSize_cockroach35073)
}

func (rc *RowChannel_cockroach35073) initWithBufSizeAndNumSenders(chanBufSize int) {
	rc.dataChan = make(chan RowChannelMsg_cockroach35073, chanBufSize)
}

type outbox_cockroach35073 struct {
	RowChannel_cockroach35073
}

func (m *outbox_cockroach35073) init() {
	m.RowChannel_cockroach35073.InitWithNumSenders()
}

func (m *outbox_cockroach35073) start(wg *sync.WaitGroup) {
	if wg != nil {
		wg.Add(1)
	}
	go m.run(wg)
}

func (m *outbox_cockroach35073) run(wg *sync.WaitGroup) {
	if wg != nil {
		wg.Done()
	}
}

func Cockroach35073() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		// deadlocks: 1
		outbox := &outbox_cockroach35073{}
		outbox.init()

		var wg sync.WaitGroup
		for i := 0; i < outboxBufRows_cockroach35073; i++ {
			outbox.Push()
		}

		var blockedPusherWg sync.WaitGroup
		blockedPusherWg.Add(1)
		go func() {
			// deadlocks: 1
			outbox.Push()
			blockedPusherWg.Done()
		}()

		outbox.start(&wg)

		wg.Wait()
		outbox.RowChannel_cockroach35073.Push()
	}()
}
