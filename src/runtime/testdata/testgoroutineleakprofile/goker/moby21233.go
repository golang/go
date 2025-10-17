// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/21233
 * Buggy version: cc12d2bfaae135e63b1f962ad80e6943dd995337
 * fix commit-id: 2f4aa9658408ac72a598363c6e22eadf93dbb8a7
 * Flaky:100/100
 */
package main

import (
	"math/rand"
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Moby21233", Moby21233)
}

type Progress_moby21233 struct{}

type Output_moby21233 interface {
	WriteProgress(Progress_moby21233) error
}

type chanOutput_moby21233 chan<- Progress_moby21233

type TransferManager_moby21233 struct {
	mu sync.Mutex
}

type Transfer_moby21233 struct {
	mu sync.Mutex
}

type Watcher_moby21233 struct {
	signalChan  chan struct{}
	releaseChan chan struct{}
	running     chan struct{}
}

func ChanOutput_moby21233(progressChan chan<- Progress_moby21233) Output_moby21233 {
	return chanOutput_moby21233(progressChan)
}
func (out chanOutput_moby21233) WriteProgress(p Progress_moby21233) error {
	out <- p
	return nil
}
func NewTransferManager_moby21233() *TransferManager_moby21233 {
	return &TransferManager_moby21233{}
}
func NewTransfer_moby21233() *Transfer_moby21233 {
	return &Transfer_moby21233{}
}
func (t *Transfer_moby21233) Release(watcher *Watcher_moby21233) {
	t.mu.Lock()
	t.mu.Unlock()
	close(watcher.releaseChan)
	<-watcher.running
}
func (t *Transfer_moby21233) Watch(progressOutput Output_moby21233) *Watcher_moby21233 {
	t.mu.Lock()
	defer t.mu.Unlock()
	lastProgress := Progress_moby21233{}
	w := &Watcher_moby21233{
		releaseChan: make(chan struct{}),
		signalChan:  make(chan struct{}),
		running:     make(chan struct{}),
	}
	go func() { // G2
		defer func() {
			close(w.running)
		}()
		done := false
		for {
			t.mu.Lock()
			t.mu.Unlock()
			if rand.Int31n(2) >= 1 {
				progressOutput.WriteProgress(lastProgress)
			}
			if done {
				return
			}
			select {
			case <-w.signalChan:
			case <-w.releaseChan:
				done = true
			}
		}
	}()
	return w
}
func (tm *TransferManager_moby21233) Transfer(progressOutput Output_moby21233) (*Transfer_moby21233, *Watcher_moby21233) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	t := NewTransfer_moby21233()
	return t, t.Watch(progressOutput)
}

func testTransfer_moby21233() { // G1
	tm := NewTransferManager_moby21233()
	progressChan := make(chan Progress_moby21233)
	progressDone := make(chan struct{})
	go func() { // G3
		time.Sleep(1 * time.Millisecond)
		for p := range progressChan { /// Chan consumer
			if rand.Int31n(2) >= 1 {
				return
			}
			_ = p
		}
		close(progressDone)
	}()
	time.Sleep(1 * time.Millisecond)
	ids := []string{"id1", "id2", "id3"}
	xrefs := make([]*Transfer_moby21233, len(ids))
	watchers := make([]*Watcher_moby21233, len(ids))
	for i := range ids {
		xrefs[i], watchers[i] = tm.Transfer(ChanOutput_moby21233(progressChan)) /// Chan producer
		time.Sleep(2 * time.Millisecond)
	}

	for i := range xrefs {
		xrefs[i].Release(watchers[i])
	}

	close(progressChan)
	<-progressDone
}

func Moby21233() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		go testTransfer_moby21233() // G1
	}
}
