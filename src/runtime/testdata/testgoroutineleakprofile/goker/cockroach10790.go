// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/10790
 * Buggy version: 96b5452557ebe26bd9d85fe7905155009204d893
 * fix commit-id: f1a5c19125c65129b966fbdc0e6408e8df214aba
 * Flaky: 28/100
 * Description:
 *   It is possible that a message from ctxDone will make the function beginCmds
 * returns without draining the channel ch, so that goroutines created by anonymous
 * function will leak.
 */

package main

import (
	"context"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Cockroach10790", Cockroach10790)
}

type Replica_cockroach10790 struct {
	chans   []chan bool
}

func (r *Replica_cockroach10790) beginCmds(ctx context.Context) {
	ctxDone := ctx.Done()
	for _, ch := range r.chans {
		select {
		case <-ch:
		case <-ctxDone:
			go func() { // G3
				for _, ch := range r.chans {
					<-ch
				}
			}()
		}
	}
}

func (r *Replica_cockroach10790) sendChans(ctx context.Context) {
	for _, ch := range r.chans {
		select {
		case ch <- true:
		case <-ctx.Done():
			return
		}
	}
}

func NewReplica_cockroach10790() *Replica_cockroach10790 {
	r := &Replica_cockroach10790{}
	r.chans = append(r.chans, make(chan bool), make(chan bool))
	return r
}

// Example of goroutine leak trace:
//
// G1              G2                G3                     helper goroutine
//--------------------------------------------------------------------------------------
// .							 .                                        r.sendChans()
// r.beginCmds()   .                                        .
// .               .                                        ch1 <-
// <-ch1 <================================================> ch1 <-
// .               .                                        select [ch2<-, <-ctx.Done()]
// .               cancel()                                 .
// .               <<done>>                                 [<-ctx.Done()] ==> return
// .                                                        <<done>>
// go func() [G3]                                           .
// .                                 <-ch1
//	------------------------------G3 leaks----------------------------------------------
//

func Cockroach10790() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			r := NewReplica_cockroach10790()
			ctx, cancel := context.WithCancel(context.Background())
			go r.sendChans(ctx) // helper goroutine
			go r.beginCmds(ctx) // G1
			go cancel()         // G2
		}()
	}
}
