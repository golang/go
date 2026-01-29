// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: grpc
 * Issue or PR  : https://github.com/grpc/grpc-go/pull/1460
 * Buggy version: 7db1564ba1229bc42919bb1f6d9c4186f3aa8678
 * fix commit-id: e605a1ecf24b634f94f4eefdab10a9ada98b70dd
 * Flaky: 100/100
 */
package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Grpc1460", Grpc1460)
}

type Stream_grpc1460 struct{}

type http2Client_grpc1460 struct {
	mu              sync.Mutex
	awakenKeepalive chan struct{}
	activeStream    []*Stream_grpc1460
}

func (t *http2Client_grpc1460) keepalive() {
	t.mu.Lock()
	if len(t.activeStream) < 1 {
		<-t.awakenKeepalive
		runtime.Gosched()
		t.mu.Unlock()
	} else {
		t.mu.Unlock()
	}
}

func (t *http2Client_grpc1460) NewStream() {
	t.mu.Lock()
	runtime.Gosched()
	t.activeStream = append(t.activeStream, &Stream_grpc1460{})
	if len(t.activeStream) == 1 {
		select {
		case t.awakenKeepalive <- struct{}{}:
		default:
		}
	}
	t.mu.Unlock()
}

///
/// G1 						G2
/// client.keepalive()
/// 						client.NewStream()
/// t.mu.Lock()
/// <-t.awakenKeepalive
/// 						t.mu.Lock()
/// ---------------G1, G2 deadlock--------------
///

func Grpc1460() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 1000; i++ {
		go func() {
			client := &http2Client_grpc1460{
				awakenKeepalive: make(chan struct{}),
			}
			go client.keepalive() //G1
			go client.NewStream() //G2
		}()
	}
}
