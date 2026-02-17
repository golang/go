// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: grpc-go
 * Issue or PR   : https://github.com/grpc/grpc-go/pull/862
 * Buggy version: d8f4ebe77f6b7b6403d7f98626de8a534f9b93a7
 * fix commit-id: dd5645bebff44f6b88780bb949022a09eadd7dae
 * Flaky: 100/100
 */
package main

import (
	"context"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

func init() {
	register("Grpc862", Grpc862)
}

type ClientConn_grpc862 struct {
	ctx    context.Context
	cancel context.CancelFunc
	conns  []*addrConn_grpc862
}

func (cc *ClientConn_grpc862) Close() {
	cc.cancel()
	conns := cc.conns
	cc.conns = nil
	for _, ac := range conns {
		ac.tearDown()
	}
}

func (cc *ClientConn_grpc862) resetAddrConn() {
	ac := &addrConn_grpc862{
		cc: cc,
	}
	cc.conns = append(cc.conns, ac)
	ac.ctx, ac.cancel = context.WithCancel(cc.ctx)
	ac.resetTransport()
}

type addrConn_grpc862 struct {
	cc     *ClientConn_grpc862
	ctx    context.Context
	cancel context.CancelFunc
}

func (ac *addrConn_grpc862) resetTransport() {
	for retries := 1; ; retries++ {
		_ = 2 * time.Nanosecond * time.Duration(retries)
		timeout := 10 * time.Nanosecond
		_, cancel := context.WithTimeout(ac.ctx, timeout)
		_ = time.Now()
		cancel()
		<-ac.ctx.Done()
		return
	}
}

func (ac *addrConn_grpc862) tearDown() {
	ac.cancel()
}

func DialContext_grpc862(ctx context.Context) (conn *ClientConn_grpc862) {
	cc := &ClientConn_grpc862{}
	cc.ctx, cc.cancel = context.WithCancel(context.Background())
	defer func() {
		select {
		case <-ctx.Done():
			if conn != nil {
				conn.Close()
			}
			conn = nil
		default:
		}
	}()
	go func() { // G2
		cc.resetAddrConn()
	}()
	return conn
}

func Grpc862() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		ctx, cancel := context.WithCancel(context.Background())
		go DialContext_grpc862(ctx) // G1
		go cancel()                 // helper goroutine
	}()
}
