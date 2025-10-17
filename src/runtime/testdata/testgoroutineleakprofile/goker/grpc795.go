// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Grpc795", Grpc795)
}

type Server_grpc795 struct {
	mu    sync.Mutex
	drain bool
}

func (s *Server_grpc795) GracefulStop() {
	s.mu.Lock()
	if s.drain {
		s.mu.Lock()
		return
	}
	s.drain = true
	s.mu.Unlock()
}
func (s *Server_grpc795) Serve() {
	s.mu.Lock()
	s.mu.Unlock()
}

func NewServer_grpc795() *Server_grpc795 {
	return &Server_grpc795{}
}

type test_grpc795 struct {
	srv *Server_grpc795
}

func (te *test_grpc795) startServer() {
	s := NewServer_grpc795()
	te.srv = s
	go s.Serve()
}

func newTest_grpc795() *test_grpc795 {
	return &test_grpc795{}
}

func testServerGracefulStopIdempotent_grpc795() {
	te := newTest_grpc795()

	te.startServer()

	for i := 0; i < 3; i++ {
		te.srv.GracefulStop()
	}
}

func Grpc795() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		go testServerGracefulStopIdempotent_grpc795()
	}
}
