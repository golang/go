// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Conn interface{ Hello() }
type Pool[T Conn] struct{}

func (p *Pool[T]) Hello(conn T) { conn.Hello() }

type PoolConn struct{ Conn }
type CustomConn struct{}

func (p CustomConn) Hello() { called = true }

func NewPool[T Conn]() *Pool[T] { return &Pool[T]{} }

var called bool

func main() {
	NewPool[*PoolConn]().Hello(&PoolConn{Conn: CustomConn{}})
	if !called {
		panic("the embedded interface method Hello was not called")
	}
}
