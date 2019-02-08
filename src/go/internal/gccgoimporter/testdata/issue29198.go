// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"
	"errors"
)

type A struct {
	x int
}

func (a *A) AMethod(y int) *Server {
	return nil
}

// FooServer is a server that provides Foo services
type FooServer Server

func (f *FooServer) WriteEvents(ctx context.Context, x int) error {
	return errors.New("hey!")
}

type Server struct {
	FooServer *FooServer
	user      string
	ctx       context.Context
}

func New(sctx context.Context, u string) (*Server, error) {
	s := &Server{user: u, ctx: sctx}
	s.FooServer = (*FooServer)(s)
	return s, nil
}
