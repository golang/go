// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19 || openbsd || js || wasip1 || solaris || android || plan9 || 386

package counter

import (
	"flag"
)

func Add(string, int64)                         {}
func Inc(string)                                {}
func Open()                                     {}
func CountFlags(prefix string, fs flag.FlagSet) {}

type Counter struct{ name string }

func New(name string) *Counter  { return &Counter{name} }
func (c *Counter) Add(n int64)  {}
func (c *Counter) Inc()         {}
func (c *Counter) Name() string { return c.name }

type StackCounter struct{ name string }

func NewStack(name string, _ int) *StackCounter { return &StackCounter{name} }
func (c *StackCounter) Counters() []*Counter    { return nil }
func (c *StackCounter) Inc()                    {}
func (c *StackCounter) Names() []string         { return nil }
