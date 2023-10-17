// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package state

import "./foo"

func Public() {
	var s Settings
	s.op()
}

type State struct{}

func (s *State) x(*Settings) {}

type Settings struct{}

func (c *Settings) x() {
	run([]foo.Op{{}})
}

func run([]foo.Op) {}

func (s *Settings) op() foo.Op {
	return foo.Op{}
}
