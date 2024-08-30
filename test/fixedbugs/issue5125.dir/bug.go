// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug

type Node interface {
	Eval(s *Scene)
}

type plug struct {
	node Node
}

type Scene struct {
	changed map[plug]bool
}
