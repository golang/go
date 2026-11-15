// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosDotMap() *Name {
	if p.mode&LogosMode == 0 || p.tok != _Map {
		return nil
	}
	name := NewName(p.pos(), "map")
	p.next()
	return name
}
