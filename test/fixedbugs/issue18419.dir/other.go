// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package other

type Exported struct {
	Member int
}

func (e *Exported) member() int { return 1 }
