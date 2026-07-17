// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package target

type Base struct{}

func (Base) M() {}

type Target struct {
	Base
}

var P *Target
