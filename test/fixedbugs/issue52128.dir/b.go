// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import (
	"./a"
)

type S struct{}

func (s *S) M1() a.I {
	return a.NewWithF(s.M2)
}

func (s *S) M2() {}
