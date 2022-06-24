// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

var N n

type n struct{}

func (r n) M1() int  { return a.G1 }
func (r n) M2() int  { return a.G2 }
func (r n) M3() int  { return a.G3 }
func (r n) M4() int  { return a.G4 }
func (r n) M5() int  { return a.G5 }
func (r n) M6() int  { return a.G6 }
func (r n) M7() int  { return a.G7 }
func (r n) M8() int  { return a.G8 }
func (r n) M9() int  { return a.G9 }
func (r n) M10() int { return a.G10 }
