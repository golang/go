// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {a int}
type P *T
type P1 *T

func (p P) val() int { return 1 }  // ERROR "receiver"
func (p *P1) val() int { return 1 }  // ERROR "receiver"
