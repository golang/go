// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go || echo BUG: compilation should succeed

package main

type S struct {
};

func (p *S) M1a() ;

func (p *S) M2a() {
  p.M1a();
}

func (p *S) M1a() {}  // this works


func (p *S) M1b() int;

func (p *S) M2b() {
  p.M1b();
}

func (p *S) M1b() int {}  // BUG this doesn't
