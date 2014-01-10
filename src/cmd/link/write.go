// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing of executable and (for hostlink mode) object files.

package main

import "io"

func (p *Prog) write(w io.Writer) {
	p.Entry = p.Syms[startSymID].Addr
	p.formatter.write(w, p)
}
