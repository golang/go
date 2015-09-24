// Copyright © 2015 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package riscv

import (
	"log"

	"cmd/internal/obj"
)

// progedit is called individually for each Prog.
// TODO(myenik)
func progedit(ctxt *obj.Link, p *obj.Prog) {
	log.Printf("progedit: ctxt: %+v p: %#v p: %s", ctxt, p, p)
}

// TODO(myenik)
func follow(ctxt *obj.Link, s *obj.LSym) {
	log.Printf("follow: ctxt: %+v", ctxt)

	for ; s != nil; s = s.Next {
		log.Printf("s: %+v", s)
	}
}

// TODO(myenik)
func preprocess(ctxt *obj.Link, cursym *obj.LSym) {
	log.Printf("preprocess: ctxt: %+v", ctxt)

	for ; cursym != nil; cursym = cursym.Next {
		log.Printf("cursym: %+v", cursym)
	}
}

// TODO(myenik)
func assemble(ctxt *obj.Link, cursym *obj.LSym) {
	log.Printf("assemble: ctxt: %+v", ctxt)

	for ; cursym != nil; cursym = cursym.Next {
		log.Printf("cursym: %+v", cursym)
	}
}
