// Inferno utils/6l/pass.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/pass.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

package obj

// Code and data passes.

func Brchain(ctxt *Link, p *Prog) *Prog {

	var i int

	for i = 0; i < 20; i++ {
		if p == nil || int(p.As) != ctxt.Arch.AJMP || p.Pcond == nil {
			return p
		}
		p = p.Pcond
	}

	return nil
}

func brloop(ctxt *Link, p *Prog) *Prog {
	var c int
	var q *Prog

	c = 0
	for q = p; q != nil; q = q.Pcond {
		if int(q.As) != ctxt.Arch.AJMP || q.Pcond == nil {
			break
		}
		c++
		if c >= 5000 {
			return nil
		}
	}

	return q
}

func linkpatch(ctxt *Link, sym *LSym) {
	var c int32
	var name string
	var p *Prog
	var q *Prog

	ctxt.Cursym = sym

	for p = sym.Text; p != nil; p = p.Link {
		if ctxt.Arch.Progedit != nil {
			ctxt.Arch.Progedit(ctxt, p)
		}
		if int(p.To.Type) != ctxt.Arch.D_BRANCH {
			continue
		}
		if p.To.U.Branch != nil {
			// TODO: Remove to.u.branch in favor of p->pcond.
			p.Pcond = p.To.U.Branch

			continue
		}

		if p.To.Sym != nil {
			continue
		}
		c = int32(p.To.Offset)
		for q = sym.Text; q != nil; {
			if int64(c) == q.Pc {
				break
			}
			if q.Forwd != nil && int64(c) >= q.Forwd.Pc {
				q = q.Forwd
			} else {

				q = q.Link
			}
		}

		if q == nil {
			name = "<nil>"
			if p.To.Sym != nil {
				name = p.To.Sym.Name
			}
			ctxt.Diag("branch out of range (%#x)\n%v [%s]", uint32(c), p, name)
			p.To.Type = int16(ctxt.Arch.D_NONE)
		}

		p.To.U.Branch = q
		p.Pcond = q
	}

	for p = sym.Text; p != nil; p = p.Link {
		p.Mark = 0 /* initialization for follow */
		if p.Pcond != nil {
			p.Pcond = brloop(ctxt, p.Pcond)
			if p.Pcond != nil {
				if int(p.To.Type) == ctxt.Arch.D_BRANCH {
					p.To.Offset = p.Pcond.Pc
				}
			}
		}
	}
}
