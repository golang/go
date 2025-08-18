// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reorder

type Element struct {
	A     string
	B     string
	C     string
	D     string
	E     string
	Text  []string
	List  []string
	Child Elements
	F     string
	G     bool
	H     bool
	I     string
}

type Elements []Element

func DoesNotCompile(ve Elements) Elements {
	aa := Elements{}
	bb := Elements{}
	cc := Elements{}
	dd := Elements{}
	ee := Elements{}
	ff := Elements{}
	gg := Elements{}
	hh := Elements{}
	ii := Elements{}

	if len(ve) != 1 {
		return ve
	}
	for _, e := range ve[0].Child {
		if len(e.Text) == 1 && (e.Text[0] == "xx") {
			ee = append(ee, e)
		} else if len(e.Text) == 1 && e.Text[0] == "yy" {
			for _, c := range e.Child {
				if len(c.Text) == 1 && c.Text[0] == "zz" {
					ii = append(ii, c)
				} else {
					hh = append(hh, c)
				}
			}
			ii = append(ii, hh...)
			e.Child = ii
			gg = append(gg, e)
		} else if len(e.Text) == 1 && e.Text[0] == "tt" {
			for _, entry := range e.Child {
				for _, c := range entry.Child {
					if len(c.Text) == 1 && c.Text[0] == "ee" {
						cc = append(cc, c)
					} else {
						dd = append(dd, c)
					}
				}
				cc = append(cc, dd...)
				entry.Child = cc
				bb = append(bb, entry)
				cc, dd = Elements{}, Elements{}
			}
			e.Child = bb
			aa = append(aa, e)
		} else {
			ff = append(ff, e)
		}
	}
	return ve
}
