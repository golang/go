// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/l.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package sym

type Symbols struct {
	symbolBatch []Symbol

	// Symbol lookup based on name and indexed by version.
	hash []map[string]*Symbol

	Allsym []*Symbol
}

func NewSymbols() *Symbols {
	return &Symbols{
		hash: []map[string]*Symbol{
			// preallocate about 2mb for hash of
			// non static symbols
			make(map[string]*Symbol, 100000),
		},
		Allsym: make([]*Symbol, 0, 100000),
	}
}

func (syms *Symbols) Newsym(name string, v int) *Symbol {
	batch := syms.symbolBatch
	if len(batch) == 0 {
		batch = make([]Symbol, 1000)
	}
	s := &batch[0]
	syms.symbolBatch = batch[1:]

	s.Dynid = -1
	s.Plt = -1
	s.Got = -1
	s.Name = name
	s.Version = int16(v)
	syms.Allsym = append(syms.Allsym, s)

	return s
}

// Look up the symbol with the given name and version, creating the
// symbol if it is not found.
func (syms *Symbols) Lookup(name string, v int) *Symbol {
	m := syms.hash[v]
	s := m[name]
	if s != nil {
		return s
	}
	s = syms.Newsym(name, v)
	s.Extname = s.Name
	m[name] = s
	return s
}

// Look up the symbol with the given name and version, returning nil
// if it is not found.
func (syms *Symbols) ROLookup(name string, v int) *Symbol {
	return syms.hash[v][name]
}

// Allocate a new version (i.e. symbol namespace).
func (syms *Symbols) IncVersion() int {
	syms.hash = append(syms.hash, make(map[string]*Symbol))
	return len(syms.hash) - 1
}

// Rename renames a symbol.
func (syms *Symbols) Rename(old, new string, v int) {
	s := syms.hash[v][old]
	s.Name = new
	if s.Extname == old {
		s.Extname = new
	}
	delete(syms.hash[v], old)

	dup := syms.hash[v][new]
	if dup == nil {
		syms.hash[v][new] = s
	} else {
		if s.Type == 0 {
			*s = *dup
		} else if dup.Type == 0 {
			*dup = *s
			syms.hash[v][new] = s
		}
	}
}
