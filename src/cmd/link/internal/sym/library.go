// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import "cmd/internal/goobj2"

type Library struct {
	Objref        string
	Srcref        string
	File          string
	Pkg           string
	Shlib         string
	Hash          string
	ImportStrings []string
	Imports       []*Library
	Textp         []*Symbol // text symbols defined in this library
	DupTextSyms   []*Symbol // dupok text symbols defined in this library
	Main          bool
	Safe          bool
	Units         []*CompilationUnit

	Readers []struct { // TODO: probably move this to Loader
		Reader  *goobj2.Reader
		Version int
	}
}

func (l Library) String() string {
	return l.Pkg
}
