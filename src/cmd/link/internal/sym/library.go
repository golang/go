// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import "cmd/internal/goobj"

type Library struct {
	Objref      string
	Srcref      string
	File        string
	Pkg         string
	Shlib       string
	Fingerprint goobj.FingerprintType
	Autolib     []goobj.ImportedPkg
	Imports     []*Library
	Main        bool
	Units       []*CompilationUnit

	Textp       []LoaderSym // text syms defined in this library
	DupTextSyms []LoaderSym // dupok text syms defined in this library
}

func (l Library) String() string {
	return l.Pkg
}
