// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// PkgDef returns the definition associated with s at package scope.
func (s *Sym) PkgDef() Object { return s.Def }

// SetPkgDef sets the definition associated with s at package scope.
func (s *Sym) SetPkgDef(n Object) { s.Def = n }
