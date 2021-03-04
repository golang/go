// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// This file exposes additional API methods related to type parameters, for use
// in the go2go rewriter.

type TypeParam = _TypeParam

func (t *Named) TArgs() []Type        { return t._TArgs() }
func (t *Named) SetTArgs(args []Type) { t._SetTArgs(args) }
func (t *Named) TParams() []*TypeName { return t._TParams() }

func (t *Interface) HasTypeList() bool { return t._HasTypeList() }

func (s *Signature) TParams() []*TypeName           { return s._TParams() }
func (s *Signature) SetTParams(tparams []*TypeName) { s._SetTParams(tparams) }

func AsPointer(t Type) *Pointer {
	return asPointer(t)
}

func AsStruct(t Type) *Struct {
	return asStruct(t)
}
