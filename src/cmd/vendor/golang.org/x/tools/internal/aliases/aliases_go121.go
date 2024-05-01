// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.22
// +build !go1.22

package aliases

import (
	"go/types"
)

// Alias is a placeholder for a go/types.Alias for <=1.21.
// It will never be created by go/types.
type Alias struct{}

func (*Alias) String() string { panic("unreachable") }

func (*Alias) Underlying() types.Type { panic("unreachable") }

func (*Alias) Obj() *types.TypeName { panic("unreachable") }

// Unalias returns the type t for go <=1.21.
func Unalias(t types.Type) types.Type { return t }

// Always false for go <=1.21. Ignores GODEBUG.
func enabled() bool { return false }

func newAlias(name *types.TypeName, rhs types.Type) *Alias { panic("unreachable") }
