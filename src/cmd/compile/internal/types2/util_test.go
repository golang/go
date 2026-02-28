// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exports various functionality of util.go
// so that it can be used in (package-external) tests.

package types2

import (
	"cmd/compile/internal/syntax"
)

func CmpPos(p, q syntax.Pos) int { return cmpPos(p, q) }

func ScopeComment(s *Scope) string         { return s.comment }
func ObjectScopePos(obj Object) syntax.Pos { return obj.scopePos() }
