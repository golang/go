// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams && go1.18
// +build typeparams,go1.18

package gcimporter

import "go/types"

const iexportVersion = iexportVersionGenerics

// additionalPredeclared returns additional predeclared types in go.1.18.
func additionalPredeclared() []types.Type {
	return []types.Type{
		// comparable
		types.Universe.Lookup("comparable").Type(),
	}
}
