// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// machine-independent optimization

//go:generate go run rulegen/rulegen.go rulegen/generic.rules genericBlockRules genericValueRules generic.go

func opt(f *Func) {
	applyRewrite(f, genericBlockRules, genericValueRules)
}
