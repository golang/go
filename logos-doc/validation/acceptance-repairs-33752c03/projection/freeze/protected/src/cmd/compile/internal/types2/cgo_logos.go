// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"bytes"
	"path/filepath"
	"strings"
)

// LogosEnableCgo resolves original C selectors against the declarations
// from cmd/cgo. The caller must provide _cgo_gotypes.go with the original
// sources. It must not set FakeImportC, which would skip type checking.
func LogosEnableCgo(conf *Config) {
	conf.go115UsesCgo = true
}

// LogosCgoTypeString spells generated C type objects in their original
// source form. Other objects, including identically named fields, retain
// the ordinary TypeString spelling. pkg is the checked source package.
func LogosCgoTypeString(typ Type, qf Qualifier, pkg *Package) string {
	assert(pkg != nil)
	var buf bytes.Buffer
	w := newTypeWriter(&buf, qf)
	w.logosCgoPkg = pkg
	w.typ(typ)
	return buf.String()
}

func (w *typeWriter) logosCgoTypeName(obj *TypeName) bool {
	if w.logosCgoPkg == nil || obj.pkg != w.logosCgoPkg || !obj.pos.IsKnown() {
		return false
	}
	name, ok := strings.CutPrefix(obj.name, "_Ctype_")
	if !ok || filepath.Base(obj.pos.FileBase().Filename()) != "_cgo_gotypes.go" {
		return false
	}
	w.string("C.")
	w.string(name)
	return true
}
