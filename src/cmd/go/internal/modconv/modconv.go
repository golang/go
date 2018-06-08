// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import "cmd/go/internal/module"

var Converters = map[string]func(string, []byte) ([]module.Version, error){
	"GLOCKFILE":          ParseGLOCKFILE,
	"Godeps/Godeps.json": ParseGodepsJSON,
	"Gopkg.lock":         ParseGopkgLock,
	"dependencies.tsv":   ParseDependenciesTSV,
	"glide.lock":         ParseGlideLock,
	"vendor.conf":        ParseVendorConf,
	"vendor.yml":         ParseVendorYML,
	"vendor/manifest":    ParseVendorManifest,
	"vendor/vendor.json": ParseVendorJSON,
}

// Prefix is a line we write at the top of auto-converted go.mod files
// for dependencies before caching them.
// In case of bugs in the converter, if we bump this version number,
// then all the cached copies will be ignored.
const Prefix = "//vgo 0.0.4\n"
