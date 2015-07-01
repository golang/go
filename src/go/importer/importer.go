// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package importer provides access to export data importers.
package importer

import (
	"go/internal/gccgoimporter"
	"go/internal/gcimporter"
	"go/types"
	"io"
	"runtime"
)

// A Lookup function returns a reader to access package data for
// a given import path, or an error if no matching package is found.
type Lookup func(path string) (io.ReadCloser, error)

// For returns an Importer for the given compiler and lookup interface,
// or nil. Supported compilers are "gc", and "gccgo". If lookup is nil,
// the default package lookup mechanism for the given compiler is used.
func For(compiler string, lookup Lookup) types.Importer {
	switch compiler {
	case "gc":
		if lookup == nil {
			return make(gcimports)
		}
		panic("gc importer for custom import path lookup not yet implemented")
	case "gccgo":
		if lookup == nil {
			var inst gccgoimporter.GccgoInstallation
			if err := inst.InitFromDriver("gccgo"); err != nil {
				return nil
			}
			return &gccgoimports{
				packages: make(map[string]*types.Package),
				importer: inst.GetImporter(nil, nil),
			}
		}
		panic("gccgo importer for custom import path lookup not yet implemented")
	}
	// compiler not supported
	return nil
}

// Default returns an Importer for the compiler that built the running binary.
func Default() types.Importer {
	return For(runtime.Compiler, nil)
}

// gc support

type gcimports map[string]*types.Package

func (m gcimports) Import(path string) (*types.Package, error) {
	return gcimporter.Import(m, path)
}

// gccgo support

type gccgoimports struct {
	packages map[string]*types.Package
	importer gccgoimporter.Importer
}

func (m *gccgoimports) Import(path string) (*types.Package, error) {
	return m.importer(m.packages, path)
}
