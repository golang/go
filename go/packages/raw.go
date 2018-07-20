// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

import (
	"context"
	"fmt"
	"os"
)

// This file contains the structs needed at the seam between the packages
// loader and the underlying build tool

// rawPackage is the serialized form of a package
type rawPackage struct {
	// ID is a unique identifier for a package.
	// This is the same as Package.ID
	ID string
	// Name is the package name as it appears in the package source code.
	// This is the same as Package.name
	Name string `json:",omitempty"`
	// This is the package path as used in the export data.
	// This is used to map entries in the export data back to the package they
	// come from.
	// This is not currently exposed in Package.
	PkgPath string `json:",omitempty"`
	// Imports maps import paths appearing in the package's Go source files
	// to corresponding package identifiers.
	// This is similar to Package.Imports, but maps to the ID rather than the
	// package itself.
	Imports map[string]string `json:",omitempty"`
	// Export is the absolute path to a file containing the export data for the
	// package.
	// This is not currently exposed in Package.
	Export string `json:",omitempty"`
	// GoFiles lists the absolute file paths of the package's Go source files.
	// This is the same as Package.GoFiles
	GoFiles []string `json:",omitempty"`
	// OtherFiles lists the absolute file paths of the package's  non-Go source
	// files.
	// This is the same as Package.OtherFiles
	OtherFiles []string `json:",omitempty"`
	// DepOnly marks a package that is in a list because it was a dependency.
	// It is used to find the roots when constructing a graph from a package list.
	// This is not exposed in Package.
	DepOnly bool `json:",omitempty"`
}

// rawConfig specifies details about what raw package information is needed
// and how the underlying build tool should load package data.
type rawConfig struct {
	Context    context.Context
	Dir        string
	Env        []string
	ExtraFlags []string
	Export     bool
	Tests      bool
	Deps       bool
}

func newRawConfig(cfg *Config) *rawConfig {
	rawCfg := &rawConfig{
		Context:    cfg.Context,
		Dir:        cfg.Dir,
		Env:        cfg.Env,
		ExtraFlags: cfg.Flags,
		Export:     cfg.Mode > LoadImports && cfg.Mode < LoadAllSyntax,
		Tests:      cfg.Tests,
		Deps:       cfg.Mode >= LoadImports,
	}
	if rawCfg.Env == nil {
		rawCfg.Env = os.Environ()
	}
	return rawCfg
}

func (cfg *rawConfig) Flags() []string {
	return append([]string{
		fmt.Sprintf("-test=%t", cfg.Tests),
		fmt.Sprintf("-export=%t", cfg.Export),
		fmt.Sprintf("-deps=%t", cfg.Deps),
	},
		cfg.ExtraFlags...,
	)
}
