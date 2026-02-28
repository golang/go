// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package list

import (
	"go/build"
)

type Context struct {
	GOARCH        string   `json:",omitempty"` // target architecture
	GOOS          string   `json:",omitempty"` // target operating system
	GOROOT        string   `json:",omitempty"` // Go root
	GOPATH        string   `json:",omitempty"` // Go path
	CgoEnabled    bool     `json:",omitempty"` // whether cgo can be used
	UseAllFiles   bool     `json:",omitempty"` // use files regardless of //go:build lines, file names
	Compiler      string   `json:",omitempty"` // compiler to assume when computing target paths
	BuildTags     []string `json:",omitempty"` // build constraints to match in +build lines
	ToolTags      []string `json:",omitempty"` // toolchain-specific build constraints
	ReleaseTags   []string `json:",omitempty"` // releases the current release is compatible with
	InstallSuffix string   `json:",omitempty"` // suffix to use in the name of the install dir
}

func newContext(c *build.Context) *Context {
	return &Context{
		GOARCH:        c.GOARCH,
		GOOS:          c.GOOS,
		GOROOT:        c.GOROOT,
		GOPATH:        c.GOPATH,
		CgoEnabled:    c.CgoEnabled,
		UseAllFiles:   c.UseAllFiles,
		Compiler:      c.Compiler,
		BuildTags:     c.BuildTags,
		ToolTags:      c.ToolTags,
		ReleaseTags:   c.ReleaseTags,
		InstallSuffix: c.InstallSuffix,
	}
}
