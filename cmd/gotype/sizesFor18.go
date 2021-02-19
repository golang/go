// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.9
// +build !go1.9

// This file contains a copy of the implementation of types.SizesFor
// since this function is not available in go/types before Go 1.9.

package main

import "go/types"

const defaultCompiler = "gc"

var gcArchSizes = map[string]*types.StdSizes{
	"386":      {4, 4},
	"arm":      {4, 4},
	"arm64":    {8, 8},
	"amd64":    {8, 8},
	"amd64p32": {4, 8},
	"mips":     {4, 4},
	"mipsle":   {4, 4},
	"mips64":   {8, 8},
	"mips64le": {8, 8},
	"ppc64":    {8, 8},
	"ppc64le":  {8, 8},
	"s390x":    {8, 8},
}

func SizesFor(compiler, arch string) types.Sizes {
	if compiler != "gc" {
		return nil
	}
	s, ok := gcArchSizes[arch]
	if !ok {
		return nil
	}
	return s
}
