// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(imagetiledTests, imagetiled)
}

var imagetiledTests = []testCase{
	{
		Name: "imagetiled.0",
		In: `package main

import (
	"foo"
	"image"
)

var (
	_ foo.Tiled
	_ image.RGBA
	_ image.Tiled
)
`,
		Out: `package main

import (
	"foo"
	"image"
)

var (
	_ foo.Tiled
	_ image.RGBA
	_ image.Repeated
)
`,
	},
}
