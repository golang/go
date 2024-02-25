// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package relnote

import (
	"fmt"

	md "rsc.io/markdown"
)

// DumpMarkdown writes the internal structure of a markdown
// document to standard output.
// It is intended for debugging.
func DumpMarkdown(d *md.Document) {
	dumpBlocks(d.Blocks, 0)
}

func dumpBlocks(bs []md.Block, depth int) {
	for _, b := range bs {
		dumpBlock(b, depth)
	}
}

func dumpBlock(b md.Block, depth int) {
	fmt.Printf("%*s%T\n", depth*4, "", b)
	switch b := b.(type) {
	case *md.Paragraph:
		dumpInlines(b.Text.Inline, depth+1)
	case *md.Heading:
		dumpInlines(b.Text.Inline, depth+1)
	case *md.List:
		dumpBlocks(b.Items, depth+1)
	case *md.Item:
		dumpBlocks(b.Blocks, depth+1)
	default:
		// TODO(jba): additional cases as needed.
	}
}

func dumpInlines(ins []md.Inline, depth int) {
	for _, in := range ins {
		fmt.Printf("%*s%#v\n", depth*4, "", in)
	}
}
