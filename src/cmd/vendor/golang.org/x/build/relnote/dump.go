// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package relnote

import (
	"fmt"
	"strings"

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
	typeName := strings.TrimPrefix(fmt.Sprintf("%T", b), "*markdown.")
	dprintf(depth, "%s\n", typeName)
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
		switch in := in.(type) {
		case *md.Plain:
			dprintf(depth, "Plain(%q)\n", in.Text)
		case *md.Code:
			dprintf(depth, "Code(%q)\n", in.Text)
		case *md.Link:
			dprintf(depth, "Link:\n")
			dumpInlines(in.Inner, depth+1)
			dprintf(depth+1, "URL: %q\n", in.URL)
		case *md.Strong:
			dprintf(depth, "Strong(%q):\n", in.Marker)
			dumpInlines(in.Inner, depth+1)
		case *md.Emph:
			dprintf(depth, "Emph(%q):\n", in.Marker)
			dumpInlines(in.Inner, depth+1)
		case *md.Del:
			dprintf(depth, "Del(%q):\n", in.Marker)
			dumpInlines(in.Inner, depth+1)
		default:
			fmt.Printf("%*s%#v\n", depth*4, "", in)
		}
	}
}

func dprintf(depth int, format string, args ...any) {
	fmt.Printf("%*s%s", depth*4, "", fmt.Sprintf(format, args...))
}
