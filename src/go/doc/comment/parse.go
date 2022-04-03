// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

// A Doc is a parsed Go doc comment.
type Doc struct {
	// Content is the sequence of content blocks in the comment.
	Content []Block

	// Links is the link definitions in the comment.
	Links []*LinkDef
}

// A LinkDef is a single link definition.
type LinkDef struct {
	Text string // the link text
	URL  string // the link URL
	Used bool   // whether the comment uses the definition
}

// A Block is block-level content in a doc comment,
// one of *[Code], *[Heading], *[List], or *[Paragraph].
type Block interface {
	block()
}

// A Heading is a doc comment heading.
type Heading struct {
	Text []Text // the heading text
}

func (*Heading) block() {}

// A List is a numbered or bullet list.
// Lists are always non-empty: len(Items) > 0.
// In a numbered list, every Items[i].Number is a non-empty string.
// In a bullet list, every Items[i].Number is an empty string.
type List struct {
	// Items is the list items.
	Items []*ListItem

	// ForceBlankBefore indicates that the list must be
	// preceded by a blank line when reformatting the comment,
	// overriding the usual conditions. See the BlankBefore method.
	//
	// The comment parser sets ForceBlankBefore for any list
	// that is preceded by a blank line, to make sure
	// the blank line is preserved when printing.
	ForceBlankBefore bool

	// ForceBlankBetween indicates that list items must be
	// separated by blank lines when reformatting the comment,
	// overriding the usual conditions. See the BlankBetween method.
	//
	// The comment parser sets ForceBlankBetween for any list
	// that has a blank line between any two of its items, to make sure
	// the blank lines are preserved when printing.
	ForceBlankBetween bool
}

func (*List) block() {}

// BlankBefore reports whether a reformatting of the comment
// should include a blank line before the list.
// The default rule is the same as for [BlankBetween]:
// if the list item content contains any blank lines
// (meaning at least one item has multiple paragraphs)
// then the list itself must be preceded by a blank line.
// A preceding blank line can be forced by setting [List].ForceBlankBefore.
func (l *List) BlankBefore() bool {
	return l.ForceBlankBefore || l.BlankBetween()
}

// BlankBetween reports whether a reformatting of the comment
// should include a blank line between each pair of list items.
// The default rule is that if the list item content contains any blank lines
// (meaning at least one item has multiple paragraphs)
// then list items must themselves be separated by blank lines.
// Blank line separators can be forced by setting [List].ForceBlankBetween.
func (l *List) BlankBetween() bool {
	if l.ForceBlankBetween {
		return true
	}
	for _, item := range l.Items {
		if len(item.Content) != 1 {
			// Unreachable for parsed comments today,
			// since the only way to get multiple item.Content
			// is multiple paragraphs, which must have been
			// separated by a blank line.
			return true
		}
	}
	return false
}

// A ListItem is a single item in a numbered or bullet list.
type ListItem struct {
	// Number is a decimal string in a numbered list
	// or an empty string in a bullet list.
	Number string // "1", "2", ...; "" for bullet list

	// Content is the list content.
	// Currently, restrictions in the parser and printer
	// require every element of Content to be a *Paragraph.
	Content []Block // Content of this item.
}

// A Paragraph is a paragraph of text.
type Paragraph struct {
	Text []Text
}

func (*Paragraph) block() {}

// A Code is a preformatted code block.
type Code struct {
	// Text is the preformatted text, ending with a newline character.
	// It may be multiple lines, each of which ends with a newline character.
	// It is never empty, nor does it start or end with a blank line.
	Text string
}

func (*Code) block() {}

// A Text is text-level content in a doc comment,
// one of [Plain], [Italic], *[Link], or *[DocLink].
type Text interface {
	text()
}

// A Plain is a string rendered as plain text (not italicized).
type Plain string

func (Plain) text() {}

// An Italic is a string rendered as italicized text.
type Italic string

func (Italic) text() {}

// A Link is a link to a specific URL.
type Link struct {
	Auto bool   // is this an automatic (implicit) link of a literal URL?
	Text []Text // text of link
	URL  string // target URL of link
}

func (*Link) text() {}

// A DocLink is a link to documentation for a Go package or symbol.
type DocLink struct {
	Text []Text // text of link

	// ImportPath, Recv, and Name identify the Go package or symbol
	// that is the link target. The potential combinations of
	// non-empty fields are:
	//  - ImportPath: a link to another package
	//  - ImportPath, Name: a link to a const, func, type, or var in another package
	//  - ImportPath, Recv, Name: a link to a method in another package
	//  - Name: a link to a const, func, type, or var in this package
	//  - Recv, Name: a link to a method in this package
	ImportPath string // import path
	Recv       string // receiver type, without any pointer star, for methods
	Name       string // const, func, type, var, or method name
}

func (*DocLink) text() {}
