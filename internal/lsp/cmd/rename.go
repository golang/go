// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
	errors "golang.org/x/xerrors"
)

// rename implements the rename verb for gopls.
type rename struct {
	Diff  bool `flag:"d" help:"display diffs instead of rewriting files"`
	Write bool `flag:"w" help:"write result to (source) file instead of stdout"`

	app *Application
}

func (r *rename) Name() string      { return "rename" }
func (r *rename) Usage() string     { return "<position>" }
func (r *rename) ShortHelp() string { return "rename selected identifier" }
func (r *rename) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

  $ # 1-based location (:line:column or :#position) of the thing to change
  $ gopls rename helper/helper.go:8:6
  $ gopls rename helper/helper.go:#53

	gopls rename flags are:
`)
	f.PrintDefaults()
}

// Run renames the specified identifier and either;
// - if -w is specified, updates the file(s) in place;
// - if -d is specified, prints out unified diffs of the changes; or
// - otherwise, prints the new versions to stdout.
func (r *rename) Run(ctx context.Context, args ...string) error {
	if len(args) != 2 {
		return tool.CommandLineErrorf("definition expects 2 arguments (position, new name)")
	}
	conn, err := r.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	from := span.Parse(args[0])
	file := conn.AddFile(ctx, from.URI())
	if file.err != nil {
		return file.err
	}

	loc, err := file.mapper.Location(from)
	if err != nil {
		return err
	}

	p := protocol.RenameParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
		NewName:      args[1],
	}
	we, err := conn.Rename(ctx, &p)
	if err != nil {
		return err
	}

	// Make output order predictable
	var keys []string
	for u, _ := range *we.Changes {
		keys = append(keys, u)
	}
	sort.Strings(keys)
	changeCount := len(keys)

	for _, u := range keys {
		edits := (*we.Changes)[u]
		uri := span.NewURI(u)
		cmdFile := conn.AddFile(ctx, uri)
		filename := cmdFile.uri.Filename()

		// convert LSP-style edits to []diff.TextEdit cuz Spans are handy
		renameEdits, err := source.FromProtocolEdits(cmdFile.mapper, edits)
		if err != nil {
			return errors.Errorf("%v: %v", edits, err)
		}

		newContent := diff.ApplyEdits(string(cmdFile.mapper.Content), renameEdits)

		switch {
		case r.Write:
			fmt.Fprintln(os.Stderr, filename)
			err := os.Rename(filename, filename+".orig")
			if err != nil {
				return errors.Errorf("%v: %v", edits, err)
			}
			ioutil.WriteFile(filename, []byte(newContent), 0644)
		case r.Diff:
			// myersEdits := diff.ComputeEdits(cmdFile.uri, string(cmdFile.mapper.Content), string(newContent))
			myersEdits := toMyersTextEdits(renameEdits, cmdFile.mapper)
			diffs := diff.ToUnified(filename+".orig", filename, string(cmdFile.mapper.Content), myersEdits)
			fmt.Print(diffs)
		default:
			fmt.Printf("%s:\n", filepath.Base(filename))
			fmt.Print(string(newContent))
			if changeCount > 1 { // if this wasn't last change, print newline
				fmt.Println()
			}
			changeCount -= 1
		}
	}
	return nil
}

type editPair [2]diff.TextEdit // container for a del/ins TextEdit pair

// toMyersTextEdits converts the "word-oriented" textEdits returned by
// source.Rename into the "line-oriented" textEdits that
// diff.ToUnified() (aka myers.toUnified()) expects.
func toMyersTextEdits(edits []diff.TextEdit, mapper *protocol.ColumnMapper) []diff.TextEdit {
	var myersEdits []diff.TextEdit

	if len(edits) == 0 {
		return myersEdits
	}

	contentByLine := strings.Split(string(mapper.Content), "\n")

	// gather all of the edits on a line, create an editPair from them,
	// and append it to the list of pairs
	var pairs []editPair
	var pending []diff.TextEdit
	currentLine := edits[0].Span.Start().Line()
	for i := 0; i < len(edits); i++ {
		if edits[i].Span.Start().Line() != currentLine {
			pairs = append(pairs, toEditPair(pending, contentByLine[currentLine-1]))
			currentLine = edits[i].Span.Start().Line()
			pending = pending[:0] // clear it, leaking not a problem...
		}
		pending = append(pending, edits[i])
	}
	pairs = append(pairs, toEditPair(pending, contentByLine[currentLine-1]))

	// reorder contiguous del/ins pairs into blocks of del and ins
	myersEdits = reorderEdits(pairs)
	return myersEdits
}

// toEditPair takes one or more "word" diff.TextEdit(s) that occur
// on a single line and creates a single equivalent
// delete-line/insert-line pair of diff.TextEdit.
func toEditPair(edits []diff.TextEdit, before string) editPair {
	// interleave retained bits of old line with new text from edits
	p := 0 // position in old line
	after := ""
	for i := 0; i < len(edits); i++ {
		after += before[p:edits[i].Span.Start().Column()-1] + edits[i].NewText
		p = edits[i].Span.End().Column() - 1
	}
	after += before[p:] + "\n"

	// seems we can get away w/out providing offsets
	u := edits[0].Span.URI()
	l := edits[0].Span.Start().Line()
	newEdits := editPair{
		diff.TextEdit{Span: span.New(u, span.NewPoint(l, 1, -1), span.NewPoint(l+1, 1, -1))},
		diff.TextEdit{Span: span.New(u, span.NewPoint(l+1, 1, -1), span.NewPoint(l+1, 1, -1)), NewText: after},
	}
	return newEdits
}

// reorderEdits reorders blocks of delete/insert pairs so that all of
// the deletes come first, resetting the spans for the insert records
// to keep them "sorted".  It assumes that each entry is a "del/ins"
// pair.
func reorderEdits(e []editPair) []diff.TextEdit {
	var r []diff.TextEdit // reordered edits
	var p []diff.TextEdit // pending insert edits, waiting for end of dels

	r = append(r, e[0][0])
	p = append(p, e[0][1])

	for i := 1; i < len(e); i++ {
		if e[i][0].Span.Start().Line() != r[len(r)-1].Span.Start().Line()+1 {
			unpend(&r, &p)
			p = p[:0] // clear it, leaking not a problem...
		}
		r = append(r, e[i][0])
		p = append(p, e[i][1])
	}
	unpend(&r, &p)

	return r
}

// unpend sets the spans of the pending TextEdits to point to the last
// line in the associated block of deletes then appends them to r.
func unpend(r, p *[]diff.TextEdit) {
	for j := 0; j < len(*p); j++ {
		prev := (*r)[len(*r)-1]
		(*p)[j].Span = span.New(prev.Span.URI(), prev.Span.End(), prev.Span.End())
	}
	*r = append(*r, (*p)...)
}
