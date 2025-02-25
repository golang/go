// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package relnote supports working with release notes.
//
// Its main feature is the ability to merge Markdown fragments into a single
// document. (See [Merge].)
//
// This package has minimal imports, so that it can be vendored into the
// main go repo.
package relnote

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"regexp"
	"slices"
	"strconv"
	"strings"

	md "rsc.io/markdown"
)

// NewParser returns a properly configured Markdown parser.
func NewParser() *md.Parser {
	var p md.Parser
	p.HeadingIDs = true
	return &p
}

// CheckFragment reports problems in a release-note fragment.
func CheckFragment(data string) error {
	doc := NewParser().Parse(data)
	// Check that the content of the document contains either a TODO or at least one sentence.
	txt := ""
	if len(doc.Blocks) > 0 {
		txt = text(doc)
	}
	if !strings.Contains(txt, "TODO") && !strings.ContainsAny(txt, ".?!") {
		return errors.New("File must contain a complete sentence or a TODO.")
	}
	return nil
}

// text returns all the text in a block, without any formatting.
func text(b md.Block) string {
	switch b := b.(type) {
	case *md.Document:
		return blocksText(b.Blocks)
	case *md.Heading:
		return text(b.Text)
	case *md.Text:
		return inlineText(b.Inline)
	case *md.CodeBlock:
		return strings.Join(b.Text, "\n")
	case *md.HTMLBlock:
		return strings.Join(b.Text, "\n")
	case *md.List:
		return blocksText(b.Items)
	case *md.Item:
		return blocksText(b.Blocks)
	case *md.Empty:
		return ""
	case *md.Paragraph:
		return text(b.Text)
	case *md.Quote:
		return blocksText(b.Blocks)
	case *md.ThematicBreak:
		return "---"
	default:
		panic(fmt.Sprintf("unknown block type %T", b))
	}
}

// blocksText returns all the text in a slice of block nodes.
func blocksText(bs []md.Block) string {
	var d strings.Builder
	for _, b := range bs {
		io.WriteString(&d, text(b))
		fmt.Fprintln(&d)
	}
	return d.String()
}

// inlineText returns all the next in a slice of inline nodes.
func inlineText(ins []md.Inline) string {
	var buf bytes.Buffer
	for _, in := range ins {
		in.PrintText(&buf)
	}
	return buf.String()
}

// Merge combines the markdown documents (files ending in ".md") in the tree rooted
// at fs into a single document.
// The blocks of the documents are concatenated in lexicographic order by filename.
// Heading with no content are removed.
// The link keys must be unique, and are combined into a single map.
//
// Files in the "minor changes" directory (the unique directory matching the glob
// "*stdlib/*minor") are named after the package to which they refer, and will have
// the package heading inserted automatically and links to other standard library
// symbols expanded automatically. For example, if a file *stdlib/minor/bytes/f.md
// contains the text
//
//	[Reader] implements [io.Reader].
//
// then that will become
//
//	[Reader](/pkg/bytes#Reader) implements [io.Reader](/pkg/io#Reader).
func Merge(fsys fs.FS) (*md.Document, error) {
	filenames, err := sortedMarkdownFilenames(fsys)
	if err != nil {
		return nil, err
	}
	doc := &md.Document{Links: map[string]*md.Link{}}
	var prevPkg string // previous stdlib package, if any
	for _, filename := range filenames {
		newdoc, err := parseMarkdownFile(fsys, filename)
		if err != nil {
			return nil, err
		}
		if len(newdoc.Blocks) == 0 {
			continue
		}
		pkg := stdlibPackage(filename)
		// Autolink Go symbols.
		addSymbolLinks(newdoc, pkg)
		if len(doc.Blocks) > 0 {
			// If this is the first file of a new stdlib package under the "Minor changes
			// to the library" section, insert a heading for the package.
			if pkg != "" && pkg != prevPkg {
				h := stdlibPackageHeading(pkg, lastBlock(doc).Pos().EndLine)
				doc.Blocks = append(doc.Blocks, h)
			}
			prevPkg = pkg
			// Put a blank line between the current and new blocks, so that the end
			// of a file acts as a blank line.
			lastLine := lastBlock(doc).Pos().EndLine
			delta := lastLine + 2 - newdoc.Blocks[0].Pos().StartLine
			for _, b := range newdoc.Blocks {
				addLines(b, delta)
			}
		}
		// Append non-empty blocks to the result document.
		for _, b := range newdoc.Blocks {
			if _, ok := b.(*md.Empty); !ok {
				doc.Blocks = append(doc.Blocks, b)
			}
		}
		// Merge link references.
		for key, link := range newdoc.Links {
			if doc.Links[key] != nil {
				return nil, fmt.Errorf("duplicate link reference %q; second in %s", key, filename)
			}
			doc.Links[key] = link
		}
	}
	// Remove headings with empty contents.
	doc.Blocks = removeEmptySections(doc.Blocks)
	if len(doc.Blocks) > 0 && len(doc.Links) > 0 {
		// Add a blank line to separate the links.
		lastPos := lastBlock(doc).Pos()
		lastPos.StartLine += 2
		lastPos.EndLine += 2
		doc.Blocks = append(doc.Blocks, &md.Empty{Position: lastPos})
	}
	return doc, nil
}

// stdlibPackage returns the standard library package for the given filename.
// If the filename does not represent a package, it returns the empty string.
// A filename represents package P if it is in a directory matching the glob
// "*stdlib/*minor/P".
func stdlibPackage(filename string) string {
	dir, rest, _ := strings.Cut(filename, "/")
	if !strings.HasSuffix(dir, "stdlib") {
		return ""
	}
	dir, rest, _ = strings.Cut(rest, "/")
	if !strings.HasSuffix(dir, "minor") {
		return ""
	}
	pkg := path.Dir(rest)
	if pkg == "." {
		return ""
	}
	return pkg
}

func stdlibPackageHeading(pkg string, lastLine int) *md.Heading {
	line := lastLine + 2
	pos := md.Position{StartLine: line, EndLine: line}
	return &md.Heading{
		Position: pos,
		Level:    4,
		Text: &md.Text{
			Position: pos,
			Inline: []md.Inline{
				&md.Link{
					Inner: []md.Inline{&md.Code{Text: pkg}},
					URL:   "/pkg/" + pkg + "/",
				},
			},
		},
	}
}

// removeEmptySections removes headings with no content. A heading has no content
// if there are no blocks between it and the next heading at the same level, or the
// end of the document.
func removeEmptySections(bs []md.Block) []md.Block {
	res := bs[:0]
	delta := 0 // number of lines by which to adjust positions

	// Remove preceding headings at same or higher level; they are empty.
	rem := func(level int) {
		for len(res) > 0 {
			last := res[len(res)-1]
			if lh, ok := last.(*md.Heading); ok && lh.Level >= level {
				res = res[:len(res)-1]
				// Adjust subsequent block positions by the size of this block
				// plus 1 for the blank line between headings.
				delta += lh.EndLine - lh.StartLine + 2
			} else {
				break
			}
		}
	}

	for _, b := range bs {
		if h, ok := b.(*md.Heading); ok {
			rem(h.Level)
		}
		addLines(b, -delta)
		res = append(res, b)
	}
	// Remove empty headings at the end of the document.
	rem(1)
	return res
}

func sortedMarkdownFilenames(fsys fs.FS) ([]string, error) {
	var filenames []string
	err := fs.WalkDir(fsys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, ".md") {
			filenames = append(filenames, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	// '.' comes before '/', which comes before alphanumeric characters.
	// So just sorting the list will put a filename like "net.md" before
	// the directory "net". That is what we want.
	slices.Sort(filenames)
	return filenames, nil
}

// lastBlock returns the last block in the document.
// It panics if the document has no blocks.
func lastBlock(doc *md.Document) md.Block {
	return doc.Blocks[len(doc.Blocks)-1]
}

// addLines adds n lines to the position of b.
// n can be negative.
func addLines(b md.Block, n int) {
	pos := position(b)
	pos.StartLine += n
	pos.EndLine += n
}

func position(b md.Block) *md.Position {
	switch b := b.(type) {
	case *md.Heading:
		return &b.Position
	case *md.Text:
		return &b.Position
	case *md.CodeBlock:
		return &b.Position
	case *md.HTMLBlock:
		return &b.Position
	case *md.List:
		return &b.Position
	case *md.Item:
		return &b.Position
	case *md.Empty:
		return &b.Position
	case *md.Paragraph:
		return &b.Position
	case *md.Quote:
		return &b.Position
	case *md.ThematicBreak:
		return &b.Position
	default:
		panic(fmt.Sprintf("unknown block type %T", b))
	}
}

func parseMarkdownFile(fsys fs.FS, path string) (*md.Document, error) {
	f, err := fsys.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	in := string(data)
	doc := NewParser().Parse(in)
	return doc, nil
}

// An APIFeature is a symbol mentioned in an API file,
// like the ones in the main go repo in the api directory.
type APIFeature struct {
	Package string // package that the feature is in
	Build   string // build that the symbol is relevant for (e.g. GOOS, GOARCH)
	Feature string // everything about the feature other than the package
	Issue   int    // the issue that introduced the feature, or 0 if none
}

// This regexp has four capturing groups: package, build, feature and issue.
var apiFileLineRegexp = regexp.MustCompile(`^pkg ([^ \t]+)[ \t]*(\([^)]+\))?, ([^#]*)(#\d+)?$`)

// parseAPIFile parses a file in the api format and returns a list of the file's features.
// A feature is represented by a single line that looks like
//
//	pkg PKG (BUILD) FEATURE #ISSUE
//
// where the BUILD and ISSUE may be absent.
func parseAPIFile(fsys fs.FS, filename string) ([]APIFeature, error) {
	f, err := fsys.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var features []APIFeature
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		line := strings.TrimSpace(scan.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		matches := apiFileLineRegexp.FindStringSubmatch(line)
		if len(matches) == 0 {
			return nil, fmt.Errorf("%s: malformed line %q", filename, line)
		}
		if len(matches) != 5 {
			return nil, fmt.Errorf("wrong number of matches for line %q", line)
		}
		f := APIFeature{
			Package: matches[1],
			Build:   matches[2],
			Feature: strings.TrimSpace(matches[3]),
		}
		if issue := matches[4]; issue != "" {
			var err error
			f.Issue, err = strconv.Atoi(issue[1:]) // skip leading '#'
			if err != nil {
				return nil, err
			}
		}
		features = append(features, f)
	}
	if scan.Err() != nil {
		return nil, scan.Err()
	}
	return features, nil
}

// GroupAPIFeaturesByFile returns a map of the given features keyed by
// the doc filename that they are associated with.
// A feature with package P and issue N should be documented in the file
// "P/N.md".
func GroupAPIFeaturesByFile(fs []APIFeature) (map[string][]APIFeature, error) {
	m := map[string][]APIFeature{}
	for _, f := range fs {
		if f.Issue == 0 {
			return nil, fmt.Errorf("%+v: zero issue", f)
		}
		filename := fmt.Sprintf("%s/%d.md", f.Package, f.Issue)
		m[filename] = append(m[filename], f)
	}
	return m, nil
}

// CheckAPIFile reads the api file at filename in apiFS, and checks the corresponding
// release-note files under docFS. It checks that the files exist and that they have
// some minimal content (see [CheckFragment]).
// The docRoot argument is the path from the repo or project root to the root of docFS.
// It is used only for error messages.
func CheckAPIFile(apiFS fs.FS, filename string, docFS fs.FS, docRoot string) error {
	features, err := parseAPIFile(apiFS, filename)
	if err != nil {
		return err
	}
	byFile, err := GroupAPIFeaturesByFile(features)
	if err != nil {
		return err
	}
	var filenames []string
	for fn := range byFile {
		filenames = append(filenames, fn)
	}
	slices.Sort(filenames)
	mcDir, err := minorChangesDir(docFS)
	if err != nil {
		return err
	}
	var errs []error
	for _, fn := range filenames {
		// Use path.Join for consistency with io/fs pathnames.
		fn = path.Join(mcDir, fn)
		// TODO(jba): check that the file mentions each feature?
		if err := checkFragmentFile(docFS, fn); err != nil {
			errs = append(errs, fmt.Errorf("%s: %v\nSee doc/README.md for more information.", path.Join(docRoot, fn), err))
		}
	}
	return errors.Join(errs...)
}

// minorChangesDir returns the unique directory in docFS that corresponds to the
// "Minor changes to the standard library" section of the release notes.
func minorChangesDir(docFS fs.FS) (string, error) {
	dirs, err := fs.Glob(docFS, "*stdlib/*minor")
	if err != nil {
		return "", err
	}
	var bad string
	if len(dirs) == 0 {
		bad = "No"
	} else if len(dirs) > 1 {
		bad = "More than one"
	}
	if bad != "" {
		return "", fmt.Errorf("%s directory matches *stdlib/*minor.\nThis shouldn't happen; please file a bug at https://go.dev/issues/new.",
			bad)
	}
	return dirs[0], nil
}

func checkFragmentFile(fsys fs.FS, filename string) error {
	f, err := fsys.Open(filename)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			err = errors.New("File does not exist. Every API change must have a corresponding release note file.")
		}
		return err
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return err
	}
	return CheckFragment(string(data))
}
