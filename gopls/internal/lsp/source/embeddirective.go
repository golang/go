// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// ErrNoEmbed is returned by EmbedDefinition when no embed
// directive is found at a particular position.
// As such it indicates that other definitions could be worth checking.
var ErrNoEmbed = errors.New("no embed directive found")

var errStopWalk = errors.New("stop walk")

// EmbedDefinition finds a file matching the embed directive at pos in the mapped file.
// If there is no embed directive at pos, returns ErrNoEmbed.
// If multiple files match the embed pattern, one is picked at random.
func EmbedDefinition(m *protocol.Mapper, pos protocol.Position) ([]protocol.Location, error) {
	pattern, _ := parseEmbedDirective(m, pos)
	if pattern == "" {
		return nil, ErrNoEmbed
	}

	// Find the first matching file.
	var match string
	dir := filepath.Dir(m.URI.Filename())
	err := filepath.WalkDir(dir, func(abs string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		rel, err := filepath.Rel(dir, abs)
		if err != nil {
			return err
		}
		ok, err := filepath.Match(pattern, rel)
		if err != nil {
			return err
		}
		if ok && !d.IsDir() {
			match = abs
			return errStopWalk
		}
		return nil
	})
	if err != nil && !errors.Is(err, errStopWalk) {
		return nil, err
	}
	if match == "" {
		return nil, fmt.Errorf("%q does not match any files in %q", pattern, dir)
	}

	loc := protocol.Location{
		URI: protocol.URIFromPath(match),
		Range: protocol.Range{
			Start: protocol.Position{Line: 0, Character: 0},
		},
	}
	return []protocol.Location{loc}, nil
}

// parseEmbedDirective attempts to parse a go:embed directive argument at pos.
// If successful it return the directive argument and its range, else zero values are returned.
func parseEmbedDirective(m *protocol.Mapper, pos protocol.Position) (string, protocol.Range) {
	lineStart, err := m.PositionOffset(protocol.Position{Line: pos.Line, Character: 0})
	if err != nil {
		return "", protocol.Range{}
	}
	lineEnd, err := m.PositionOffset(protocol.Position{Line: pos.Line + 1, Character: 0})
	if err != nil {
		return "", protocol.Range{}
	}

	text := string(m.Content[lineStart:lineEnd])
	if !strings.HasPrefix(text, "//go:embed") {
		return "", protocol.Range{}
	}
	text = text[len("//go:embed"):]
	offset := lineStart + len("//go:embed")

	// Find the first pattern in text that covers the offset of the pos we are looking for.
	findOffset, err := m.PositionOffset(pos)
	if err != nil {
		return "", protocol.Range{}
	}
	patterns, err := parseGoEmbed(text, offset)
	if err != nil {
		return "", protocol.Range{}
	}
	for _, p := range patterns {
		if p.startOffset <= findOffset && findOffset <= p.endOffset {
			// Found our match.
			rng, err := m.OffsetRange(p.startOffset, p.endOffset)
			if err != nil {
				return "", protocol.Range{}
			}
			return p.pattern, rng
		}
	}

	return "", protocol.Range{}
}

type fileEmbed struct {
	pattern     string
	startOffset int
	endOffset   int
}

// parseGoEmbed patterns that come after the directive.
//
// Copied and adapted from go/build/read.go.
// Replaced token.Position with start/end offset (including quotes if present).
func parseGoEmbed(args string, offset int) ([]fileEmbed, error) {
	trimBytes := func(n int) {
		offset += n
		args = args[n:]
	}
	trimSpace := func() {
		trim := strings.TrimLeftFunc(args, unicode.IsSpace)
		trimBytes(len(args) - len(trim))
	}

	var list []fileEmbed
	for trimSpace(); args != ""; trimSpace() {
		var path string
		pathOffset := offset
	Switch:
		switch args[0] {
		default:
			i := len(args)
			for j, c := range args {
				if unicode.IsSpace(c) {
					i = j
					break
				}
			}
			path = args[:i]
			trimBytes(i)

		case '`':
			var ok bool
			path, _, ok = strings.Cut(args[1:], "`")
			if !ok {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
			trimBytes(1 + len(path) + 1)

		case '"':
			i := 1
			for ; i < len(args); i++ {
				if args[i] == '\\' {
					i++
					continue
				}
				if args[i] == '"' {
					q, err := strconv.Unquote(args[:i+1])
					if err != nil {
						return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args[:i+1])
					}
					path = q
					trimBytes(i + 1)
					break Switch
				}
			}
			if i >= len(args) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}

		if args != "" {
			r, _ := utf8.DecodeRuneInString(args)
			if !unicode.IsSpace(r) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}
		list = append(list, fileEmbed{
			pattern:     path,
			startOffset: pathOffset,
			endOffset:   offset,
		})
	}
	return list, nil
}
