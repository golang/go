// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"context"
	"errors"
	"fmt"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func Completion(ctx context.Context, snapshot source.Snapshot, fh source.VersionedFileHandle, position protocol.Position) (*protocol.CompletionList, error) {
	ctx, done := event.Start(ctx, "work.Completion")
	defer done()

	// Get the position of the cursor.
	pw, err := snapshot.ParseWork(ctx, fh)
	if err != nil {
		return nil, fmt.Errorf("getting go.work file handle: %w", err)
	}
	spn, err := pw.Mapper.PointSpan(position)
	if err != nil {
		return nil, fmt.Errorf("computing cursor position: %w", err)
	}
	rng, err := spn.Range(pw.Mapper.Converter)
	if err != nil {
		return nil, fmt.Errorf("computing range: %w", err)
	}

	// Find the use statement the user is in.
	cursor := rng.Start - 1
	use, pathStart, _ := usePath(pw, cursor)
	if use == nil {
		return &protocol.CompletionList{}, nil
	}
	completingFrom := use.Path[:cursor-token.Pos(pathStart)]

	// We're going to find the completions of the user input
	// (completingFrom) by doing a walk on the innermost directory
	// of the given path, and comparing the found paths to make sure
	// that they match the component of the path after the
	// innermost directory.
	//
	// We'll maintain two paths when doing this: pathPrefixSlash
	// is essentially the path the user typed in, and pathPrefixAbs
	// is the path made absolute from the go.work directory.

	pathPrefixSlash := completingFrom
	pathPrefixAbs := filepath.FromSlash(pathPrefixSlash)
	if !filepath.IsAbs(pathPrefixAbs) {
		pathPrefixAbs = filepath.Join(filepath.Dir(pw.URI.Filename()), pathPrefixAbs)
	}

	// pathPrefixDir is the directory that will be walked to find matches.
	// If pathPrefixSlash is not explicitly a directory boundary (is either equivalent to "." or
	// ends in a separator) we need to examine its parent directory to find sibling files that
	// match.
	depthBound := 5
	pathPrefixDir, pathPrefixBase := pathPrefixAbs, ""
	pathPrefixSlashDir := pathPrefixSlash
	if filepath.Clean(pathPrefixSlash) != "." && !strings.HasSuffix(pathPrefixSlash, "/") {
		depthBound++
		pathPrefixDir, pathPrefixBase = filepath.Split(pathPrefixAbs)
		pathPrefixSlashDir = dirNonClean(pathPrefixSlash)
	}

	var completions []string
	// Stop traversing deeper once we've hit 10k files to try to stay generally under 100ms.
	const numSeenBound = 10000
	var numSeen int
	stopWalking := errors.New("hit numSeenBound")
	err = filepath.Walk(pathPrefixDir, func(wpath string, info os.FileInfo, err error) error {
		if numSeen > numSeenBound {
			// Stop traversing if we hit bound.
			return stopWalking
		}
		numSeen++

		// rel is the path relative to pathPrefixDir.
		// Make sure that it has pathPrefixBase as a prefix
		// otherwise it won't match the beginning of the
		// base component of the path the user typed in.
		rel := strings.TrimPrefix(wpath[len(pathPrefixDir):], string(filepath.Separator))
		if info.IsDir() && wpath != pathPrefixDir && !strings.HasPrefix(rel, pathPrefixBase) {
			return filepath.SkipDir
		}

		// Check for a match (a module directory).
		if filepath.Base(rel) == "go.mod" {
			relDir := strings.TrimSuffix(dirNonClean(rel), string(os.PathSeparator))
			completionPath := join(pathPrefixSlashDir, filepath.ToSlash(relDir))

			if !strings.HasPrefix(completionPath, completingFrom) {
				return nil
			}
			if strings.HasSuffix(completionPath, "/") {
				// Don't suggest paths that end in "/". This happens
				// when the input is a path that ends in "/" and
				// the completion is empty.
				return nil
			}
			completion := completionPath[len(completingFrom):]
			if completingFrom == "" && !strings.HasPrefix(completion, "./") {
				// Bias towards "./" prefixes.
				completion = join(".", completion)
			}

			completions = append(completions, completion)
		}

		if depth := strings.Count(rel, string(filepath.Separator)); depth >= depthBound {
			return filepath.SkipDir
		}
		return nil
	})
	if err != nil && !errors.Is(err, stopWalking) {
		return nil, fmt.Errorf("walking to find completions: %w", err)
	}

	sort.Strings(completions)

	var items []protocol.CompletionItem
	for _, c := range completions {
		items = append(items, protocol.CompletionItem{
			Label:      c,
			InsertText: c,
		})
	}
	return &protocol.CompletionList{Items: items}, nil
}

// dirNonClean is filepath.Dir, without the Clean at the end.
func dirNonClean(path string) string {
	vol := filepath.VolumeName(path)
	i := len(path) - 1
	for i >= len(vol) && !os.IsPathSeparator(path[i]) {
		i--
	}
	return path[len(vol) : i+1]
}

func join(a, b string) string {
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	return strings.TrimSuffix(a, "/") + "/" + b
}
