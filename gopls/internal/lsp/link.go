// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"net/url"
	"regexp"
	"strings"
	"sync"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

func (s *Server) documentLink(ctx context.Context, params *protocol.DocumentLinkParams) (links []protocol.DocumentLink, err error) {
	ctx, done := event.Start(ctx, "lsp.Server.documentLink")
	defer done()

	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	switch snapshot.FileKind(fh) {
	case source.Mod:
		links, err = modLinks(ctx, snapshot, fh)
	case source.Go:
		links, err = goLinks(ctx, snapshot, fh)
	}
	// Don't return errors for document links.
	if err != nil {
		event.Error(ctx, "failed to compute document links", err, tag.URI.Of(fh.URI()))
		return nil, nil
	}
	return links, nil
}

func modLinks(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.DocumentLink, error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil, err
	}

	var links []protocol.DocumentLink
	for _, req := range pm.File.Require {
		if req.Syntax == nil {
			continue
		}
		// See golang/go#36998: don't link to modules matching GOPRIVATE.
		if snapshot.View().IsGoPrivatePath(req.Mod.Path) {
			continue
		}
		dep := []byte(req.Mod.Path)
		start, end := req.Syntax.Start.Byte, req.Syntax.End.Byte
		i := bytes.Index(pm.Mapper.Content[start:end], dep)
		if i == -1 {
			continue
		}
		// Shift the start position to the location of the
		// dependency within the require statement.
		target := source.BuildLink(snapshot.Options().LinkTarget, "mod/"+req.Mod.String(), "")
		l, err := toProtocolLink(pm.Mapper, target, start+i, start+i+len(dep))
		if err != nil {
			return nil, err
		}
		links = append(links, l)
	}
	// TODO(ridersofrohan): handle links for replace and exclude directives.
	if syntax := pm.File.Syntax; syntax == nil {
		return links, nil
	}

	// Get all the links that are contained in the comments of the file.
	urlRegexp := snapshot.Options().URLRegexp
	for _, expr := range pm.File.Syntax.Stmt {
		comments := expr.Comment()
		if comments == nil {
			continue
		}
		for _, section := range [][]modfile.Comment{comments.Before, comments.Suffix, comments.After} {
			for _, comment := range section {
				l, err := findLinksInString(urlRegexp, comment.Token, comment.Start.Byte, pm.Mapper)
				if err != nil {
					return nil, err
				}
				links = append(links, l...)
			}
		}
	}
	return links, nil
}

// goLinks returns the set of hyperlink annotations for the specified Go file.
func goLinks(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.DocumentLink, error) {

	pgf, err := snapshot.ParseGo(ctx, fh, source.ParseFull)
	if err != nil {
		return nil, err
	}

	var links []protocol.DocumentLink

	// Create links for import specs.
	if snapshot.Options().ImportShortcut.ShowLinks() {

		// If links are to pkg.go.dev, append module version suffixes.
		// This requires the import map from the package metadata. Ignore errors.
		var depsByImpPath map[source.ImportPath]source.PackageID
		if strings.ToLower(snapshot.Options().LinkTarget) == "pkg.go.dev" {
			if meta, err := source.NarrowestMetadataForFile(ctx, snapshot, fh.URI()); err == nil {
				depsByImpPath = meta.DepsByImpPath
			}
		}

		for _, imp := range pgf.File.Imports {
			importPath := source.UnquoteImportPath(imp)
			if importPath == "" {
				continue // bad import
			}
			// See golang/go#36998: don't link to modules matching GOPRIVATE.
			if snapshot.View().IsGoPrivatePath(string(importPath)) {
				continue
			}

			urlPath := string(importPath)

			// For pkg.go.dev, append module version suffix to package import path.
			if m := snapshot.Metadata(depsByImpPath[importPath]); m != nil && m.Module != nil && m.Module.Path != "" && m.Module.Version != "" {
				urlPath = strings.Replace(urlPath, m.Module.Path, m.Module.Path+"@"+m.Module.Version, 1)
			}

			start, end, err := safetoken.Offsets(pgf.Tok, imp.Path.Pos(), imp.Path.End())
			if err != nil {
				return nil, err
			}
			targetURL := source.BuildLink(snapshot.Options().LinkTarget, urlPath, "")
			// Account for the quotation marks in the positions.
			l, err := toProtocolLink(pgf.Mapper, targetURL, start+len(`"`), end-len(`"`))
			if err != nil {
				return nil, err
			}
			links = append(links, l)
		}
	}

	urlRegexp := snapshot.Options().URLRegexp

	// Gather links found in string literals.
	var str []*ast.BasicLit
	ast.Inspect(pgf.File, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.ImportSpec:
			return false // don't process import strings again
		case *ast.BasicLit:
			if n.Kind == token.STRING {
				str = append(str, n)
			}
		}
		return true
	})
	for _, s := range str {
		strOffset, err := safetoken.Offset(pgf.Tok, s.Pos())
		if err != nil {
			return nil, err
		}
		l, err := findLinksInString(urlRegexp, s.Value, strOffset, pgf.Mapper)
		if err != nil {
			return nil, err
		}
		links = append(links, l...)
	}

	// Gather links found in comments.
	for _, commentGroup := range pgf.File.Comments {
		for _, comment := range commentGroup.List {
			commentOffset, err := safetoken.Offset(pgf.Tok, comment.Pos())
			if err != nil {
				return nil, err
			}
			l, err := findLinksInString(urlRegexp, comment.Text, commentOffset, pgf.Mapper)
			if err != nil {
				return nil, err
			}
			links = append(links, l...)
		}
	}

	return links, nil
}

// acceptedSchemes controls the schemes that URLs must have to be shown to the
// user. Other schemes can't be opened by LSP clients, so linkifying them is
// distracting. See golang/go#43990.
var acceptedSchemes = map[string]bool{
	"http":  true,
	"https": true,
}

// urlRegexp is the user-supplied regular expression to match URL.
// srcOffset is the start offset of 'src' within m's file.
func findLinksInString(urlRegexp *regexp.Regexp, src string, srcOffset int, m *protocol.Mapper) ([]protocol.DocumentLink, error) {
	var links []protocol.DocumentLink
	for _, index := range urlRegexp.FindAllIndex([]byte(src), -1) {
		start, end := index[0], index[1]
		link := src[start:end]
		linkURL, err := url.Parse(link)
		// Fallback: Linkify IP addresses as suggested in golang/go#18824.
		if err != nil {
			linkURL, err = url.Parse("//" + link)
			// Not all potential links will be valid, so don't return this error.
			if err != nil {
				continue
			}
		}
		// If the URL has no scheme, use https.
		if linkURL.Scheme == "" {
			linkURL.Scheme = "https"
		}
		if !acceptedSchemes[linkURL.Scheme] {
			continue
		}

		l, err := toProtocolLink(m, linkURL.String(), srcOffset+start, srcOffset+end)
		if err != nil {
			return nil, err
		}
		links = append(links, l)
	}
	// Handle golang/go#1234-style links.
	r := getIssueRegexp()
	for _, index := range r.FindAllIndex([]byte(src), -1) {
		start, end := index[0], index[1]
		matches := r.FindStringSubmatch(src)
		if len(matches) < 4 {
			continue
		}
		org, repo, number := matches[1], matches[2], matches[3]
		targetURL := fmt.Sprintf("https://github.com/%s/%s/issues/%s", org, repo, number)
		l, err := toProtocolLink(m, targetURL, srcOffset+start, srcOffset+end)
		if err != nil {
			return nil, err
		}
		links = append(links, l)
	}
	return links, nil
}

func getIssueRegexp() *regexp.Regexp {
	once.Do(func() {
		issueRegexp = regexp.MustCompile(`(\w+)/([\w-]+)#([0-9]+)`)
	})
	return issueRegexp
}

var (
	once        sync.Once
	issueRegexp *regexp.Regexp
)

func toProtocolLink(m *protocol.Mapper, targetURL string, start, end int) (protocol.DocumentLink, error) {
	rng, err := m.OffsetRange(start, end)
	if err != nil {
		return protocol.DocumentLink{}, err
	}
	return protocol.DocumentLink{
		Range:  rng,
		Target: &targetURL,
	}, nil
}
