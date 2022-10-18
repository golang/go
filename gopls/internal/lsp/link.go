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
	"strconv"
	"strings"
	"sync"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

func (s *Server) documentLink(ctx context.Context, params *protocol.DocumentLinkParams) (links []protocol.DocumentLink, err error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	switch snapshot.View().FileKind(fh) {
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
	tokFile := pm.Mapper.TokFile

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
		s, e := req.Syntax.Start.Byte, req.Syntax.End.Byte
		i := bytes.Index(pm.Mapper.Content[s:e], dep)
		if i == -1 {
			continue
		}
		// Shift the start position to the location of the
		// dependency within the require statement.
		start, end := tokFile.Pos(s+i), tokFile.Pos(s+i+len(dep))
		target := source.BuildLink(snapshot.View().Options().LinkTarget, "mod/"+req.Mod.String(), "")
		l, err := toProtocolLink(tokFile, pm.Mapper, target, start, end)
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
	for _, expr := range pm.File.Syntax.Stmt {
		comments := expr.Comment()
		if comments == nil {
			continue
		}
		for _, section := range [][]modfile.Comment{comments.Before, comments.Suffix, comments.After} {
			for _, comment := range section {
				start := tokFile.Pos(comment.Start.Byte)
				l, err := findLinksInString(ctx, snapshot, comment.Token, start, tokFile, pm.Mapper)
				if err != nil {
					return nil, err
				}
				links = append(links, l...)
			}
		}
	}
	return links, nil
}

func goLinks(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.DocumentLink, error) {
	view := snapshot.View()
	// We don't actually need type information, so any typecheck mode is fine.
	pkg, err := snapshot.PackageForFile(ctx, fh.URI(), source.TypecheckWorkspace, source.WidestPackage)
	if err != nil {
		return nil, err
	}
	pgf, err := snapshot.ParseGo(ctx, fh, source.ParseFull)
	if err != nil {
		return nil, err
	}
	var imports []*ast.ImportSpec
	var str []*ast.BasicLit
	ast.Inspect(pgf.File, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.ImportSpec:
			imports = append(imports, n)
			return false
		case *ast.BasicLit:
			// Look for links in string literals.
			if n.Kind == token.STRING {
				str = append(str, n)
			}
			return false
		}
		return true
	})
	var links []protocol.DocumentLink
	// For import specs, provide a link to a documentation website, like
	// https://pkg.go.dev.
	if view.Options().ImportShortcut.ShowLinks() {
		for _, imp := range imports {
			target, err := strconv.Unquote(imp.Path.Value)
			if err != nil {
				continue
			}
			// See golang/go#36998: don't link to modules matching GOPRIVATE.
			if view.IsGoPrivatePath(target) {
				continue
			}
			if mod, version, ok := moduleAtVersion(target, pkg); ok && strings.ToLower(view.Options().LinkTarget) == "pkg.go.dev" {
				target = strings.Replace(target, mod, mod+"@"+version, 1)
			}
			// Account for the quotation marks in the positions.
			start := imp.Path.Pos() + 1
			end := imp.Path.End() - 1
			targetURL := source.BuildLink(view.Options().LinkTarget, target, "")
			l, err := toProtocolLink(pgf.Tok, pgf.Mapper, targetURL, start, end)
			if err != nil {
				return nil, err
			}
			links = append(links, l)
		}
	}
	for _, s := range str {
		l, err := findLinksInString(ctx, snapshot, s.Value, s.Pos(), pgf.Tok, pgf.Mapper)
		if err != nil {
			return nil, err
		}
		links = append(links, l...)
	}
	for _, commentGroup := range pgf.File.Comments {
		for _, comment := range commentGroup.List {
			l, err := findLinksInString(ctx, snapshot, comment.Text, comment.Pos(), pgf.Tok, pgf.Mapper)
			if err != nil {
				return nil, err
			}
			links = append(links, l...)
		}
	}
	return links, nil
}

func moduleAtVersion(targetImportPath string, pkg source.Package) (string, string, bool) {
	impPkg, err := pkg.ResolveImportPath(targetImportPath)
	if err != nil {
		return "", "", false
	}
	if impPkg.Version() == nil {
		return "", "", false
	}
	version, modpath := impPkg.Version().Version, impPkg.Version().Path
	if modpath == "" || version == "" {
		return "", "", false
	}
	return modpath, version, true
}

// acceptedSchemes controls the schemes that URLs must have to be shown to the
// user. Other schemes can't be opened by LSP clients, so linkifying them is
// distracting. See golang/go#43990.
var acceptedSchemes = map[string]bool{
	"http":  true,
	"https": true,
}

// tokFile may be a throwaway File for non-Go files.
func findLinksInString(ctx context.Context, snapshot source.Snapshot, src string, pos token.Pos, tokFile *token.File, m *protocol.ColumnMapper) ([]protocol.DocumentLink, error) {
	var links []protocol.DocumentLink
	for _, index := range snapshot.View().Options().URLRegexp.FindAllIndex([]byte(src), -1) {
		start, end := index[0], index[1]
		startPos := token.Pos(int(pos) + start)
		endPos := token.Pos(int(pos) + end)
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
		l, err := toProtocolLink(tokFile, m, linkURL.String(), startPos, endPos)
		if err != nil {
			return nil, err
		}
		links = append(links, l)
	}
	// Handle golang/go#1234-style links.
	r := getIssueRegexp()
	for _, index := range r.FindAllIndex([]byte(src), -1) {
		start, end := index[0], index[1]
		startPos := token.Pos(int(pos) + start)
		endPos := token.Pos(int(pos) + end)
		matches := r.FindStringSubmatch(src)
		if len(matches) < 4 {
			continue
		}
		org, repo, number := matches[1], matches[2], matches[3]
		targetURL := fmt.Sprintf("https://github.com/%s/%s/issues/%s", org, repo, number)
		l, err := toProtocolLink(tokFile, m, targetURL, startPos, endPos)
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

func toProtocolLink(tokFile *token.File, m *protocol.ColumnMapper, targetURL string, start, end token.Pos) (protocol.DocumentLink, error) {
	spn, err := span.NewRange(tokFile, start, end).Span()
	if err != nil {
		return protocol.DocumentLink{}, err
	}
	rng, err := m.Range(spn)
	if err != nil {
		return protocol.DocumentLink{}, err
	}
	return protocol.DocumentLink{
		Range:  rng,
		Target: targetURL,
	}, nil
}
