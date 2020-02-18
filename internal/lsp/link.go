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

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
)

func (s *Server) documentLink(ctx context.Context, params *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	// TODO(golang/go#36501): Support document links for go.mod files.
	snapshot, fh, ok, err := s.beginFileRequest(params.TextDocument.URI, source.UnknownKind)
	if !ok {
		return nil, err
	}
	switch fh.Identity().Kind {
	case source.Mod:
		return modLinks(ctx, snapshot, fh)
	case source.Go:
		return goLinks(ctx, snapshot.View(), fh)
	}
	return nil, nil
}

func modLinks(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]protocol.DocumentLink, error) {
	view := snapshot.View()

	file, m, err := snapshot.ModHandle(ctx, fh).Parse(ctx)
	if err != nil {
		return nil, err
	}
	var links []protocol.DocumentLink
	for _, req := range file.Require {
		dep := []byte(req.Mod.Path)
		s, e := req.Syntax.Start.Byte, req.Syntax.End.Byte
		i := bytes.Index(m.Content[s:e], dep)
		if i == -1 {
			continue
		}
		// Shift the start position to the location of the
		// dependency within the require statement.
		start, end := token.Pos(s+i), token.Pos(s+i+len(dep))
		target := fmt.Sprintf("https://%s/mod/%s", view.Options().LinkTarget, req.Mod.String())
		if l, err := toProtocolLink(view, m, target, start, end, source.Mod); err == nil {
			links = append(links, l)
		} else {
			log.Error(ctx, "failed to create protocol link", err)
		}
	}
	// TODO(ridersofrohan): handle links for replace and exclude directives
	if syntax := file.Syntax; syntax == nil {
		return links, nil
	}
	// Get all the links that are contained in the comments of the file.
	for _, expr := range file.Syntax.Stmt {
		comments := expr.Comment()
		if comments == nil {
			continue
		}
		for _, cmt := range comments.Before {
			links = append(links, findLinksInString(ctx, view, cmt.Token, token.Pos(cmt.Start.Byte), m, source.Mod)...)
		}
		for _, cmt := range comments.Suffix {
			links = append(links, findLinksInString(ctx, view, cmt.Token, token.Pos(cmt.Start.Byte), m, source.Mod)...)
		}
		for _, cmt := range comments.After {
			links = append(links, findLinksInString(ctx, view, cmt.Token, token.Pos(cmt.Start.Byte), m, source.Mod)...)
		}
	}
	return links, nil
}

func goLinks(ctx context.Context, view source.View, fh source.FileHandle) ([]protocol.DocumentLink, error) {
	phs, err := view.Snapshot().PackageHandles(ctx, fh)
	if err != nil {
		return nil, err
	}
	ph, err := source.WidestPackageHandle(phs)
	if err != nil {
		return nil, err
	}
	file, _, m, _, err := view.Session().Cache().ParseGoHandle(fh, source.ParseFull).Parse(ctx)
	if err != nil {
		return nil, err
	}
	var links []protocol.DocumentLink
	ast.Inspect(file, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.ImportSpec:
			// For import specs, provide a link to a documentation website, like https://pkg.go.dev.
			if target, err := strconv.Unquote(n.Path.Value); err == nil {
				if mod, version, ok := moduleAtVersion(ctx, target, ph); ok && strings.ToLower(view.Options().LinkTarget) == "pkg.go.dev" {
					target = strings.Replace(target, mod, mod+"@"+version, 1)
				}
				target = fmt.Sprintf("https://%s/%s", view.Options().LinkTarget, target)
				// Account for the quotation marks in the positions.
				start, end := n.Path.Pos()+1, n.Path.End()-1
				if l, err := toProtocolLink(view, m, target, start, end, source.Go); err == nil {
					links = append(links, l)
				} else {
					log.Error(ctx, "failed to create protocol link", err)
				}
			}
			return false
		case *ast.BasicLit:
			// Look for links in string literals.
			if n.Kind == token.STRING {
				links = append(links, findLinksInString(ctx, view, n.Value, n.Pos(), m, source.Go)...)
			}
			return false
		}
		return true
	})
	// Look for links in comments.
	for _, commentGroup := range file.Comments {
		for _, comment := range commentGroup.List {
			links = append(links, findLinksInString(ctx, view, comment.Text, comment.Pos(), m, source.Go)...)
		}
	}
	return links, nil
}

func moduleAtVersion(ctx context.Context, target string, ph source.PackageHandle) (string, string, bool) {
	pkg, err := ph.Check(ctx)
	if err != nil {
		return "", "", false
	}
	impPkg, err := pkg.GetImport(target)
	if err != nil {
		return "", "", false
	}
	if impPkg.Module() == nil {
		return "", "", false
	}
	version, modpath := impPkg.Module().Version, impPkg.Module().Path
	if modpath == "" || version == "" {
		return "", "", false
	}
	return modpath, version, true
}

func findLinksInString(ctx context.Context, view source.View, src string, pos token.Pos, m *protocol.ColumnMapper, fileKind source.FileKind) []protocol.DocumentLink {
	var links []protocol.DocumentLink
	for _, index := range view.Options().URLRegexp.FindAllIndex([]byte(src), -1) {
		start, end := index[0], index[1]
		startPos := token.Pos(int(pos) + start)
		endPos := token.Pos(int(pos) + end)
		url, err := url.Parse(src[start:end])
		if err != nil {
			log.Error(ctx, "failed to parse matching URL", err)
			continue
		}
		// If the URL has no scheme, use https.
		if url.Scheme == "" {
			url.Scheme = "https"
		}
		l, err := toProtocolLink(view, m, url.String(), startPos, endPos, fileKind)
		if err != nil {
			log.Error(ctx, "failed to create protocol link", err)
			continue
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
		target := fmt.Sprintf("https://github.com/%s/%s/issues/%s", org, repo, number)
		l, err := toProtocolLink(view, m, target, startPos, endPos, fileKind)
		if err != nil {
			log.Error(ctx, "failed to create protocol link", err)
			continue
		}
		links = append(links, l)
	}
	return links
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

func toProtocolLink(view source.View, m *protocol.ColumnMapper, target string, start, end token.Pos, fileKind source.FileKind) (protocol.DocumentLink, error) {
	var rng protocol.Range
	switch fileKind {
	case source.Go:
		spn, err := span.NewRange(view.Session().Cache().FileSet(), start, end).Span()
		if err != nil {
			return protocol.DocumentLink{}, err
		}
		rng, err = m.Range(spn)
		if err != nil {
			return protocol.DocumentLink{}, err
		}
	case source.Mod:
		s, e := int(start), int(end)
		line, col, err := m.Converter.ToPosition(s)
		if err != nil {
			return protocol.DocumentLink{}, err
		}
		start := span.NewPoint(line, col, s)
		line, col, err = m.Converter.ToPosition(e)
		if err != nil {
			return protocol.DocumentLink{}, err
		}
		end := span.NewPoint(line, col, e)
		rng, err = m.Range(span.New(m.URI, start, end))
		if err != nil {
			return protocol.DocumentLink{}, err
		}
	}
	return protocol.DocumentLink{
		Range:  rng,
		Target: target,
	}, nil
}
