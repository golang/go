// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"net/url"
	"regexp"
	"strconv"
	"sync"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
)

func (s *Server) documentLink(ctx context.Context, params *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	fh, err := view.Snapshot().GetFile(uri)
	if err != nil {
		return nil, err
	}
	// TODO(golang/go#36501): Support document links for go.mod files.
	if fh.Identity().Kind == source.Mod {
		return nil, nil
	}
	file, m, _, err := view.Session().Cache().ParseGoHandle(fh, source.ParseFull).Parse(ctx)
	if err != nil {
		return nil, err
	}
	var links []protocol.DocumentLink
	ast.Inspect(file, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.ImportSpec:
			// For import specs, provide a link to a documentation website, like https://pkg.go.dev.
			if target, err := strconv.Unquote(n.Path.Value); err == nil {
				target = fmt.Sprintf("https://%s/%s", view.Options().LinkTarget, target)

				// Account for the quotation marks in the positions.
				start, end := n.Path.Pos()+1, n.Path.End()-1
				if l, err := toProtocolLink(view, m, target, start, end); err == nil {
					links = append(links, l)
				} else {
					log.Error(ctx, "failed to create protocol link", err)
				}
			}
			return false
		case *ast.BasicLit:
			// Look for links in string literals.
			if n.Kind == token.STRING {
				links = append(links, findLinksInString(ctx, view, n.Value, n.Pos(), m)...)
			}
			return false
		}
		return true
	})
	// Look for links in comments.
	for _, commentGroup := range file.Comments {
		for _, comment := range commentGroup.List {
			links = append(links, findLinksInString(ctx, view, comment.Text, comment.Pos(), m)...)
		}
	}
	return links, nil
}

func findLinksInString(ctx context.Context, view source.View, src string, pos token.Pos, m *protocol.ColumnMapper) []protocol.DocumentLink {
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
		l, err := toProtocolLink(view, m, url.String(), startPos, endPos)
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
		l, err := toProtocolLink(view, m, target, startPos, endPos)
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

func toProtocolLink(view source.View, m *protocol.ColumnMapper, target string, start, end token.Pos) (protocol.DocumentLink, error) {
	spn, err := span.NewRange(view.Session().Cache().FileSet(), start, end).Span()
	if err != nil {
		return protocol.DocumentLink{}, err
	}
	rng, err := m.Range(spn)
	if err != nil {
		return protocol.DocumentLink{}, err
	}
	return protocol.DocumentLink{
		Range:  rng,
		Target: target,
	}, nil
}
