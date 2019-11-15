// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"go/ast"
	"go/token"
	"regexp"
	"strconv"
	"sync"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	errors "golang.org/x/xerrors"
)

func (s *Server) documentLink(ctx context.Context, params *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	fh := view.Snapshot().Handle(ctx, f)
	file, m, _, err := view.Session().Cache().ParseGoHandle(fh, source.ParseFull).Parse(ctx)
	if err != nil {
		return nil, err
	}
	var links []protocol.DocumentLink
	ast.Inspect(file, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.ImportSpec:
			target, err := strconv.Unquote(n.Path.Value)
			if err != nil {
				log.Error(ctx, "cannot unquote import path", err, tag.Of("Path", n.Path.Value))
				return false
			}
			if target == "" {
				return false
			}
			target = "https://godoc.org/" + target
			l, err := toProtocolLink(view, m, target, n.Path.Pos()+1, n.Path.End()-1)
			if err != nil {
				log.Error(ctx, "cannot initialize DocumentLink", err, tag.Of("Path", n.Path.Value))
				return false
			}
			links = append(links, l)
			return false
		case *ast.BasicLit:
			if n.Kind != token.STRING {
				return false
			}
			l, err := findLinksInString(n.Value, n.Pos(), view, m)
			if err != nil {
				log.Error(ctx, "cannot find links in string", err)
				return false
			}
			links = append(links, l...)
			return false
		}
		return true
	})

	for _, commentGroup := range file.Comments {
		for _, comment := range commentGroup.List {
			l, err := findLinksInString(comment.Text, comment.Pos(), view, m)
			if err != nil {
				log.Error(ctx, "cannot find links in comment", err)
				continue
			}
			links = append(links, l...)
		}
	}

	return links, nil
}

func findLinksInString(src string, pos token.Pos, view source.View, mapper *protocol.ColumnMapper) ([]protocol.DocumentLink, error) {
	var links []protocol.DocumentLink
	re, err := getURLRegexp()
	if err != nil {
		return nil, errors.Errorf("cannot create regexp for links: %s", err.Error())
	}
	for _, urlIndex := range re.FindAllIndex([]byte(src), -1) {
		start := urlIndex[0]
		end := urlIndex[1]
		startPos := token.Pos(int(pos) + start)
		endPos := token.Pos(int(pos) + end)
		target := src[start:end]
		l, err := toProtocolLink(view, mapper, target, startPos, endPos)
		if err != nil {
			return nil, err
		}
		links = append(links, l)
	}
	return links, nil
}

const urlRegexpString = "(http|ftp|https)://([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:/~+#-]*[\\w@?^=%&/~+#-])?"

var (
	urlRegexp  *regexp.Regexp
	regexpOnce sync.Once
	regexpErr  error
)

func getURLRegexp() (*regexp.Regexp, error) {
	regexpOnce.Do(func() {
		urlRegexp, regexpErr = regexp.Compile(urlRegexpString)
	})
	return urlRegexp, regexpErr
}

func toProtocolLink(view source.View, mapper *protocol.ColumnMapper, target string, start, end token.Pos) (protocol.DocumentLink, error) {
	spn, err := span.NewRange(view.Session().Cache().FileSet(), start, end).Span()
	if err != nil {
		return protocol.DocumentLink{}, err
	}
	rng, err := mapper.Range(spn)
	if err != nil {
		return protocol.DocumentLink{}, err
	}
	l := protocol.DocumentLink{
		Range:  rng,
		Target: target,
	}
	return l, nil
}
