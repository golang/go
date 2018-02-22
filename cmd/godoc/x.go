// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the handlers that serve go-import redirects for Go
// sub-repositories. It specifies the mapping from import paths like
// "golang.org/x/tools" to the actual repository locations.

package main

import (
	"html/template"
	"log"
	"net/http"
	"strings"
)

const xPrefix = "/x/"

type xRepo struct {
	URL, VCS string
}

var xMap = map[string]xRepo{
	"codereview": {"https://code.google.com/p/go.codereview", "hg"},

	"arch":       {"https://go.googlesource.com/arch", "git"},
	"benchmarks": {"https://go.googlesource.com/benchmarks", "git"},
	"blog":       {"https://go.googlesource.com/blog", "git"},
	"build":      {"https://go.googlesource.com/build", "git"},
	"crypto":     {"https://go.googlesource.com/crypto", "git"},
	"debug":      {"https://go.googlesource.com/debug", "git"},
	"exp":        {"https://go.googlesource.com/exp", "git"},
	"image":      {"https://go.googlesource.com/image", "git"},
	"lint":       {"https://go.googlesource.com/lint", "git"},
	"mobile":     {"https://go.googlesource.com/mobile", "git"},
	"net":        {"https://go.googlesource.com/net", "git"},
	"oauth2":     {"https://go.googlesource.com/oauth2", "git"},
	"perf":       {"https://go.googlesource.com/perf", "git"},
	"playground": {"https://go.googlesource.com/playground", "git"},
	"review":     {"https://go.googlesource.com/review", "git"},
	"sync":       {"https://go.googlesource.com/sync", "git"},
	"sys":        {"https://go.googlesource.com/sys", "git"},
	"talks":      {"https://go.googlesource.com/talks", "git"},
	"term":       {"https://go.googlesource.com/term", "git"},
	"text":       {"https://go.googlesource.com/text", "git"},
	"time":       {"https://go.googlesource.com/time", "git"},
	"tools":      {"https://go.googlesource.com/tools", "git"},
	"tour":       {"https://go.googlesource.com/tour", "git"},
	"vgo":        {"https://go.googlesource.com/vgo", "git"},
}

func init() {
	http.HandleFunc(xPrefix, xHandler)
}

func xHandler(w http.ResponseWriter, r *http.Request) {
	head, tail := strings.TrimPrefix(r.URL.Path, xPrefix), ""
	if i := strings.Index(head, "/"); i != -1 {
		head, tail = head[:i], head[i:]
	}
	if head == "" {
		http.Redirect(w, r, "https://godoc.org/-/subrepo", http.StatusTemporaryRedirect)
		return
	}
	repo, ok := xMap[head]
	if !ok {
		http.NotFound(w, r)
		return
	}
	data := struct {
		Prefix, Head, Tail string
		Repo               xRepo
	}{xPrefix, head, tail, repo}
	if err := xTemplate.Execute(w, data); err != nil {
		log.Println("xHandler:", err)
	}
}

var xTemplate = template.Must(template.New("x").Parse(`<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<meta name="go-import" content="golang.org{{.Prefix}}{{.Head}} {{.Repo.VCS}} {{.Repo.URL}}">
<meta name="go-source" content="golang.org{{.Prefix}}{{.Head}} https://github.com/golang/{{.Head}}/ https://github.com/golang/{{.Head}}/tree/master{/dir} https://github.com/golang/{{.Head}}/blob/master{/dir}/{file}#L{line}">
<meta http-equiv="refresh" content="0; url=https://godoc.org/golang.org{{.Prefix}}{{.Head}}{{.Tail}}">
</head>
<body>
Nothing to see here; <a href="https://godoc.org/golang.org{{.Prefix}}{{.Head}}{{.Tail}}">move along</a>.
</body>
</html>
`))
