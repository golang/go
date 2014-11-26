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
	"benchmarks": {"https://code.google.com/p/go.benchmarks", "hg"},
	"blog":       {"https://code.google.com/p/go.blog", "hg"},
	"codereview": {"https://code.google.com/p/go.codereview", "hg"},
	"crypto":     {"https://code.google.com/p/go.crypto", "hg"},
	"exp":        {"https://code.google.com/p/go.exp", "hg"},
	"image":      {"https://code.google.com/p/go.image", "hg"},
	"mobile":     {"https://code.google.com/p/go.mobile", "hg"},
	"net":        {"https://code.google.com/p/go.net", "hg"},
	"sys":        {"https://code.google.com/p/go.sys", "hg"},
	"talks":      {"https://code.google.com/p/go.talks", "hg"},
	"text":       {"https://code.google.com/p/go.text", "hg"},
	"tools":      {"https://code.google.com/p/go.tools", "hg"},

	"oauth2": {"https://go.googlesource.com/oauth2", "git"},
	"review": {"https://go.googlesource.com/review", "git"},
}

func init() {
	http.HandleFunc(xPrefix, xHandler)
}

func xHandler(w http.ResponseWriter, r *http.Request) {
	head, tail := strings.TrimPrefix(r.URL.Path, xPrefix), ""
	if i := strings.Index(head, "/"); i != -1 {
		head, tail = head[:i], head[i:]
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
<meta http-equiv="refresh" content="0; url=https://godoc.org/golang.org{{.Prefix}}{{.Head}}{{.Tail}}">
</head>
<body>
Nothing to see here; <a href="https://godoc.org/golang.org{{.Prefix}}{{.Head}}{{.Tail}}">move along</a>.
</body>
</html>
`))
