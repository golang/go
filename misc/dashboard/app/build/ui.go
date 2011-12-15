// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(adg): packages at weekly/release
// TODO(adg): some means to register new packages

package build

import (
	"appengine"
	"appengine/datastore"
	"exp/template/html"
	"http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"template"
)

func init() {
	http.HandleFunc("/", uiHandler)
	html.Escape(uiTemplate)
}

// uiHandler draws the build status page.
func uiHandler(w http.ResponseWriter, r *http.Request) {
	// TODO(adg): put the HTML in memcache and invalidate on updates
	c := appengine.NewContext(r)

	page, _ := strconv.Atoi(r.FormValue("page"))
	if page < 0 {
		page = 0
	}

	commits, err := goCommits(c, page)
	if err != nil {
		logErr(w, r, err)
		return
	}
	builders := commitBuilders(commits)

	tipState, err := TagState(c, "tip")
	if err != nil {
		logErr(w, r, err)
		return
	}

	p := &Pagination{}
	if len(commits) == commitsPerPage {
		p.Next = page + 1
	}
	if page > 0 {
		p.Prev = page - 1
		p.HasPrev = true
	}
	data := &uiTemplateData{commits, builders, tipState, p}
	if err := uiTemplate.Execute(w, data); err != nil {
		logErr(w, r, err)
	}
}

type Pagination struct {
	Next, Prev int
	HasPrev    bool
}

// goCommits gets a slice of the latest Commits to the Go repository.
// If page > 0 it paginates by commitsPerPage.
func goCommits(c appengine.Context, page int) ([]*Commit, os.Error) {
	q := datastore.NewQuery("Commit").
		Ancestor((&Package{}).Key(c)).
		Order("-Time").
		Limit(commitsPerPage).
		Offset(page * commitsPerPage)
	var commits []*Commit
	_, err := q.GetAll(c, &commits)
	return commits, err
}

// commitBuilders returns the names of the builders that provided
// Results for the provided commits.
func commitBuilders(commits []*Commit) []string {
	builders := make(map[string]bool)
	for _, commit := range commits {
		for _, r := range commit.Results("") {
			builders[r.Builder] = true
		}
	}
	return keys(builders)
}

func keys(m map[string]bool) (s []string) {
	for k := range m {
		s = append(s, k)
	}
	sort.Strings(s)
	return
}

// PackageState represents the state of a Package at a tag.
type PackageState struct {
	*Package
	*Commit
	Results []*Result
	OK      bool
}

// TagState fetches the results for all non-Go packages at the specified tag.
func TagState(c appengine.Context, name string) ([]*PackageState, os.Error) {
	tag, err := GetTag(c, name)
	if err != nil {
		return nil, err
	}
	pkgs, err := Packages(c)
	if err != nil {
		return nil, err
	}
	var states []*PackageState
	for _, pkg := range pkgs {
		commit, err := pkg.LastCommit(c)
		if err != nil {
			c.Errorf("no Commit found: %v", pkg)
			continue
		}
		results := commit.Results(tag.Hash)
		ok := len(results) > 0
		for _, r := range results {
			ok = ok && r.OK
		}
		states = append(states, &PackageState{
			pkg, commit, results, ok,
		})
	}
	return states, nil
}

type uiTemplateData struct {
	Commits    []*Commit
	Builders   []string
	TipState   []*PackageState
	Pagination *Pagination
}

var uiTemplate = template.Must(
	template.New("ui").
		Funcs(template.FuncMap{
			"builderTitle": builderTitle,
			"shortHash":    shortHash,
			"repoURL":      repoURL,
		}).
		ParseFile("build/ui.html"),
)

// builderTitle formats "linux-amd64-foo" as "linux amd64 foo".
func builderTitle(s string) string {
	return strings.Replace(s, "-", " ", -1)
}

// shortHash returns a the short version of a hash.
func shortHash(hash string) string {
	if len(hash) > 12 {
		hash = hash[:12]
	}
	return hash
}

// repoRe matches Google Code repositories and subrepositories (without paths).
var repoRe = regexp.MustCompile(`^code\.google\.com/p/([a-z0-9\-]+)(\.[a-z0-9\-]+)?$`)

// repoURL returns the URL of a change at a Google Code repository or subrepo.
func repoURL(hash, packagePath string) (string, os.Error) {
	if packagePath == "" {
		return "https://code.google.com/p/go/source/detail?r=" + hash, nil
	}
	m := repoRe.FindStringSubmatch(packagePath)
	if m == nil {
		return "", os.NewError("unrecognized package: " + packagePath)
	}
	url := "https://code.google.com/p/" + m[1] + "/source/detail?r=" + hash
	if len(m) > 2 {
		url += "&repo=" + m[2][1:]
	}
	return url, nil
}
