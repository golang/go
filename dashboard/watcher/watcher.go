// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command watcher watches the specified repository for new commits
// and reports them to the build dashboard.
package main

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

var (
	repoURL      = flag.String("repo", "https://code.google.com/p/go", "Repository URL")
	dashboard    = flag.String("dash", "https://build.golang.org/", "Dashboard URL (must end in /)")
	keyFile      = flag.String("key", defaultKeyFile, "Build dashboard key file")
	pollInterval = flag.Duration("poll", 10*time.Second, "Remote repo poll interval")
)

var (
	defaultKeyFile = filepath.Join(homeDir(), ".gobuildkey")
	dashboardKey   = ""
)

// The first main repo commit on the dashboard; ignore commits before this.
// This is for the main Go repo only.
const dashboardStart = "2f970046e1ba96f32de62f5639b7141cda2e977c"

func main() {
	flag.Parse()

	err := run()
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

// run is a little wrapper so we can use defer and return to signal
// errors. It should only return a non-nil error.
func run() error {
	if !strings.HasSuffix(*dashboard, "/") {
		return errors.New("dashboard URL (-dashboard) must end in /")
	}
	if err := checkHgVersion(); err != nil {
		return err
	}

	if k, err := readKey(); err != nil {
		return err
	} else {
		dashboardKey = k
	}

	dir, err := ioutil.TempDir("", "watcher")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	errc := make(chan error)

	go func() {
		r, err := NewRepo(dir, *repoURL, "")
		if err != nil {
			errc <- err
			return
		}
		errc <- r.Watch()
	}()

	subrepos, err := subrepoList()
	if err != nil {
		return err
	}
	for _, path := range subrepos {
		go func(path string) {
			url := "https://" + path
			r, err := NewRepo(dir, url, path)
			if err != nil {
				errc <- err
				return
			}
			errc <- r.Watch()
		}(path)
	}

	// Must be non-nil.
	return <-errc
}

// Repo represents a repository to be watched.
type Repo struct {
	root     string             // on-disk location of the hg repo
	path     string             // base import path for repo (blank for main repo)
	commits  map[string]*Commit // keyed by full commit hash (40 lowercase hex digits)
	branches map[string]*Branch // keyed by branch name, eg "release-branch.go1.3" (or empty for default)
}

// NewRepo checks out a new instance of the Mercurial repository
// specified by url to a new directory inside dir.
// The path argument is the base import path of the repository,
// and should be empty for the main Go repo.
func NewRepo(dir, url, path string) (*Repo, error) {
	r := &Repo{
		path: path,
		root: filepath.Join(dir, filepath.Base(path)),
	}

	r.logf("cloning %v", url)
	cmd := exec.Command("hg", "clone", url, r.root)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("%v\n\n%s", err, out)
	}

	r.logf("loading commit log")
	if err := r.loadCommits(); err != nil {
		return nil, err
	}
	if err := r.findBranches(); err != nil {
		return nil, err
	}

	r.logf("found %v branches among %v commits\n", len(r.branches), len(r.commits))
	return r, nil
}

// Watch continuously runs "hg pull" in the repo, checks for
// new commits, and posts any new commits to the dashboard.
// It only returns a non-nil error.
func (r *Repo) Watch() error {
	for {
		if err := hgPull(r.root); err != nil {
			return err
		}
		if err := r.update(); err != nil {
			return err
		}
		for _, b := range r.branches {
			if err := r.postNewCommits(b); err != nil {
				return err
			}
		}
		time.Sleep(*pollInterval)
	}
}

func (r *Repo) logf(format string, args ...interface{}) {
	p := "go"
	if r.path != "" {
		p = path.Base(r.path)
	}
	log.Printf(p+": "+format, args...)
}

// postNewCommits looks for unseen commits on the specified branch and
// posts them to the dashboard.
func (r *Repo) postNewCommits(b *Branch) error {
	if b.Head == b.LastSeen {
		return nil
	}
	c := b.LastSeen
	if c == nil {
		// Haven't seen any: find the commit that this branch forked from.
		for c := b.Head; c.Branch == b.Name; c = c.parent {
		}
	}
	// Add unseen commits on this branch, working forward from last seen.
	for c.children != nil {
		// Find the next commit on this branch.
		var next *Commit
		for _, c2 := range c.children {
			if c2.Branch != b.Name {
				continue
			}
			if next != nil {
				// Shouldn't happen, but be paranoid.
				return fmt.Errorf("found multiple children of %v on branch %q: %v and %v", c, b.Name, next, c2)
			}
			next = c2
		}
		if next == nil {
			// No more children on this branch, bail.
			break
		}
		// Found it.
		c = next

		if err := r.postCommit(c); err != nil {
			return err
		}
		b.LastSeen = c
	}
	return nil
}

// postCommit sends a commit to the build dashboard.
func (r *Repo) postCommit(c *Commit) error {
	r.logf("sending commit to dashboard: %v", c)

	t, err := time.Parse(time.RFC3339, c.Date)
	if err != nil {
		return err
	}
	dc := struct {
		PackagePath string // (empty for main repo commits)
		Hash        string
		ParentHash  string

		User string
		Desc string
		Time time.Time

		NeedsBenchmarking bool
	}{
		PackagePath: r.path,
		Hash:        c.Hash,
		ParentHash:  c.Parent,

		User: c.Author,
		Desc: c.Desc,
		Time: t,

		NeedsBenchmarking: c.NeedsBenchmarking(),
	}
	b, err := json.Marshal(dc)
	if err != nil {
		return err
	}

	u := *dashboard + "commit?version=2&key=" + dashboardKey
	resp, err := http.Post(u, "text/json", bytes.NewReader(b))
	if err != nil {
		return err
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("status: %v", resp.Status)
	}
	return nil
}

// loadCommits runs "hg log" and populates the Repo's commit map.
func (r *Repo) loadCommits() error {
	log, err := hgLog(r.root)
	if err != nil {
		return err
	}
	r.commits = make(map[string]*Commit)
	for _, c := range log {
		r.commits[c.Hash] = c
	}
	for _, c := range r.commits {
		if p, ok := r.commits[c.Parent]; ok {
			c.parent = p
			p.children = append(p.children, c)
		}
	}
	return nil
}

// findBranches finds branch heads in the Repo's commit map
// and populates its branch map.
func (r *Repo) findBranches() error {
	r.branches = make(map[string]*Branch)
	for _, c := range r.commits {
		if c.children == nil {
			if !validHead(c) {
				continue
			}
			seen, err := r.lastSeen(c.Hash)
			if err != nil {
				return err
			}
			b := &Branch{Name: c.Branch, Head: c, LastSeen: seen}
			r.branches[c.Branch] = b
			r.logf("found branch: %v", b)
		}
	}
	return nil
}

// validHead reports whether the specified commit should be considered a branch
// head. It considers pre-go1 branches and certain specific commits as invalid.
func validHead(c *Commit) bool {
	// Pre Go-1 releases branches are irrelevant.
	if strings.HasPrefix(c.Branch, "release-branch.r") {
		return false
	}
	// Not sure why these revisions have no child commits,
	// but they're old so let's just ignore them.
	if c.Hash == "b59f4ff1b51094314f735a4d57a2b8f06cfadf15" ||
		c.Hash == "fc75f13840b896e82b9fa6165cf705fbacaf019c" {
		return false
	}
	// All other branches are valid.
	return true
}

// update runs "hg pull" in the specified reporoot,
// looks for new commits and branches,
// and updates the comits and branches maps.
func (r *Repo) update() error {
	// TODO(adg): detect new branches with "hg branches".

	// Check each branch for new commits.
	for _, b := range r.branches {

		// Find all commits on this branch from known head.
		// The logic of this function assumes that "hg log $HASH:"
		// returns hashes in the order they were committed (parent first).
		bname := b.Name
		if bname == "" {
			bname = "default"
		}
		log, err := hgLog(r.root, "-r", b.Head.Hash+":", "-b", bname)
		if err != nil {
			return err
		}

		// Add unknown commits to r.commits, and update branch head.
		for _, c := range log {
			// Ignore if we already know this commit.
			if _, ok := r.commits[c.Hash]; ok {
				continue
			}
			r.logf("found new commit %v", c)

			// Sanity check that we're looking at a commit on this branch.
			if c.Branch != b.Name {
				return fmt.Errorf("hg log gave us a commit from wrong branch: want %q, got %q", b.Name, c.Branch)
			}

			// Find parent commit.
			p, ok := r.commits[c.Parent]
			if !ok {
				return fmt.Errorf("can't find parent hash %q for %v", c.Parent, c)
			}

			// Link parent and child Commits.
			c.parent = p
			p.children = append(p.children, c)

			// Update branch head.
			b.Head = c

			// Add new commit to map.
			r.commits[c.Hash] = c
		}
	}

	return nil
}

// lastSeen finds the most recent commit the dashboard has seen,
// starting at the specified head. If the dashboard hasn't seen
// any of the commits from head to the beginning, it returns nil.
func (r *Repo) lastSeen(head string) (*Commit, error) {
	h, ok := r.commits[head]
	if !ok {
		return nil, fmt.Errorf("lastSeen: can't find %q in commits", head)
	}

	var s []*Commit
	for c := h; c != nil; c = c.parent {
		s = append(s, c)
		if r.path == "" && c.Hash == dashboardStart {
			break
		}
	}

	for _, c := range s {
		v := url.Values{"hash": {c.Hash}, "packagePath": {r.path}}
		u := *dashboard + "commit?" + v.Encode()
		r, err := http.Get(u)
		if err != nil {
			return nil, err
		}
		var resp struct {
			Error string
		}
		err = json.NewDecoder(r.Body).Decode(&resp)
		r.Body.Close()
		if err != nil {
			return nil, err
		}
		switch resp.Error {
		case "":
			// Found one.
			return c, nil
		case "Commit not found":
			// Commit not found, keep looking for earlier commits.
			continue
		default:
			return nil, fmt.Errorf("dashboard: %v", resp.Error)
		}
	}

	// Dashboard saw no commits.
	return nil, nil
}

// hgLog runs "hg log" with the supplied arguments
// and parses the output into Commit values.
func hgLog(dir string, args ...string) ([]*Commit, error) {
	args = append([]string{"log", "--template", xmlLogTemplate}, args...)
	cmd := exec.Command("hg", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, err
	}

	// We have a commit with description that contains 0x1b byte.
	// Mercurial does not escape it, but xml.Unmarshal does not accept it.
	out = bytes.Replace(out, []byte{0x1b}, []byte{'?'}, -1)

	xr := io.MultiReader(
		strings.NewReader("<Top>"),
		bytes.NewReader(out),
		strings.NewReader("</Top>"),
	)
	var logStruct struct {
		Log []*Commit
	}
	err = xml.NewDecoder(xr).Decode(&logStruct)
	if err != nil {
		return nil, err
	}
	return logStruct.Log, nil
}

// hgPull runs "hg pull" in the specified directory.
// It tries three times, just in case it failed because of a transient error.
func hgPull(dir string) error {
	var err error
	for tries := 0; tries < 3; tries++ {
		time.Sleep(time.Duration(tries) * 5 * time.Second) // Linear back-off.
		cmd := exec.Command("hg", "pull")
		cmd.Dir = dir
		if out, e := cmd.CombinedOutput(); err != nil {
			e = fmt.Errorf("%v\n\n%s", e, out)
			log.Printf("hg pull error %v: %v", dir, e)
			if err == nil {
				err = e
			}
			continue
		}
		return nil
	}
	return err
}

// Branch represents a Mercurial branch.
type Branch struct {
	Name     string
	Head     *Commit
	LastSeen *Commit // the last commit posted to the dashboard
}

func (b *Branch) String() string {
	return fmt.Sprintf("%q(Head: %v LastSeen: %v)", b.Name, b.Head, b.LastSeen)
}

// Commit represents a single Mercurial revision.
type Commit struct {
	Hash   string
	Author string
	Date   string
	Desc   string // Plain text, first linefeed-terminated line is a short description.
	Parent string
	Branch string
	Files  string

	// For walking the graph.
	parent   *Commit
	children []*Commit
}

func (c *Commit) String() string {
	return fmt.Sprintf("%v(%q)", c.Hash, strings.SplitN(c.Desc, "\n", 2)[0])
}

// NeedsBenchmarking reports whether the Commit needs benchmarking.
func (c *Commit) NeedsBenchmarking() bool {
	// Do not benchmark branch commits, they are usually not interesting
	// and fall out of the trunk succession.
	if c.Branch != "" {
		return false
	}
	// Do not benchmark commits that do not touch source files (e.g. CONTRIBUTORS).
	for _, f := range strings.Split(c.Files, " ") {
		if (strings.HasPrefix(f, "include") || strings.HasPrefix(f, "src")) &&
			!strings.HasSuffix(f, "_test.go") && !strings.Contains(f, "testdata") {
			return true
		}
	}
	return false
}

// xmlLogTemplate is a template to pass to Mercurial to make
// hg log print the log in valid XML for parsing with xml.Unmarshal.
// Can not escape branches and files, because it crashes python with:
// AttributeError: 'NoneType' object has no attribute 'replace'
const xmlLogTemplate = `
        <Log>
        <Hash>{node|escape}</Hash>
        <Parent>{p1node}</Parent>
        <Author>{author|escape}</Author>
        <Date>{date|rfc3339date}</Date>
        <Desc>{desc|escape}</Desc>
        <Branch>{branches}</Branch>
        <Files>{files}</Files>
        </Log>
`

func homeDir() string {
	switch runtime.GOOS {
	case "plan9":
		return os.Getenv("home")
	case "windows":
		return os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
	}
	return os.Getenv("HOME")
}

func readKey() (string, error) {
	c, err := ioutil.ReadFile(*keyFile)
	if err != nil {
		return "", err
	}
	return string(bytes.TrimSpace(bytes.SplitN(c, []byte("\n"), 2)[0])), nil
}

// subrepoList fetches a list of sub-repositories from the dashboard
// and returns them as a slice of base import paths.
// Eg, []string{"golang.org/x/tools", "golang.org/x/net"}.
func subrepoList() ([]string, error) {
	r, err := http.Get(*dashboard + "packages?kind=subrepo")
	if err != nil {
		return nil, err
	}
	var resp struct {
		Response []struct {
			Path string
		}
		Error string
	}
	err = json.NewDecoder(r.Body).Decode(&resp)
	r.Body.Close()
	if err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, errors.New(resp.Error)
	}
	var pkgs []string
	for _, r := range resp.Response {
		pkgs = append(pkgs, r.Path)
	}
	return pkgs, nil
}

// checkHgVersion checks whether the installed version of hg supports the
// template features we need. (May not be precise.)
func checkHgVersion() error {
	out, err := exec.Command("hg", "help", "templates").CombinedOutput()
	if err != nil {
		return fmt.Errorf("error running hg help templates: %v\n\n%s", err, out)
	}
	if !bytes.Contains(out, []byte("p1node")) {
		return errors.New("installed hg doesn't support 'p1node' template keyword; please upgrade")
	}
	return nil
}
