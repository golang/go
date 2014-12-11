// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command watcher watches the specified repository for new commits
// and reports them to the build dashboard.
package main // import "golang.org/x/tools/dashboard/watcher"

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const (
	goBase         = "https://go.googlesource.com/"
	watcherVersion = 3 // must match dashboard/app/build/handler.go's watcherVersion
	origin         = "origin/"
	master         = origin + "master" // name of the master branch
)

var (
	repoURL      = flag.String("repo", goBase+"go", "Repository URL")
	dashboard    = flag.String("dash", "https://build.golang.org/", "Dashboard URL (must end in /)")
	keyFile      = flag.String("key", defaultKeyFile, "Build dashboard key file")
	pollInterval = flag.Duration("poll", 10*time.Second, "Remote repo poll interval")
	network      = flag.Bool("network", true, "Enable network calls (disable for testing)")
)

var (
	defaultKeyFile = filepath.Join(homeDir(), ".gobuildkey")
	dashboardKey   = ""
	networkSeen    = make(map[string]bool) // track known hashes for testing
)

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
			url := goBase + strings.TrimPrefix(path, "golang.org/x/")
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
	root     string             // on-disk location of the git repo
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
		path:     path,
		root:     filepath.Join(dir, filepath.Base(path)),
		commits:  make(map[string]*Commit),
		branches: make(map[string]*Branch),
	}

	r.logf("cloning %v", url)
	cmd := exec.Command("git", "clone", url, r.root)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("%v\n\n%s", err, out)
	}

	r.logf("loading commit log")
	if err := r.update(false); err != nil {
		return nil, err
	}

	r.logf("found %v branches among %v commits\n", len(r.branches), len(r.commits))
	return r, nil
}

// Watch continuously runs "git fetch" in the repo, checks for
// new commits, and posts any new commits to the dashboard.
// It only returns a non-nil error.
func (r *Repo) Watch() error {
	for {
		if err := r.fetch(); err != nil {
			return err
		}
		if err := r.update(true); err != nil {
			return err
		}
		remotes, err := r.remotes()
		if err != nil {
			return err
		}
		for _, name := range remotes {
			b, ok := r.branches[name]
			if !ok {
				// skip branch; must be already merged
				continue
			}
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
		// Haven't seen anything on this branch yet:
		if b.Name == master {
			// For the master branch, bootstrap by creating a dummy
			// commit with a lone child that is the initial commit.
			c = &Commit{}
			for _, c2 := range r.commits {
				if c2.Parent == "" {
					c.children = []*Commit{c2}
					break
				}
			}
			if c.children == nil {
				return fmt.Errorf("couldn't find initial commit")
			}
		} else {
			// Find the commit that this branch forked from.
			base, err := r.mergeBase(b.Name, master)
			if err != nil {
				return err
			}
			var ok bool
			c, ok = r.commits[base]
			if !ok {
				return fmt.Errorf("couldn't find base commit: %v", base)
			}
		}
	}
	if err := r.postChildren(b, c); err != nil {
		return err
	}
	b.LastSeen = b.Head
	return nil
}

// postChildren posts to the dashboard all descendants of the given parent.
// It ignores descendants that are not on the given branch.
func (r *Repo) postChildren(b *Branch, parent *Commit) error {
	for _, c := range parent.children {
		if c.Branch != b.Name {
			continue
		}
		if err := r.postCommit(c); err != nil {
			return err
		}
	}
	for _, c := range parent.children {
		if err := r.postChildren(b, c); err != nil {
			return err
		}
	}
	return nil
}

// postCommit sends a commit to the build dashboard.
func (r *Repo) postCommit(c *Commit) error {
	r.logf("sending commit to dashboard: %v", c)

	t, err := time.Parse("Mon, 2 Jan 2006 15:04:05 -0700", c.Date)
	if err != nil {
		return fmt.Errorf("postCommit: parsing date %q for commit %v: %v", c.Date, c, err)
	}
	dc := struct {
		PackagePath string // (empty for main repo commits)
		Hash        string
		ParentHash  string

		User   string
		Desc   string
		Time   time.Time
		Branch string

		NeedsBenchmarking bool
	}{
		PackagePath: r.path,
		Hash:        c.Hash,
		ParentHash:  c.Parent,

		User:   c.Author,
		Desc:   c.Desc,
		Time:   t,
		Branch: strings.TrimPrefix(c.Branch, origin),

		NeedsBenchmarking: c.NeedsBenchmarking(),
	}
	b, err := json.Marshal(dc)
	if err != nil {
		return fmt.Errorf("postCommit: marshaling request body: %v", err)
	}

	if !*network {
		if c.Parent != "" {
			if !networkSeen[c.Parent] {
				r.logf("%v: %v", c.Parent, r.commits[c.Parent])
				return fmt.Errorf("postCommit: no parent %v found on dashboard for %v", c.Parent, c)
			}
		}
		if networkSeen[c.Hash] {
			return fmt.Errorf("postCommit: already seen %v", c)
		}
		networkSeen[c.Hash] = true
		return nil
	}

	u := fmt.Sprintf("%vcommit?version=%v&key=%v", *dashboard, watcherVersion, dashboardKey)
	resp, err := http.Post(u, "text/json", bytes.NewReader(b))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("postCommit: status: %v", resp.Status)
	}

	var s struct {
		Error string
	}
	err = json.NewDecoder(resp.Body).Decode(&s)
	if err != nil {
		return fmt.Errorf("postCommit: decoding response: %v", err)
	}
	if s.Error != "" {
		return fmt.Errorf("postCommit: error: %v", s.Error)
	}
	return nil
}

// update looks for new commits and branches,
// and updates the commits and branches maps.
func (r *Repo) update(noisy bool) error {
	remotes, err := r.remotes()
	if err != nil {
		return err
	}
	for _, name := range remotes {
		b := r.branches[name]

		// Find all unseen commits on this branch.
		revspec := name
		if b != nil {
			// If we know about this branch,
			// only log commits down to the known head.
			revspec = b.Head.Hash + ".." + name
		} else if revspec != master {
			// If this is an unknown non-master branch,
			// log up to where it forked from master.
			base, err := r.mergeBase(name, master)
			if err != nil {
				return err
			}
			revspec = base + ".." + name
		}
		log, err := r.log("--topo-order", revspec)
		if err != nil {
			return err
		}
		if len(log) == 0 {
			// No commits to handle; carry on.
			continue
		}

		// Add unknown commits to r.commits.
		var added []*Commit
		for _, c := range log {
			// Sanity check: we shouldn't see the same commit twice.
			if _, ok := r.commits[c.Hash]; ok {
				return fmt.Errorf("found commit we already knew about: %v", c.Hash)
			}
			if noisy {
				r.logf("found new commit %v", c)
			}
			c.Branch = name
			r.commits[c.Hash] = c
			added = append(added, c)
		}

		// Link added commits.
		for _, c := range added {
			if c.Parent == "" {
				// This is the initial commit; no parent.
				r.logf("no parents for initial commit %v", c)
				continue
			}
			// Find parent commit.
			p, ok := r.commits[c.Parent]
			if !ok {
				return fmt.Errorf("can't find parent %q for %v", c.Parent, c)
			}
			// Link parent Commit.
			c.parent = p
			// Link child Commits.
			p.children = append(p.children, c)
		}

		// Update branch head, or add newly discovered branch.
		head := log[0]
		if b != nil {
			// Known branch; update head.
			b.Head = head
			r.logf("updated branch head: %v", b)
		} else {
			// It's a new branch; add it.
			seen, err := r.lastSeen(head.Hash)
			if err != nil {
				return err
			}
			b = &Branch{Name: name, Head: head, LastSeen: seen}
			r.branches[name] = b
			r.logf("found branch: %v", b)
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
	}

	var err error
	i := sort.Search(len(s), func(i int) bool {
		if err != nil {
			return false
		}
		ok, err = r.dashSeen(s[i].Hash)
		return ok
	})
	switch {
	case err != nil:
		return nil, fmt.Errorf("lastSeen: %v", err)
	case i < len(s):
		return s[i], nil
	default:
		// Dashboard saw no commits.
		return nil, nil
	}
}

// dashSeen reports whether the build dashboard knows the specified commit.
func (r *Repo) dashSeen(hash string) (bool, error) {
	if !*network {
		return networkSeen[hash], nil
	}
	v := url.Values{"hash": {hash}, "packagePath": {r.path}}
	u := *dashboard + "commit?" + v.Encode()
	resp, err := http.Get(u)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return false, fmt.Errorf("status: %v", resp.Status)
	}
	var s struct {
		Error string
	}
	err = json.NewDecoder(resp.Body).Decode(&s)
	if err != nil {
		return false, err
	}
	switch s.Error {
	case "":
		// Found one.
		return true, nil
	case "Commit not found":
		// Commit not found, keep looking for earlier commits.
		return false, nil
	default:
		return false, fmt.Errorf("dashboard: %v", s.Error)
	}
}

// mergeBase returns the hash of the merge base for revspecs a and b.
func (r *Repo) mergeBase(a, b string) (string, error) {
	cmd := exec.Command("git", "merge-base", a, b)
	cmd.Dir = r.root
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git merge-base: %v", err)
	}
	return string(bytes.TrimSpace(out)), nil
}

// remotes returns a slice of remote branches known to the git repo.
// It always puts "origin/master" first.
func (r *Repo) remotes() ([]string, error) {
	cmd := exec.Command("git", "branch", "-r")
	cmd.Dir = r.root
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("git branch: %v", err)
	}
	bs := []string{master}
	for _, b := range strings.Split(string(out), "\n") {
		b = strings.TrimSpace(b)
		// Ignore aliases, blank lines, and master (it's already in bs).
		if b == "" || strings.Contains(b, "->") || b == master {
			continue
		}
		// Ignore pre-go1 release branches; they are just noise.
		if strings.HasPrefix(b, origin+"release-branch.r") {
			continue
		}
		bs = append(bs, b)
	}
	return bs, nil
}

const logFormat = `--format=format:%H
%P
%an <%ae>
%cD
%B
` + logBoundary

const logBoundary = `_-_- magic boundary -_-_`

// log runs "git log" with the supplied arguments
// and parses the output into Commit values.
func (r *Repo) log(dir string, args ...string) ([]*Commit, error) {
	args = append([]string{"log", "--date=rfc", logFormat}, args...)
	cmd := exec.Command("git", args...)
	cmd.Dir = r.root
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("git log %v: %v", strings.Join(args, " "), err)
	}

	// We have a commit with description that contains 0x1b byte.
	// Mercurial does not escape it, but xml.Unmarshal does not accept it.
	// TODO(adg): do we still need to scrub this? Probably.
	out = bytes.Replace(out, []byte{0x1b}, []byte{'?'}, -1)

	var cs []*Commit
	for _, text := range strings.Split(string(out), logBoundary) {
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		p := strings.SplitN(text, "\n", 5)
		if len(p) != 5 {
			return nil, fmt.Errorf("git log %v: malformed commit: %q", strings.Join(args, " "), text)
		}
		cs = append(cs, &Commit{
			Hash: p[0],
			// TODO(adg): This may break with branch merges.
			Parent: strings.Split(p[1], " ")[0],
			Author: p[2],
			Date:   p[3],
			Desc:   strings.TrimSpace(p[4]),
			// TODO(adg): populate Files
		})
	}
	return cs, nil
}

// fetch runs "git fetch" in the repository root.
// It tries three times, just in case it failed because of a transient error.
func (r *Repo) fetch() error {
	var err error
	for tries := 0; tries < 3; tries++ {
		time.Sleep(time.Duration(tries) * 5 * time.Second) // Linear back-off.
		cmd := exec.Command("git", "fetch", "--all")
		cmd.Dir = r.root
		if out, e := cmd.CombinedOutput(); err != nil {
			e = fmt.Errorf("%v\n\n%s", e, out)
			log.Printf("git fetch error %v: %v", r.root, e)
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

// Commit represents a single Git commit.
type Commit struct {
	Hash   string
	Author string
	Date   string // Format: "Mon, 2 Jan 2006 15:04:05 -0700"
	Desc   string // Plain text, first line is a short description.
	Parent string
	Branch string
	Files  string

	// For walking the graph.
	parent   *Commit
	children []*Commit
}

func (c *Commit) String() string {
	s := c.Hash
	if c.Branch != "" {
		s += fmt.Sprintf("[%v]", strings.TrimPrefix(c.Branch, origin))
	}
	s += fmt.Sprintf("(%q)", strings.SplitN(c.Desc, "\n", 2)[0])
	return s
}

// NeedsBenchmarking reports whether the Commit needs benchmarking.
func (c *Commit) NeedsBenchmarking() bool {
	// Do not benchmark branch commits, they are usually not interesting
	// and fall out of the trunk succession.
	if c.Branch != master {
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
	if !*network {
		return nil, nil
	}

	r, err := http.Get(*dashboard + "packages?kind=subrepo")
	if err != nil {
		return nil, fmt.Errorf("subrepo list: %v", err)
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		return nil, fmt.Errorf("subrepo list: got status %v", r.Status)
	}
	var resp struct {
		Response []struct {
			Path string
		}
		Error string
	}
	err = json.NewDecoder(r.Body).Decode(&resp)
	if err != nil {
		return nil, fmt.Errorf("subrepo list: %v", err)
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("subrepo list: %v", resp.Error)
	}
	var pkgs []string
	for _, r := range resp.Response {
		pkgs = append(pkgs, r.Path)
	}
	return pkgs, nil
}
