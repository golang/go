// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	exec "internal/execabs"
	"internal/lazyregexp"
	"internal/singleflight"
	"io/fs"
	"log"
	urlpkg "net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/go/internal/web"

	"golang.org/x/mod/module"
)

// A Cmd describes how to use a version control system
// like Mercurial, Git, or Subversion.
type Cmd struct {
	Name      string
	Cmd       string   // name of binary to invoke command
	RootNames []string // filename indicating the root of a checkout directory

	CreateCmd   []string // commands to download a fresh copy of a repository
	DownloadCmd []string // commands to download updates into an existing repository

	TagCmd         []tagCmd // commands to list tags
	TagLookupCmd   []tagCmd // commands to lookup tags before running tagSyncCmd
	TagSyncCmd     []string // commands to sync to specific tag
	TagSyncDefault []string // commands to sync to default tag

	Scheme  []string
	PingCmd string

	RemoteRepo  func(v *Cmd, rootDir string) (remoteRepo string, err error)
	ResolveRepo func(v *Cmd, rootDir, remoteRepo string) (realRepo string, err error)
	Status      func(v *Cmd, rootDir string) (Status, error)
}

// Status is the current state of a local repository.
type Status struct {
	Revision    string    // Optional.
	CommitTime  time.Time // Optional.
	Uncommitted bool      // Required.
}

var defaultSecureScheme = map[string]bool{
	"https":   true,
	"git+ssh": true,
	"bzr+ssh": true,
	"svn+ssh": true,
	"ssh":     true,
}

func (v *Cmd) IsSecure(repo string) bool {
	u, err := urlpkg.Parse(repo)
	if err != nil {
		// If repo is not a URL, it's not secure.
		return false
	}
	return v.isSecureScheme(u.Scheme)
}

func (v *Cmd) isSecureScheme(scheme string) bool {
	switch v.Cmd {
	case "git":
		// GIT_ALLOW_PROTOCOL is an environment variable defined by Git. It is a
		// colon-separated list of schemes that are allowed to be used with git
		// fetch/clone. Any scheme not mentioned will be considered insecure.
		if allow := os.Getenv("GIT_ALLOW_PROTOCOL"); allow != "" {
			for _, s := range strings.Split(allow, ":") {
				if s == scheme {
					return true
				}
			}
			return false
		}
	}
	return defaultSecureScheme[scheme]
}

// A tagCmd describes a command to list available tags
// that can be passed to tagSyncCmd.
type tagCmd struct {
	cmd     string // command to list tags
	pattern string // regexp to extract tags from list
}

// vcsList lists the known version control systems
var vcsList = []*Cmd{
	vcsHg,
	vcsGit,
	vcsSvn,
	vcsBzr,
	vcsFossil,
}

// vcsMod is a stub for the "mod" scheme. It's returned by
// repoRootForImportPathDynamic, but is otherwise not treated as a VCS command.
var vcsMod = &Cmd{Name: "mod"}

// vcsByCmd returns the version control system for the given
// command name (hg, git, svn, bzr).
func vcsByCmd(cmd string) *Cmd {
	for _, vcs := range vcsList {
		if vcs.Cmd == cmd {
			return vcs
		}
	}
	return nil
}

// vcsHg describes how to use Mercurial.
var vcsHg = &Cmd{
	Name:      "Mercurial",
	Cmd:       "hg",
	RootNames: []string{".hg"},

	CreateCmd:   []string{"clone -U -- {repo} {dir}"},
	DownloadCmd: []string{"pull"},

	// We allow both tag and branch names as 'tags'
	// for selecting a version. This lets people have
	// a go.release.r60 branch and a go1 branch
	// and make changes in both, without constantly
	// editing .hgtags.
	TagCmd: []tagCmd{
		{"tags", `^(\S+)`},
		{"branches", `^(\S+)`},
	},
	TagSyncCmd:     []string{"update -r {tag}"},
	TagSyncDefault: []string{"update default"},

	Scheme:     []string{"https", "http", "ssh"},
	PingCmd:    "identify -- {scheme}://{repo}",
	RemoteRepo: hgRemoteRepo,
	Status:     hgStatus,
}

func hgRemoteRepo(vcsHg *Cmd, rootDir string) (remoteRepo string, err error) {
	out, err := vcsHg.runOutput(rootDir, "paths default")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func hgStatus(vcsHg *Cmd, rootDir string) (Status, error) {
	// Output changeset ID and seconds since epoch.
	out, err := vcsHg.runOutputVerboseOnly(rootDir, `log -l1 -T {node}:{date(date,"%s")}`)
	if err != nil {
		return Status{}, err
	}

	// Successful execution without output indicates an empty repo (no commits).
	var rev string
	var commitTime time.Time
	if len(out) > 0 {
		rev, commitTime, err = parseRevTime(out)
		if err != nil {
			return Status{}, err
		}
	}

	// Also look for untracked files.
	out, err = vcsHg.runOutputVerboseOnly(rootDir, "status")
	if err != nil {
		return Status{}, err
	}
	uncommitted := len(out) > 0

	return Status{
		Revision:    rev,
		CommitTime:  commitTime,
		Uncommitted: uncommitted,
	}, nil
}

// parseRevTime parses commit details in "revision:seconds" format.
func parseRevTime(out []byte) (string, time.Time, error) {
	buf := string(bytes.TrimSpace(out))

	i := strings.IndexByte(buf, ':')
	if i < 1 {
		return "", time.Time{}, errors.New("unrecognized VCS tool output")
	}
	rev := buf[:i]

	secs, err := strconv.ParseInt(string(buf[i+1:]), 10, 64)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("unrecognized VCS tool output: %v", err)
	}

	return rev, time.Unix(secs, 0), nil
}

// vcsGit describes how to use Git.
var vcsGit = &Cmd{
	Name:      "Git",
	Cmd:       "git",
	RootNames: []string{".git"},

	CreateCmd:   []string{"clone -- {repo} {dir}", "-go-internal-cd {dir} submodule update --init --recursive"},
	DownloadCmd: []string{"pull --ff-only", "submodule update --init --recursive"},

	TagCmd: []tagCmd{
		// tags/xxx matches a git tag named xxx
		// origin/xxx matches a git branch named xxx on the default remote repository
		{"show-ref", `(?:tags|origin)/(\S+)$`},
	},
	TagLookupCmd: []tagCmd{
		{"show-ref tags/{tag} origin/{tag}", `((?:tags|origin)/\S+)$`},
	},
	TagSyncCmd: []string{"checkout {tag}", "submodule update --init --recursive"},
	// both createCmd and downloadCmd update the working dir.
	// No need to do more here. We used to 'checkout master'
	// but that doesn't work if the default branch is not named master.
	// DO NOT add 'checkout master' here.
	// See golang.org/issue/9032.
	TagSyncDefault: []string{"submodule update --init --recursive"},

	Scheme: []string{"git", "https", "http", "git+ssh", "ssh"},

	// Leave out the '--' separator in the ls-remote command: git 2.7.4 does not
	// support such a separator for that command, and this use should be safe
	// without it because the {scheme} value comes from the predefined list above.
	// See golang.org/issue/33836.
	PingCmd: "ls-remote {scheme}://{repo}",

	RemoteRepo: gitRemoteRepo,
	Status:     gitStatus,
}

// scpSyntaxRe matches the SCP-like addresses used by Git to access
// repositories by SSH.
var scpSyntaxRe = lazyregexp.New(`^([a-zA-Z0-9_]+)@([a-zA-Z0-9._-]+):(.*)$`)

func gitRemoteRepo(vcsGit *Cmd, rootDir string) (remoteRepo string, err error) {
	cmd := "config remote.origin.url"
	errParse := errors.New("unable to parse output of git " + cmd)
	errRemoteOriginNotFound := errors.New("remote origin not found")
	outb, err := vcsGit.run1(rootDir, cmd, nil, false)
	if err != nil {
		// if it doesn't output any message, it means the config argument is correct,
		// but the config value itself doesn't exist
		if outb != nil && len(outb) == 0 {
			return "", errRemoteOriginNotFound
		}
		return "", err
	}
	out := strings.TrimSpace(string(outb))

	var repoURL *urlpkg.URL
	if m := scpSyntaxRe.FindStringSubmatch(out); m != nil {
		// Match SCP-like syntax and convert it to a URL.
		// Eg, "git@github.com:user/repo" becomes
		// "ssh://git@github.com/user/repo".
		repoURL = &urlpkg.URL{
			Scheme: "ssh",
			User:   urlpkg.User(m[1]),
			Host:   m[2],
			Path:   m[3],
		}
	} else {
		repoURL, err = urlpkg.Parse(out)
		if err != nil {
			return "", err
		}
	}

	// Iterate over insecure schemes too, because this function simply
	// reports the state of the repo. If we can't see insecure schemes then
	// we can't report the actual repo URL.
	for _, s := range vcsGit.Scheme {
		if repoURL.Scheme == s {
			return repoURL.String(), nil
		}
	}
	return "", errParse
}

func gitStatus(vcsGit *Cmd, rootDir string) (Status, error) {
	out, err := vcsGit.runOutputVerboseOnly(rootDir, "status --porcelain")
	if err != nil {
		return Status{}, err
	}
	uncommitted := len(out) > 0

	// "git status" works for empty repositories, but "git show" does not.
	// Assume there are no commits in the repo when "git show" fails with
	// uncommitted files and skip tagging revision / committime.
	var rev string
	var commitTime time.Time
	out, err = vcsGit.runOutputVerboseOnly(rootDir, "show -s --format=%H:%ct")
	if err != nil && !uncommitted {
		return Status{}, err
	} else if err == nil {
		rev, commitTime, err = parseRevTime(out)
		if err != nil {
			return Status{}, err
		}
	}

	return Status{
		Revision:    rev,
		CommitTime:  commitTime,
		Uncommitted: uncommitted,
	}, nil
}

// vcsBzr describes how to use Bazaar.
var vcsBzr = &Cmd{
	Name:      "Bazaar",
	Cmd:       "bzr",
	RootNames: []string{".bzr"},

	CreateCmd: []string{"branch -- {repo} {dir}"},

	// Without --overwrite bzr will not pull tags that changed.
	// Replace by --overwrite-tags after http://pad.lv/681792 goes in.
	DownloadCmd: []string{"pull --overwrite"},

	TagCmd:         []tagCmd{{"tags", `^(\S+)`}},
	TagSyncCmd:     []string{"update -r {tag}"},
	TagSyncDefault: []string{"update -r revno:-1"},

	Scheme:      []string{"https", "http", "bzr", "bzr+ssh"},
	PingCmd:     "info -- {scheme}://{repo}",
	RemoteRepo:  bzrRemoteRepo,
	ResolveRepo: bzrResolveRepo,
	Status:      bzrStatus,
}

func bzrRemoteRepo(vcsBzr *Cmd, rootDir string) (remoteRepo string, err error) {
	outb, err := vcsBzr.runOutput(rootDir, "config parent_location")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(outb)), nil
}

func bzrResolveRepo(vcsBzr *Cmd, rootDir, remoteRepo string) (realRepo string, err error) {
	outb, err := vcsBzr.runOutput(rootDir, "info "+remoteRepo)
	if err != nil {
		return "", err
	}
	out := string(outb)

	// Expect:
	// ...
	//   (branch root|repository branch): <URL>
	// ...

	found := false
	for _, prefix := range []string{"\n  branch root: ", "\n  repository branch: "} {
		i := strings.Index(out, prefix)
		if i >= 0 {
			out = out[i+len(prefix):]
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("unable to parse output of bzr info")
	}

	i := strings.Index(out, "\n")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of bzr info")
	}
	out = out[:i]
	return strings.TrimSpace(out), nil
}

func bzrStatus(vcsBzr *Cmd, rootDir string) (Status, error) {
	outb, err := vcsBzr.runOutputVerboseOnly(rootDir, "version-info")
	if err != nil {
		return Status{}, err
	}
	out := string(outb)

	// Expect (non-empty repositories only):
	//
	// revision-id: gopher@gopher.net-20211021072330-qshok76wfypw9lpm
	// date: 2021-09-21 12:00:00 +1000
	// ...
	var rev string
	var commitTime time.Time

	for _, line := range strings.Split(out, "\n") {
		i := strings.IndexByte(line, ':')
		if i < 0 {
			continue
		}
		key := line[:i]
		value := strings.TrimSpace(line[i+1:])

		switch key {
		case "revision-id":
			rev = value
		case "date":
			var err error
			commitTime, err = time.Parse("2006-01-02 15:04:05 -0700", value)
			if err != nil {
				return Status{}, errors.New("unable to parse output of bzr version-info")
			}
		}
	}

	outb, err = vcsBzr.runOutputVerboseOnly(rootDir, "status")
	if err != nil {
		return Status{}, err
	}

	// Skip warning when working directory is set to an older revision.
	if bytes.HasPrefix(outb, []byte("working tree is out of date")) {
		i := bytes.IndexByte(outb, '\n')
		if i < 0 {
			i = len(outb)
		}
		outb = outb[:i]
	}
	uncommitted := len(outb) > 0

	return Status{
		Revision:    rev,
		CommitTime:  commitTime,
		Uncommitted: uncommitted,
	}, nil
}

// vcsSvn describes how to use Subversion.
var vcsSvn = &Cmd{
	Name:      "Subversion",
	Cmd:       "svn",
	RootNames: []string{".svn"},

	CreateCmd:   []string{"checkout -- {repo} {dir}"},
	DownloadCmd: []string{"update"},

	// There is no tag command in subversion.
	// The branch information is all in the path names.

	Scheme:     []string{"https", "http", "svn", "svn+ssh"},
	PingCmd:    "info -- {scheme}://{repo}",
	RemoteRepo: svnRemoteRepo,
}

func svnRemoteRepo(vcsSvn *Cmd, rootDir string) (remoteRepo string, err error) {
	outb, err := vcsSvn.runOutput(rootDir, "info")
	if err != nil {
		return "", err
	}
	out := string(outb)

	// Expect:
	//
	//	 ...
	// 	URL: <URL>
	// 	...
	//
	// Note that we're not using the Repository Root line,
	// because svn allows checking out subtrees.
	// The URL will be the URL of the subtree (what we used with 'svn co')
	// while the Repository Root may be a much higher parent.
	i := strings.Index(out, "\nURL: ")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of svn info")
	}
	out = out[i+len("\nURL: "):]
	i = strings.Index(out, "\n")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of svn info")
	}
	out = out[:i]
	return strings.TrimSpace(out), nil
}

// fossilRepoName is the name go get associates with a fossil repository. In the
// real world the file can be named anything.
const fossilRepoName = ".fossil"

// vcsFossil describes how to use Fossil (fossil-scm.org)
var vcsFossil = &Cmd{
	Name:      "Fossil",
	Cmd:       "fossil",
	RootNames: []string{".fslckout", "_FOSSIL_"},

	CreateCmd:   []string{"-go-internal-mkdir {dir} clone -- {repo} " + filepath.Join("{dir}", fossilRepoName), "-go-internal-cd {dir} open .fossil"},
	DownloadCmd: []string{"up"},

	TagCmd:         []tagCmd{{"tag ls", `(.*)`}},
	TagSyncCmd:     []string{"up tag:{tag}"},
	TagSyncDefault: []string{"up trunk"},

	Scheme:     []string{"https", "http"},
	RemoteRepo: fossilRemoteRepo,
	Status:     fossilStatus,
}

func fossilRemoteRepo(vcsFossil *Cmd, rootDir string) (remoteRepo string, err error) {
	out, err := vcsFossil.runOutput(rootDir, "remote-url")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

var errFossilInfo = errors.New("unable to parse output of fossil info")

func fossilStatus(vcsFossil *Cmd, rootDir string) (Status, error) {
	outb, err := vcsFossil.runOutputVerboseOnly(rootDir, "info")
	if err != nil {
		return Status{}, err
	}
	out := string(outb)

	// Expect:
	// ...
	// checkout:     91ed71f22c77be0c3e250920f47bfd4e1f9024d2 2021-09-21 12:00:00 UTC
	// ...

	// Extract revision and commit time.
	// Ensure line ends with UTC (known timezone offset).
	const prefix = "\ncheckout:"
	const suffix = " UTC"
	i := strings.Index(out, prefix)
	if i < 0 {
		return Status{}, errFossilInfo
	}
	checkout := out[i+len(prefix):]
	i = strings.Index(checkout, suffix)
	if i < 0 {
		return Status{}, errFossilInfo
	}
	checkout = strings.TrimSpace(checkout[:i])

	i = strings.IndexByte(checkout, ' ')
	if i < 0 {
		return Status{}, errFossilInfo
	}
	rev := checkout[:i]

	commitTime, err := time.ParseInLocation("2006-01-02 15:04:05", checkout[i+1:], time.UTC)
	if err != nil {
		return Status{}, fmt.Errorf("%v: %v", errFossilInfo, err)
	}

	// Also look for untracked changes.
	outb, err = vcsFossil.runOutputVerboseOnly(rootDir, "changes --differ")
	if err != nil {
		return Status{}, err
	}
	uncommitted := len(outb) > 0

	return Status{
		Revision:    rev,
		CommitTime:  commitTime,
		Uncommitted: uncommitted,
	}, nil
}

func (v *Cmd) String() string {
	return v.Name
}

// run runs the command line cmd in the given directory.
// keyval is a list of key, value pairs. run expands
// instances of {key} in cmd into value, but only after
// splitting cmd into individual arguments.
// If an error occurs, run prints the command line and the
// command's combined stdout+stderr to standard error.
// Otherwise run discards the command's output.
func (v *Cmd) run(dir string, cmd string, keyval ...string) error {
	_, err := v.run1(dir, cmd, keyval, true)
	return err
}

// runVerboseOnly is like run but only generates error output to standard error in verbose mode.
func (v *Cmd) runVerboseOnly(dir string, cmd string, keyval ...string) error {
	_, err := v.run1(dir, cmd, keyval, false)
	return err
}

// runOutput is like run but returns the output of the command.
func (v *Cmd) runOutput(dir string, cmd string, keyval ...string) ([]byte, error) {
	return v.run1(dir, cmd, keyval, true)
}

// runOutputVerboseOnly is like runOutput but only generates error output to
// standard error in verbose mode.
func (v *Cmd) runOutputVerboseOnly(dir string, cmd string, keyval ...string) ([]byte, error) {
	return v.run1(dir, cmd, keyval, false)
}

// run1 is the generalized implementation of run and runOutput.
func (v *Cmd) run1(dir string, cmdline string, keyval []string, verbose bool) ([]byte, error) {
	m := make(map[string]string)
	for i := 0; i < len(keyval); i += 2 {
		m[keyval[i]] = keyval[i+1]
	}
	args := strings.Fields(cmdline)
	for i, arg := range args {
		args[i] = expand(m, arg)
	}

	if len(args) >= 2 && args[0] == "-go-internal-mkdir" {
		var err error
		if filepath.IsAbs(args[1]) {
			err = os.Mkdir(args[1], fs.ModePerm)
		} else {
			err = os.Mkdir(filepath.Join(dir, args[1]), fs.ModePerm)
		}
		if err != nil {
			return nil, err
		}
		args = args[2:]
	}

	if len(args) >= 2 && args[0] == "-go-internal-cd" {
		if filepath.IsAbs(args[1]) {
			dir = args[1]
		} else {
			dir = filepath.Join(dir, args[1])
		}
		args = args[2:]
	}

	_, err := exec.LookPath(v.Cmd)
	if err != nil {
		fmt.Fprintf(os.Stderr,
			"go: missing %s command. See https://golang.org/s/gogetcmd\n",
			v.Name)
		return nil, err
	}

	cmd := exec.Command(v.Cmd, args...)
	cmd.Dir = dir
	cmd.Env = base.AppendPWD(os.Environ(), cmd.Dir)
	if cfg.BuildX {
		fmt.Fprintf(os.Stderr, "cd %s\n", dir)
		fmt.Fprintf(os.Stderr, "%s %s\n", v.Cmd, strings.Join(args, " "))
	}
	out, err := cmd.Output()
	if err != nil {
		if verbose || cfg.BuildV {
			fmt.Fprintf(os.Stderr, "# cd %s; %s %s\n", dir, v.Cmd, strings.Join(args, " "))
			if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
				os.Stderr.Write(ee.Stderr)
			} else {
				fmt.Fprintf(os.Stderr, err.Error())
			}
		}
	}
	return out, err
}

// Ping pings to determine scheme to use.
func (v *Cmd) Ping(scheme, repo string) error {
	return v.runVerboseOnly(".", v.PingCmd, "scheme", scheme, "repo", repo)
}

// Create creates a new copy of repo in dir.
// The parent of dir must exist; dir must not.
func (v *Cmd) Create(dir, repo string) error {
	for _, cmd := range v.CreateCmd {
		if err := v.run(".", cmd, "dir", dir, "repo", repo); err != nil {
			return err
		}
	}
	return nil
}

// Download downloads any new changes for the repo in dir.
func (v *Cmd) Download(dir string) error {
	for _, cmd := range v.DownloadCmd {
		if err := v.run(dir, cmd); err != nil {
			return err
		}
	}
	return nil
}

// Tags returns the list of available tags for the repo in dir.
func (v *Cmd) Tags(dir string) ([]string, error) {
	var tags []string
	for _, tc := range v.TagCmd {
		out, err := v.runOutput(dir, tc.cmd)
		if err != nil {
			return nil, err
		}
		re := regexp.MustCompile(`(?m-s)` + tc.pattern)
		for _, m := range re.FindAllStringSubmatch(string(out), -1) {
			tags = append(tags, m[1])
		}
	}
	return tags, nil
}

// tagSync syncs the repo in dir to the named tag,
// which either is a tag returned by tags or is v.tagDefault.
func (v *Cmd) TagSync(dir, tag string) error {
	if v.TagSyncCmd == nil {
		return nil
	}
	if tag != "" {
		for _, tc := range v.TagLookupCmd {
			out, err := v.runOutput(dir, tc.cmd, "tag", tag)
			if err != nil {
				return err
			}
			re := regexp.MustCompile(`(?m-s)` + tc.pattern)
			m := re.FindStringSubmatch(string(out))
			if len(m) > 1 {
				tag = m[1]
				break
			}
		}
	}

	if tag == "" && v.TagSyncDefault != nil {
		for _, cmd := range v.TagSyncDefault {
			if err := v.run(dir, cmd); err != nil {
				return err
			}
		}
		return nil
	}

	for _, cmd := range v.TagSyncCmd {
		if err := v.run(dir, cmd, "tag", tag); err != nil {
			return err
		}
	}
	return nil
}

// A vcsPath describes how to convert an import path into a
// version control system and repository name.
type vcsPath struct {
	pathPrefix     string                              // prefix this description applies to
	regexp         *lazyregexp.Regexp                  // compiled pattern for import path
	repo           string                              // repository to use (expand with match of re)
	vcs            string                              // version control system to use (expand with match of re)
	check          func(match map[string]string) error // additional checks
	schemelessRepo bool                                // if true, the repo pattern lacks a scheme
}

// FromDir inspects dir and its parents to determine the
// version control system and code repository to use.
// If no repository is found, FromDir returns an error
// equivalent to os.ErrNotExist.
func FromDir(dir, srcRoot string, allowNesting bool) (repoDir string, vcsCmd *Cmd, err error) {
	// Clean and double-check that dir is in (a subdirectory of) srcRoot.
	dir = filepath.Clean(dir)
	if srcRoot != "" {
		srcRoot = filepath.Clean(srcRoot)
		if len(dir) <= len(srcRoot) || dir[len(srcRoot)] != filepath.Separator {
			return "", nil, fmt.Errorf("directory %q is outside source root %q", dir, srcRoot)
		}
	}

	origDir := dir
	for len(dir) > len(srcRoot) {
		for _, vcs := range vcsList {
			if _, err := statAny(dir, vcs.RootNames); err == nil {
				// Record first VCS we find.
				// If allowNesting is false (as it is in GOPATH), keep looking for
				// repositories in parent directories and report an error if one is
				// found to mitigate VCS injection attacks.
				if vcsCmd == nil {
					vcsCmd = vcs
					repoDir = dir
					if allowNesting {
						return repoDir, vcsCmd, nil
					}
					continue
				}
				// Allow .git inside .git, which can arise due to submodules.
				if vcsCmd == vcs && vcs.Cmd == "git" {
					continue
				}
				// Otherwise, we have one VCS inside a different VCS.
				return "", nil, fmt.Errorf("directory %q uses %s, but parent %q uses %s",
					repoDir, vcsCmd.Cmd, dir, vcs.Cmd)
			}
		}

		// Move to parent.
		ndir := filepath.Dir(dir)
		if len(ndir) >= len(dir) {
			break
		}
		dir = ndir
	}
	if vcsCmd == nil {
		return "", nil, &vcsNotFoundError{dir: origDir}
	}
	return repoDir, vcsCmd, nil
}

// statAny provides FileInfo for the first filename found in the directory.
// Otherwise, it returns the last error seen.
func statAny(dir string, filenames []string) (os.FileInfo, error) {
	if len(filenames) == 0 {
		return nil, errors.New("invalid argument: no filenames provided")
	}

	var err error
	var fi os.FileInfo
	for _, name := range filenames {
		fi, err = os.Stat(filepath.Join(dir, name))
		if err == nil {
			return fi, nil
		}
	}

	return nil, err
}

type vcsNotFoundError struct {
	dir string
}

func (e *vcsNotFoundError) Error() string {
	return fmt.Sprintf("directory %q is not using a known version control system", e.dir)
}

func (e *vcsNotFoundError) Is(err error) bool {
	return err == os.ErrNotExist
}

// A govcsRule is a single GOVCS rule like private:hg|svn.
type govcsRule struct {
	pattern string
	allowed []string
}

// A govcsConfig is a full GOVCS configuration.
type govcsConfig []govcsRule

func parseGOVCS(s string) (govcsConfig, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var cfg govcsConfig
	have := make(map[string]string)
	for _, item := range strings.Split(s, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			return nil, fmt.Errorf("empty entry in GOVCS")
		}
		i := strings.Index(item, ":")
		if i < 0 {
			return nil, fmt.Errorf("malformed entry in GOVCS (missing colon): %q", item)
		}
		pattern, list := strings.TrimSpace(item[:i]), strings.TrimSpace(item[i+1:])
		if pattern == "" {
			return nil, fmt.Errorf("empty pattern in GOVCS: %q", item)
		}
		if list == "" {
			return nil, fmt.Errorf("empty VCS list in GOVCS: %q", item)
		}
		if search.IsRelativePath(pattern) {
			return nil, fmt.Errorf("relative pattern not allowed in GOVCS: %q", pattern)
		}
		if old := have[pattern]; old != "" {
			return nil, fmt.Errorf("unreachable pattern in GOVCS: %q after %q", item, old)
		}
		have[pattern] = item
		allowed := strings.Split(list, "|")
		for i, a := range allowed {
			a = strings.TrimSpace(a)
			if a == "" {
				return nil, fmt.Errorf("empty VCS name in GOVCS: %q", item)
			}
			allowed[i] = a
		}
		cfg = append(cfg, govcsRule{pattern, allowed})
	}
	return cfg, nil
}

func (c *govcsConfig) allow(path string, private bool, vcs string) bool {
	for _, rule := range *c {
		match := false
		switch rule.pattern {
		case "private":
			match = private
		case "public":
			match = !private
		default:
			// Note: rule.pattern is known to be comma-free,
			// so MatchPrefixPatterns is only matching a single pattern for us.
			match = module.MatchPrefixPatterns(rule.pattern, path)
		}
		if !match {
			continue
		}
		for _, allow := range rule.allowed {
			if allow == vcs || allow == "all" {
				return true
			}
		}
		return false
	}

	// By default, nothing is allowed.
	return false
}

var (
	govcs     govcsConfig
	govcsErr  error
	govcsOnce sync.Once
)

// defaultGOVCS is the default setting for GOVCS.
// Setting GOVCS adds entries ahead of these but does not remove them.
// (They are appended to the parsed GOVCS setting.)
//
// The rationale behind allowing only Git and Mercurial is that
// these two systems have had the most attention to issues
// of being run as clients of untrusted servers. In contrast,
// Bazaar, Fossil, and Subversion have primarily been used
// in trusted, authenticated environments and are not as well
// scrutinized as attack surfaces.
//
// See golang.org/issue/41730 for details.
var defaultGOVCS = govcsConfig{
	{"private", []string{"all"}},
	{"public", []string{"git", "hg"}},
}

// CheckGOVCS checks whether the policy defined by the environment variable
// GOVCS allows the given vcs command to be used with the given repository
// root path. Note that root may not be a real package or module path; it's
// the same as the root path in the go-import meta tag.
func CheckGOVCS(vcs *Cmd, root string) error {
	if vcs == vcsMod {
		// Direct module (proxy protocol) fetches don't
		// involve an external version control system
		// and are always allowed.
		return nil
	}

	govcsOnce.Do(func() {
		govcs, govcsErr = parseGOVCS(os.Getenv("GOVCS"))
		govcs = append(govcs, defaultGOVCS...)
	})
	if govcsErr != nil {
		return govcsErr
	}

	private := module.MatchPrefixPatterns(cfg.GOPRIVATE, root)
	if !govcs.allow(root, private, vcs.Cmd) {
		what := "public"
		if private {
			what = "private"
		}
		return fmt.Errorf("GOVCS disallows using %s for %s %s; see 'go help vcs'", vcs.Cmd, what, root)
	}

	return nil
}

// CheckNested checks for an incorrectly-nested VCS-inside-VCS
// situation for dir, checking parents up until srcRoot.
func CheckNested(vcs *Cmd, dir, srcRoot string) error {
	if len(dir) <= len(srcRoot) || dir[len(srcRoot)] != filepath.Separator {
		return fmt.Errorf("directory %q is outside source root %q", dir, srcRoot)
	}

	otherDir := dir
	for len(otherDir) > len(srcRoot) {
		for _, otherVCS := range vcsList {
			if _, err := statAny(otherDir, otherVCS.RootNames); err == nil {
				// Allow expected vcs in original dir.
				if otherDir == dir && otherVCS == vcs {
					continue
				}
				// Allow .git inside .git, which can arise due to submodules.
				if otherVCS == vcs && vcs.Cmd == "git" {
					continue
				}
				// Otherwise, we have one VCS inside a different VCS.
				return fmt.Errorf("directory %q uses %s, but parent %q uses %s", dir, vcs.Cmd, otherDir, otherVCS.Cmd)
			}
		}
		// Move to parent.
		newDir := filepath.Dir(otherDir)
		if len(newDir) >= len(otherDir) {
			// Shouldn't happen, but just in case, stop.
			break
		}
		otherDir = newDir
	}

	return nil
}

// RepoRoot describes the repository root for a tree of source code.
type RepoRoot struct {
	Repo     string // repository URL, including scheme
	Root     string // import path corresponding to root of repo
	IsCustom bool   // defined by served <meta> tags (as opposed to hard-coded pattern)
	VCS      *Cmd
}

func httpPrefix(s string) string {
	for _, prefix := range [...]string{"http:", "https:"} {
		if strings.HasPrefix(s, prefix) {
			return prefix
		}
	}
	return ""
}

// ModuleMode specifies whether to prefer modules when looking up code sources.
type ModuleMode int

const (
	IgnoreMod ModuleMode = iota
	PreferMod
)

// RepoRootForImportPath analyzes importPath to determine the
// version control system, and code repository to use.
func RepoRootForImportPath(importPath string, mod ModuleMode, security web.SecurityMode) (*RepoRoot, error) {
	rr, err := repoRootFromVCSPaths(importPath, security, vcsPaths)
	if err == errUnknownSite {
		rr, err = repoRootForImportDynamic(importPath, mod, security)
		if err != nil {
			err = importErrorf(importPath, "unrecognized import path %q: %v", importPath, err)
		}
	}
	if err != nil {
		rr1, err1 := repoRootFromVCSPaths(importPath, security, vcsPathsAfterDynamic)
		if err1 == nil {
			rr = rr1
			err = nil
		}
	}

	// Should have been taken care of above, but make sure.
	if err == nil && strings.Contains(importPath, "...") && strings.Contains(rr.Root, "...") {
		// Do not allow wildcards in the repo root.
		rr = nil
		err = importErrorf(importPath, "cannot expand ... in %q", importPath)
	}
	return rr, err
}

var errUnknownSite = errors.New("dynamic lookup required to find mapping")

// repoRootFromVCSPaths attempts to map importPath to a repoRoot
// using the mappings defined in vcsPaths.
func repoRootFromVCSPaths(importPath string, security web.SecurityMode, vcsPaths []*vcsPath) (*RepoRoot, error) {
	if str.HasPathPrefix(importPath, "example.net") {
		// TODO(rsc): This should not be necessary, but it's required to keep
		// tests like ../../testdata/script/mod_get_extra.txt from using the network.
		// That script has everything it needs in the replacement set, but it is still
		// doing network calls.
		return nil, fmt.Errorf("no modules on example.net")
	}
	if importPath == "rsc.io" {
		// This special case allows tests like ../../testdata/script/govcs.txt
		// to avoid making any network calls. The module lookup for a path
		// like rsc.io/nonexist.svn/foo needs to not make a network call for
		// a lookup on rsc.io.
		return nil, fmt.Errorf("rsc.io is not a module")
	}
	// A common error is to use https://packagepath because that's what
	// hg and git require. Diagnose this helpfully.
	if prefix := httpPrefix(importPath); prefix != "" {
		// The importPath has been cleaned, so has only one slash. The pattern
		// ignores the slashes; the error message puts them back on the RHS at least.
		return nil, fmt.Errorf("%q not allowed in import path", prefix+"//")
	}
	for _, srv := range vcsPaths {
		if !str.HasPathPrefix(importPath, srv.pathPrefix) {
			continue
		}
		m := srv.regexp.FindStringSubmatch(importPath)
		if m == nil {
			if srv.pathPrefix != "" {
				return nil, importErrorf(importPath, "invalid %s import path %q", srv.pathPrefix, importPath)
			}
			continue
		}

		// Build map of named subexpression matches for expand.
		match := map[string]string{
			"prefix": srv.pathPrefix + "/",
			"import": importPath,
		}
		for i, name := range srv.regexp.SubexpNames() {
			if name != "" && match[name] == "" {
				match[name] = m[i]
			}
		}
		if srv.vcs != "" {
			match["vcs"] = expand(match, srv.vcs)
		}
		if srv.repo != "" {
			match["repo"] = expand(match, srv.repo)
		}
		if srv.check != nil {
			if err := srv.check(match); err != nil {
				return nil, err
			}
		}
		vcs := vcsByCmd(match["vcs"])
		if vcs == nil {
			return nil, fmt.Errorf("unknown version control system %q", match["vcs"])
		}
		if err := CheckGOVCS(vcs, match["root"]); err != nil {
			return nil, err
		}
		var repoURL string
		if !srv.schemelessRepo {
			repoURL = match["repo"]
		} else {
			scheme := vcs.Scheme[0] // default to first scheme
			repo := match["repo"]
			if vcs.PingCmd != "" {
				// If we know how to test schemes, scan to find one.
				for _, s := range vcs.Scheme {
					if security == web.SecureOnly && !vcs.isSecureScheme(s) {
						continue
					}
					if vcs.Ping(s, repo) == nil {
						scheme = s
						break
					}
				}
			}
			repoURL = scheme + "://" + repo
		}
		rr := &RepoRoot{
			Repo: repoURL,
			Root: match["root"],
			VCS:  vcs,
		}
		return rr, nil
	}
	return nil, errUnknownSite
}

// urlForImportPath returns a partially-populated URL for the given Go import path.
//
// The URL leaves the Scheme field blank so that web.Get will try any scheme
// allowed by the selected security mode.
func urlForImportPath(importPath string) (*urlpkg.URL, error) {
	slash := strings.Index(importPath, "/")
	if slash < 0 {
		slash = len(importPath)
	}
	host, path := importPath[:slash], importPath[slash:]
	if !strings.Contains(host, ".") {
		return nil, errors.New("import path does not begin with hostname")
	}
	if len(path) == 0 {
		path = "/"
	}
	return &urlpkg.URL{Host: host, Path: path, RawQuery: "go-get=1"}, nil
}

// repoRootForImportDynamic finds a *RepoRoot for a custom domain that's not
// statically known by repoRootForImportPathStatic.
//
// This handles custom import paths like "name.tld/pkg/foo" or just "name.tld".
func repoRootForImportDynamic(importPath string, mod ModuleMode, security web.SecurityMode) (*RepoRoot, error) {
	url, err := urlForImportPath(importPath)
	if err != nil {
		return nil, err
	}
	resp, err := web.Get(security, url)
	if err != nil {
		msg := "https fetch: %v"
		if security == web.Insecure {
			msg = "http/" + msg
		}
		return nil, fmt.Errorf(msg, err)
	}
	body := resp.Body
	defer body.Close()
	imports, err := parseMetaGoImports(body, mod)
	if len(imports) == 0 {
		if respErr := resp.Err(); respErr != nil {
			// If the server's status was not OK, prefer to report that instead of
			// an XML parse error.
			return nil, respErr
		}
	}
	if err != nil {
		return nil, fmt.Errorf("parsing %s: %v", importPath, err)
	}
	// Find the matched meta import.
	mmi, err := matchGoImport(imports, importPath)
	if err != nil {
		if _, ok := err.(ImportMismatchError); !ok {
			return nil, fmt.Errorf("parse %s: %v", url, err)
		}
		return nil, fmt.Errorf("parse %s: no go-import meta tags (%s)", resp.URL, err)
	}
	if cfg.BuildV {
		log.Printf("get %q: found meta tag %#v at %s", importPath, mmi, url)
	}
	// If the import was "uni.edu/bob/project", which said the
	// prefix was "uni.edu" and the RepoRoot was "evilroot.com",
	// make sure we don't trust Bob and check out evilroot.com to
	// "uni.edu" yet (possibly overwriting/preempting another
	// non-evil student). Instead, first verify the root and see
	// if it matches Bob's claim.
	if mmi.Prefix != importPath {
		if cfg.BuildV {
			log.Printf("get %q: verifying non-authoritative meta tag", importPath)
		}
		var imports []metaImport
		url, imports, err = metaImportsForPrefix(mmi.Prefix, mod, security)
		if err != nil {
			return nil, err
		}
		metaImport2, err := matchGoImport(imports, importPath)
		if err != nil || mmi != metaImport2 {
			return nil, fmt.Errorf("%s and %s disagree about go-import for %s", resp.URL, url, mmi.Prefix)
		}
	}

	if err := validateRepoRoot(mmi.RepoRoot); err != nil {
		return nil, fmt.Errorf("%s: invalid repo root %q: %v", resp.URL, mmi.RepoRoot, err)
	}
	var vcs *Cmd
	if mmi.VCS == "mod" {
		vcs = vcsMod
	} else {
		vcs = vcsByCmd(mmi.VCS)
		if vcs == nil {
			return nil, fmt.Errorf("%s: unknown vcs %q", resp.URL, mmi.VCS)
		}
	}

	if err := CheckGOVCS(vcs, mmi.Prefix); err != nil {
		return nil, err
	}

	rr := &RepoRoot{
		Repo:     mmi.RepoRoot,
		Root:     mmi.Prefix,
		IsCustom: true,
		VCS:      vcs,
	}
	return rr, nil
}

// validateRepoRoot returns an error if repoRoot does not seem to be
// a valid URL with scheme.
func validateRepoRoot(repoRoot string) error {
	url, err := urlpkg.Parse(repoRoot)
	if err != nil {
		return err
	}
	if url.Scheme == "" {
		return errors.New("no scheme")
	}
	if url.Scheme == "file" {
		return errors.New("file scheme disallowed")
	}
	return nil
}

var fetchGroup singleflight.Group
var (
	fetchCacheMu sync.Mutex
	fetchCache   = map[string]fetchResult{} // key is metaImportsForPrefix's importPrefix
)

// metaImportsForPrefix takes a package's root import path as declared in a <meta> tag
// and returns its HTML discovery URL and the parsed metaImport lines
// found on the page.
//
// The importPath is of the form "golang.org/x/tools".
// It is an error if no imports are found.
// url will still be valid if err != nil.
// The returned url will be of the form "https://golang.org/x/tools?go-get=1"
func metaImportsForPrefix(importPrefix string, mod ModuleMode, security web.SecurityMode) (*urlpkg.URL, []metaImport, error) {
	setCache := func(res fetchResult) (fetchResult, error) {
		fetchCacheMu.Lock()
		defer fetchCacheMu.Unlock()
		fetchCache[importPrefix] = res
		return res, nil
	}

	resi, _, _ := fetchGroup.Do(importPrefix, func() (resi interface{}, err error) {
		fetchCacheMu.Lock()
		if res, ok := fetchCache[importPrefix]; ok {
			fetchCacheMu.Unlock()
			return res, nil
		}
		fetchCacheMu.Unlock()

		url, err := urlForImportPath(importPrefix)
		if err != nil {
			return setCache(fetchResult{err: err})
		}
		resp, err := web.Get(security, url)
		if err != nil {
			return setCache(fetchResult{url: url, err: fmt.Errorf("fetching %s: %v", importPrefix, err)})
		}
		body := resp.Body
		defer body.Close()
		imports, err := parseMetaGoImports(body, mod)
		if len(imports) == 0 {
			if respErr := resp.Err(); respErr != nil {
				// If the server's status was not OK, prefer to report that instead of
				// an XML parse error.
				return setCache(fetchResult{url: url, err: respErr})
			}
		}
		if err != nil {
			return setCache(fetchResult{url: url, err: fmt.Errorf("parsing %s: %v", resp.URL, err)})
		}
		if len(imports) == 0 {
			err = fmt.Errorf("fetching %s: no go-import meta tag found in %s", importPrefix, resp.URL)
		}
		return setCache(fetchResult{url: url, imports: imports, err: err})
	})
	res := resi.(fetchResult)
	return res.url, res.imports, res.err
}

type fetchResult struct {
	url     *urlpkg.URL
	imports []metaImport
	err     error
}

// metaImport represents the parsed <meta name="go-import"
// content="prefix vcs reporoot" /> tags from HTML files.
type metaImport struct {
	Prefix, VCS, RepoRoot string
}

// A ImportMismatchError is returned where metaImport/s are present
// but none match our import path.
type ImportMismatchError struct {
	importPath string
	mismatches []string // the meta imports that were discarded for not matching our importPath
}

func (m ImportMismatchError) Error() string {
	formattedStrings := make([]string, len(m.mismatches))
	for i, pre := range m.mismatches {
		formattedStrings[i] = fmt.Sprintf("meta tag %s did not match import path %s", pre, m.importPath)
	}
	return strings.Join(formattedStrings, ", ")
}

// matchGoImport returns the metaImport from imports matching importPath.
// An error is returned if there are multiple matches.
// An ImportMismatchError is returned if none match.
func matchGoImport(imports []metaImport, importPath string) (metaImport, error) {
	match := -1

	errImportMismatch := ImportMismatchError{importPath: importPath}
	for i, im := range imports {
		if !str.HasPathPrefix(importPath, im.Prefix) {
			errImportMismatch.mismatches = append(errImportMismatch.mismatches, im.Prefix)
			continue
		}

		if match >= 0 {
			if imports[match].VCS == "mod" && im.VCS != "mod" {
				// All the mod entries precede all the non-mod entries.
				// We have a mod entry and don't care about the rest,
				// matching or not.
				break
			}
			return metaImport{}, fmt.Errorf("multiple meta tags match import path %q", importPath)
		}
		match = i
	}

	if match == -1 {
		return metaImport{}, errImportMismatch
	}
	return imports[match], nil
}

// expand rewrites s to replace {k} with match[k] for each key k in match.
func expand(match map[string]string, s string) string {
	// We want to replace each match exactly once, and the result of expansion
	// must not depend on the iteration order through the map.
	// A strings.Replacer has exactly the properties we're looking for.
	oldNew := make([]string, 0, 2*len(match))
	for k, v := range match {
		oldNew = append(oldNew, "{"+k+"}", v)
	}
	return strings.NewReplacer(oldNew...).Replace(s)
}

// vcsPaths defines the meaning of import paths referring to
// commonly-used VCS hosting sites (github.com/user/dir)
// and import paths referring to a fully-qualified importPath
// containing a VCS type (foo.com/repo.git/dir)
var vcsPaths = []*vcsPath{
	// GitHub
	{
		pathPrefix: "github.com",
		regexp:     lazyregexp.New(`^(?P<root>github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)*$`),
		vcs:        "git",
		repo:       "https://{root}",
		check:      noVCSSuffix,
	},

	// Bitbucket
	{
		pathPrefix: "bitbucket.org",
		regexp:     lazyregexp.New(`^(?P<root>bitbucket\.org/(?P<bitname>[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`),
		repo:       "https://{root}",
		check:      bitbucketVCS,
	},

	// IBM DevOps Services (JazzHub)
	{
		pathPrefix: "hub.jazz.net/git",
		regexp:     lazyregexp.New(`^(?P<root>hub\.jazz\.net/git/[a-z0-9]+/[A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)*$`),
		vcs:        "git",
		repo:       "https://{root}",
		check:      noVCSSuffix,
	},

	// Git at Apache
	{
		pathPrefix: "git.apache.org",
		regexp:     lazyregexp.New(`^(?P<root>git\.apache\.org/[a-z0-9_.\-]+\.git)(/[A-Za-z0-9_.\-]+)*$`),
		vcs:        "git",
		repo:       "https://{root}",
	},

	// Git at OpenStack
	{
		pathPrefix: "git.openstack.org",
		regexp:     lazyregexp.New(`^(?P<root>git\.openstack\.org/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(\.git)?(/[A-Za-z0-9_.\-]+)*$`),
		vcs:        "git",
		repo:       "https://{root}",
	},

	// chiselapp.com for fossil
	{
		pathPrefix: "chiselapp.com",
		regexp:     lazyregexp.New(`^(?P<root>chiselapp\.com/user/[A-Za-z0-9]+/repository/[A-Za-z0-9_.\-]+)$`),
		vcs:        "fossil",
		repo:       "https://{root}",
	},

	// General syntax for any server.
	// Must be last.
	{
		regexp:         lazyregexp.New(`(?P<root>(?P<repo>([a-z0-9.\-]+\.)+[a-z0-9.\-]+(:[0-9]+)?(/~?[A-Za-z0-9_.\-]+)+?)\.(?P<vcs>bzr|fossil|git|hg|svn))(/~?[A-Za-z0-9_.\-]+)*$`),
		schemelessRepo: true,
	},
}

// vcsPathsAfterDynamic gives additional vcsPaths entries
// to try after the dynamic HTML check.
// This gives those sites a chance to introduce <meta> tags
// as part of a graceful transition away from the hard-coded logic.
var vcsPathsAfterDynamic = []*vcsPath{
	// Launchpad. See golang.org/issue/11436.
	{
		pathPrefix: "launchpad.net",
		regexp:     lazyregexp.New(`^(?P<root>launchpad\.net/((?P<project>[A-Za-z0-9_.\-]+)(?P<series>/[A-Za-z0-9_.\-]+)?|~[A-Za-z0-9_.\-]+/(\+junk|[A-Za-z0-9_.\-]+)/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`),
		vcs:        "bzr",
		repo:       "https://{root}",
		check:      launchpadVCS,
	},
}

// noVCSSuffix checks that the repository name does not
// end in .foo for any version control system foo.
// The usual culprit is ".git".
func noVCSSuffix(match map[string]string) error {
	repo := match["repo"]
	for _, vcs := range vcsList {
		if strings.HasSuffix(repo, "."+vcs.Cmd) {
			return fmt.Errorf("invalid version control suffix in %s path", match["prefix"])
		}
	}
	return nil
}

// bitbucketVCS determines the version control system for a
// Bitbucket repository, by using the Bitbucket API.
func bitbucketVCS(match map[string]string) error {
	if err := noVCSSuffix(match); err != nil {
		return err
	}

	var resp struct {
		SCM string `json:"scm"`
	}
	url := &urlpkg.URL{
		Scheme:   "https",
		Host:     "api.bitbucket.org",
		Path:     expand(match, "/2.0/repositories/{bitname}"),
		RawQuery: "fields=scm",
	}
	data, err := web.GetBytes(url)
	if err != nil {
		if httpErr, ok := err.(*web.HTTPError); ok && httpErr.StatusCode == 403 {
			// this may be a private repository. If so, attempt to determine which
			// VCS it uses. See issue 5375.
			root := match["root"]
			for _, vcs := range []string{"git", "hg"} {
				if vcsByCmd(vcs).Ping("https", root) == nil {
					resp.SCM = vcs
					break
				}
			}
		}

		if resp.SCM == "" {
			return err
		}
	} else {
		if err := json.Unmarshal(data, &resp); err != nil {
			return fmt.Errorf("decoding %s: %v", url, err)
		}
	}

	if vcsByCmd(resp.SCM) != nil {
		match["vcs"] = resp.SCM
		if resp.SCM == "git" {
			match["repo"] += ".git"
		}
		return nil
	}

	return fmt.Errorf("unable to detect version control system for bitbucket.org/ path")
}

// launchpadVCS solves the ambiguity for "lp.net/project/foo". In this case,
// "foo" could be a series name registered in Launchpad with its own branch,
// and it could also be the name of a directory within the main project
// branch one level up.
func launchpadVCS(match map[string]string) error {
	if match["project"] == "" || match["series"] == "" {
		return nil
	}
	url := &urlpkg.URL{
		Scheme: "https",
		Host:   "code.launchpad.net",
		Path:   expand(match, "/{project}{series}/.bzr/branch-format"),
	}
	_, err := web.GetBytes(url)
	if err != nil {
		match["root"] = expand(match, "launchpad.net/{project}")
		match["repo"] = expand(match, "https://{root}")
	}
	return nil
}

// importError is a copy of load.importError, made to avoid a dependency cycle
// on cmd/go/internal/load. It just needs to satisfy load.ImportPathError.
type importError struct {
	importPath string
	err        error
}

func importErrorf(path, format string, args ...interface{}) error {
	err := &importError{importPath: path, err: fmt.Errorf(format, args...)}
	if errStr := err.Error(); !strings.Contains(errStr, path) {
		panic(fmt.Sprintf("path %q not in error %q", path, errStr))
	}
	return err
}

func (e *importError) Error() string {
	return e.err.Error()
}

func (e *importError) Unwrap() error {
	// Don't return e.err directly, since we're only wrapping an error if %w
	// was passed to ImportErrorf.
	return errors.Unwrap(e.err)
}

func (e *importError) ImportPath() string {
	return e.importPath
}
