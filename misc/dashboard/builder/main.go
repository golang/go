// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

const (
	codeProject      = "go"
	codePyScript     = "misc/dashboard/googlecode_upload.py"
	hgUrl            = "https://code.google.com/p/go/"
	mkdirPerm        = 0750
	waitInterval     = 30 * time.Second // time to wait before checking for new revs
	pkgBuildInterval = 24 * time.Hour   // rebuild packages every 24 hours
)

// These variables are copied from the gobuilder's environment
// to the envv of its subprocesses.
var extraEnv = []string{
	"CC",
	"GOARM",
	"PATH",
	"TMPDIR",
	"USER",
}

type Builder struct {
	goroot       *Repo
	name         string
	goos, goarch string
	key          string
}

var (
	buildroot      = flag.String("buildroot", defaultBuildRoot(), "Directory under which to build")
	dashboard      = flag.String("dashboard", "build.golang.org", "Go Dashboard Host")
	buildRelease   = flag.Bool("release", false, "Build and upload binary release archives")
	buildRevision  = flag.String("rev", "", "Build specified revision and exit")
	buildCmd       = flag.String("cmd", filepath.Join(".", allCmd), "Build command (specify relative to go/src/)")
	failAll        = flag.Bool("fail", false, "fail all builds")
	parallel       = flag.Bool("parallel", false, "Build multiple targets in parallel")
	buildTimeout   = flag.Duration("buildTimeout", 60*time.Minute, "Maximum time to wait for builds and tests")
	cmdTimeout     = flag.Duration("cmdTimeout", 5*time.Minute, "Maximum time to wait for an external command")
	commitInterval = flag.Duration("commitInterval", 1*time.Minute, "Time to wait between polling for new commits (0 disables commit poller)")
	verbose        = flag.Bool("v", false, "verbose")
)

var (
	binaryTagRe = regexp.MustCompile(`^(release\.r|weekly\.)[0-9\-.]+`)
	releaseRe   = regexp.MustCompile(`^release\.r[0-9\-.]+`)
	allCmd      = "all" + suffix
	raceCmd     = "race" + suffix
	cleanCmd    = "clean" + suffix
	suffix      = defaultSuffix()
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s goos-goarch...\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(2)
	}
	flag.Parse()
	if len(flag.Args()) == 0 {
		flag.Usage()
	}
	goroot := &Repo{
		Path: filepath.Join(*buildroot, "goroot"),
	}

	// set up work environment, use existing enviroment if possible
	if goroot.Exists() || *failAll {
		log.Print("Found old workspace, will use it")
	} else {
		if err := os.RemoveAll(*buildroot); err != nil {
			log.Fatalf("Error removing build root (%s): %s", *buildroot, err)
		}
		if err := os.Mkdir(*buildroot, mkdirPerm); err != nil {
			log.Fatalf("Error making build root (%s): %s", *buildroot, err)
		}
		var err error
		goroot, err = RemoteRepo(hgUrl).Clone(goroot.Path, "tip")
		if err != nil {
			log.Fatal("Error cloning repository:", err)
		}
	}

	// set up builders
	builders := make([]*Builder, len(flag.Args()))
	for i, name := range flag.Args() {
		b, err := NewBuilder(goroot, name)
		if err != nil {
			log.Fatal(err)
		}
		builders[i] = b
	}

	if *failAll {
		failMode(builders)
		return
	}

	// if specified, build revision and return
	if *buildRevision != "" {
		hash, err := goroot.FullHash(*buildRevision)
		if err != nil {
			log.Fatal("Error finding revision: ", err)
		}
		for _, b := range builders {
			if err := b.buildHash(hash); err != nil {
				log.Println(err)
			}
		}
		return
	}

	// Start commit watcher
	go commitWatcher(goroot)

	// go continuous build mode
	// check for new commits and build them
	for {
		built := false
		t := time.Now()
		if *parallel {
			done := make(chan bool)
			for _, b := range builders {
				go func(b *Builder) {
					done <- b.build()
				}(b)
			}
			for _ = range builders {
				built = <-done || built
			}
		} else {
			for _, b := range builders {
				built = b.build() || built
			}
		}
		// sleep if there was nothing to build
		if !built {
			time.Sleep(waitInterval)
		}
		// sleep if we're looping too fast.
		dt := time.Now().Sub(t)
		if dt < waitInterval {
			time.Sleep(waitInterval - dt)
		}
	}
}

// go continuous fail mode
// check for new commits and FAIL them
func failMode(builders []*Builder) {
	for {
		built := false
		for _, b := range builders {
			built = b.failBuild() || built
		}
		// stop if there was nothing to fail
		if !built {
			break
		}
	}
}

func NewBuilder(goroot *Repo, name string) (*Builder, error) {
	b := &Builder{
		goroot: goroot,
		name:   name,
	}

	// get goos/goarch from builder string
	s := strings.SplitN(b.name, "-", 3)
	if len(s) >= 2 {
		b.goos, b.goarch = s[0], s[1]
	} else {
		return nil, fmt.Errorf("unsupported builder form: %s", name)
	}

	// read keys from keyfile
	fn := ""
	if runtime.GOOS == "windows" {
		fn = os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
	} else {
		fn = os.Getenv("HOME")
	}
	fn = filepath.Join(fn, ".gobuildkey")
	if s := fn + "-" + b.name; isFile(s) { // builder-specific file
		fn = s
	}
	c, err := ioutil.ReadFile(fn)
	if err != nil {
		return nil, fmt.Errorf("readKeys %s (%s): %s", b.name, fn, err)
	}
	b.key = string(bytes.TrimSpace(bytes.SplitN(c, []byte("\n"), 2)[0]))
	return b, nil
}

// buildCmd returns the build command to invoke.
// Builders which contain the string '-race' in their
// name will override *buildCmd and return raceCmd.
func (b *Builder) buildCmd() string {
	if strings.Contains(b.name, "-race") {
		return raceCmd
	}
	return *buildCmd
}

// build checks for a new commit for this builder
// and builds it if one is found.
// It returns true if a build was attempted.
func (b *Builder) build() bool {
	hash, err := b.todo("build-go-commit", "", "")
	if err != nil {
		log.Println(err)
		return false
	}
	if hash == "" {
		return false
	}

	if err := b.buildHash(hash); err != nil {
		log.Println(err)
	}
	return true
}

func (b *Builder) buildHash(hash string) error {
	log.Println(b.name, "building", hash)

	// create place in which to do work
	workpath := filepath.Join(*buildroot, b.name+"-"+hash[:12])
	if err := os.Mkdir(workpath, mkdirPerm); err != nil {
		return err
	}
	defer os.RemoveAll(workpath)

	// pull before cloning to ensure we have the revision
	if err := b.goroot.Pull(); err != nil {
		return err
	}

	// clone repo at specified revision
	if _, err := b.goroot.Clone(filepath.Join(workpath, "go"), hash); err != nil {
		return err
	}

	srcDir := filepath.Join(workpath, "go", "src")

	// build
	var buildlog bytes.Buffer
	logfile := filepath.Join(workpath, "build.log")
	f, err := os.Create(logfile)
	if err != nil {
		return err
	}
	defer f.Close()
	w := io.MultiWriter(f, &buildlog)

	cmd := b.buildCmd()
	if !filepath.IsAbs(cmd) {
		cmd = filepath.Join(srcDir, cmd)
	}
	startTime := time.Now()
	ok, err := runOutput(*buildTimeout, b.envv(), w, srcDir, cmd)
	runTime := time.Now().Sub(startTime)
	errf := func() string {
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		if !ok {
			return "failed"
		}
		return "success"
	}
	fmt.Fprintf(w, "Build complete, duration %v. Result: %v\n", runTime, errf())

	if err != nil || !ok {
		// record failure
		return b.recordResult(false, "", hash, "", buildlog.String(), runTime)
	}

	// record success
	if err = b.recordResult(true, "", hash, "", "", runTime); err != nil {
		return fmt.Errorf("recordResult: %s", err)
	}

	// build Go sub-repositories
	goRoot := filepath.Join(workpath, "go")
	goPath := workpath
	b.buildSubrepos(goRoot, goPath, hash)

	return nil
}

// failBuild checks for a new commit for this builder
// and fails it if one is found.
// It returns true if a build was "attempted".
func (b *Builder) failBuild() bool {
	hash, err := b.todo("build-go-commit", "", "")
	if err != nil {
		log.Println(err)
		return false
	}
	if hash == "" {
		return false
	}

	log.Printf("fail %s %s\n", b.name, hash)

	if err := b.recordResult(false, "", hash, "", "auto-fail mode run by "+os.Getenv("USER"), 0); err != nil {
		log.Print(err)
	}
	return true
}

func (b *Builder) buildSubrepos(goRoot, goPath, goHash string) {
	for _, pkg := range dashboardPackages("subrepo") {
		// get the latest todo for this package
		hash, err := b.todo("build-package", pkg, goHash)
		if err != nil {
			log.Printf("buildSubrepos %s: %v", pkg, err)
			continue
		}
		if hash == "" {
			continue
		}

		// build the package
		if *verbose {
			log.Printf("buildSubrepos %s: building %q", pkg, hash)
		}
		buildLog, err := b.buildSubrepo(goRoot, goPath, pkg, hash)
		if err != nil {
			if buildLog == "" {
				buildLog = err.Error()
			}
			log.Printf("buildSubrepos %s: %v", pkg, err)
		}

		// record the result
		err = b.recordResult(err == nil, pkg, hash, goHash, buildLog, 0)
		if err != nil {
			log.Printf("buildSubrepos %s: %v", pkg, err)
		}
	}
}

// buildSubrepo fetches the given package, updates it to the specified hash,
// and runs 'go test -short pkg/...'. It returns the build log and any error.
func (b *Builder) buildSubrepo(goRoot, goPath, pkg, hash string) (string, error) {
	goTool := filepath.Join(goRoot, "bin", "go")
	env := append(b.envv(), "GOROOT="+goRoot, "GOPATH="+goPath)

	// add $GOROOT/bin and $GOPATH/bin to PATH
	for i, e := range env {
		const p = "PATH="
		if !strings.HasPrefix(e, p) {
			continue
		}
		sep := string(os.PathListSeparator)
		env[i] = p + filepath.Join(goRoot, "bin") + sep + filepath.Join(goPath, "bin") + sep + e[len(p):]
	}

	// fetch package and dependencies
	log, ok, err := runLog(*cmdTimeout, env, goPath, goTool, "get", "-d", pkg+"/...")
	if err == nil && !ok {
		err = fmt.Errorf("go exited with status 1")
	}
	if err != nil {
		return log, err
	}

	// hg update to the specified hash
	repo := Repo{Path: filepath.Join(goPath, "src", pkg)}
	if err := repo.UpdateTo(hash); err != nil {
		return "", err
	}

	// test the package
	log, ok, err = runLog(*buildTimeout, env, goPath, goTool, "test", "-short", pkg+"/...")
	if err == nil && !ok {
		err = fmt.Errorf("go exited with status 1")
	}
	return log, err
}

// envv returns an environment for build/bench execution
func (b *Builder) envv() []string {
	if runtime.GOOS == "windows" {
		return b.envvWindows()
	}
	e := []string{
		"GOOS=" + b.goos,
		"GOHOSTOS=" + b.goos,
		"GOARCH=" + b.goarch,
		"GOHOSTARCH=" + b.goarch,
		"GOROOT_FINAL=/usr/local/go",
	}
	for _, k := range extraEnv {
		if s, ok := getenvOk(k); ok {
			e = append(e, k+"="+s)
		}
	}
	return e
}

// windows version of envv
func (b *Builder) envvWindows() []string {
	start := map[string]string{
		"GOOS":         b.goos,
		"GOHOSTOS":     b.goos,
		"GOARCH":       b.goarch,
		"GOHOSTARCH":   b.goarch,
		"GOROOT_FINAL": `c:\go`,
		"GOBUILDEXIT":  "1", // exit all.bat with completion status.
	}
	for _, name := range extraEnv {
		if s, ok := getenvOk(name); ok {
			start[name] = s
		}
	}
	skip := map[string]bool{
		"GOBIN":   true,
		"GOROOT":  true,
		"INCLUDE": true,
		"LIB":     true,
	}
	var e []string
	for name, v := range start {
		e = append(e, name+"="+v)
		skip[name] = true
	}
	for _, kv := range os.Environ() {
		s := strings.SplitN(kv, "=", 2)
		name := strings.ToUpper(s[0])
		switch {
		case name == "":
			// variables, like "=C:=C:\", just copy them
			e = append(e, kv)
		case !skip[name]:
			e = append(e, kv)
			skip[name] = true
		}
	}
	return e
}

func isDirectory(name string) bool {
	s, err := os.Stat(name)
	return err == nil && s.IsDir()
}

func isFile(name string) bool {
	s, err := os.Stat(name)
	return err == nil && !s.IsDir()
}

// commitWatcher polls hg for new commits and tells the dashboard about them.
func commitWatcher(goroot *Repo) {
	if *commitInterval == 0 {
		log.Printf("commitInterval is %s, disabling commitWatcher", *commitInterval)
		return
	}
	// Create builder just to get master key.
	b, err := NewBuilder(goroot, "mercurial-commit")
	if err != nil {
		log.Fatal(err)
	}
	key := b.key

	for {
		if *verbose {
			log.Printf("poll...")
		}
		// Main Go repository.
		commitPoll(goroot, "", key)
		// Go sub-repositories.
		for _, pkg := range dashboardPackages("subrepo") {
			pkgroot := &Repo{
				Path: filepath.Join(*buildroot, pkg),
			}
			commitPoll(pkgroot, pkg, key)
		}
		if *verbose {
			log.Printf("sleep...")
		}
		time.Sleep(*commitInterval)
	}
}

// logByHash is a cache of all Mercurial revisions we know about,
// indexed by full hash.
var logByHash = map[string]*HgLog{}

// commitPoll pulls any new revisions from the hg server
// and tells the server about them.
func commitPoll(repo *Repo, pkg, key string) {
	if !repo.Exists() {
		var err error
		repo, err = RemoteRepo(repoURL(pkg)).Clone(repo.Path, "tip")
		if err != nil {
			log.Printf("%s: hg clone failed: %v", pkg, err)
			if err := os.RemoveAll(repo.Path); err != nil {
				log.Printf("%s: %v", pkg, err)
			}
		}
		return
	}

	logs, err := repo.Log() // repo.Log calls repo.Pull internally
	if err != nil {
		log.Printf("hg log: %v", err)
		return
	}

	// Pass 1.  Fill in parents and add new log entries to logsByHash.
	// Empty parent means take parent from next log entry.
	// Non-empty parent has form 1234:hashhashhash; we want full hash.
	for i := range logs {
		l := &logs[i]
		if l.Parent == "" && i+1 < len(logs) {
			l.Parent = logs[i+1].Hash
		} else if l.Parent != "" {
			l.Parent, _ = repo.FullHash(l.Parent)
		}
		if *verbose {
			log.Printf("hg log %s: %s < %s\n", pkg, l.Hash, l.Parent)
		}
		if logByHash[l.Hash] == nil {
			// Make copy to avoid pinning entire slice when only one entry is new.
			t := *l
			logByHash[t.Hash] = &t
		}
	}

	for _, l := range logs {
		addCommit(pkg, l.Hash, key)
	}
}

// addCommit adds the commit with the named hash to the dashboard.
// key is the secret key for authentication to the dashboard.
// It avoids duplicate effort.
func addCommit(pkg, hash, key string) bool {
	l := logByHash[hash]
	if l == nil {
		return false
	}
	if l.added {
		return true
	}

	// Check for already added, perhaps in an earlier run.
	if dashboardCommit(pkg, hash) {
		log.Printf("%s already on dashboard\n", hash)
		// Record that this hash is on the dashboard,
		// as must be all its parents.
		for l != nil {
			l.added = true
			l = logByHash[l.Parent]
		}
		return true
	}

	// Create parent first, to maintain some semblance of order.
	if l.Parent != "" {
		if !addCommit(pkg, l.Parent, key) {
			return false
		}
	}

	// Create commit.
	if err := postCommit(key, pkg, l); err != nil {
		log.Printf("failed to add %s to dashboard: %v", key, err)
		return false
	}
	return true
}

var repoRe = regexp.MustCompile(`^code\.google\.com/p/([a-z0-9\-]+(\.[a-z0-9\-]+)?)(/[a-z0-9A-Z_.\-/]+)?$`)

// repoURL returns the repository URL for the supplied import path.
func repoURL(importPath string) string {
	m := repoRe.FindStringSubmatch(importPath)
	if len(m) < 2 {
		log.Printf("repoURL: couldn't decipher %q", importPath)
		return ""
	}
	return "https://code.google.com/p/" + m[1]
}

// defaultSuffix returns file extension used for command files in
// current os environment.
func defaultSuffix() string {
	if runtime.GOOS == "windows" {
		return ".bat"
	}
	return ".bash"
}

// defaultBuildRoot returns default buildroot directory.
func defaultBuildRoot() string {
	var d string
	if runtime.GOOS == "windows" {
		// will use c:\, otherwise absolute paths become too long
		// during builder run, see http://golang.org/issue/3358.
		d = `c:\`
	} else {
		d = os.TempDir()
	}
	return filepath.Join(d, "gobuilder")
}

func getenvOk(k string) (v string, ok bool) {
	v = os.Getenv(k)
	if v != "" {
		return v, true
	}
	keq := k + "="
	for _, kv := range os.Environ() {
		if kv == keq {
			return "", true
		}
	}
	return "", false
}
