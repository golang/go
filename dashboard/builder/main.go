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

	"code.google.com/p/go.tools/go/vcs"
)

const (
	codeProject          = "go"
	codePyScript         = "misc/dashboard/googlecode_upload.py"
	goImportPath         = "code.google.com/p/go"
	gofrontendImportPath = "code.google.com/p/gofrontend"
	mkdirPerm            = 0750
	waitInterval         = 30 * time.Second // time to wait before checking for new revs
	pkgBuildInterval     = 24 * time.Hour   // rebuild packages every 24 hours
)

type Builder struct {
	goroot       *Repo
	name         string
	goos, goarch string
	key          string
	env          builderEnv
}

var (
	buildroot      = flag.String("buildroot", defaultBuildRoot(), "Directory under which to build")
	dashboard      = flag.String("dashboard", "build.golang.org", "Go Dashboard Host")
	buildRelease   = flag.Bool("release", false, "Build and upload binary release archives")
	buildRevision  = flag.String("rev", "", "Build specified revision and exit")
	buildCmd       = flag.String("cmd", filepath.Join(".", allCmd), "Build command (specify relative to go/src/)")
	buildTool      = flag.String("tool", "go", "Tool to build.")
	gccPath        = flag.String("gccpath", "svn://gcc.gnu.org/svn/gcc/trunk", "Path to download gcc from")
	failAll        = flag.Bool("fail", false, "fail all builds")
	parallel       = flag.Bool("parallel", false, "Build multiple targets in parallel")
	buildTimeout   = flag.Duration("buildTimeout", 60*time.Minute, "Maximum time to wait for builds and tests")
	cmdTimeout     = flag.Duration("cmdTimeout", 10*time.Minute, "Maximum time to wait for an external command")
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

	vcs.ShowCmd = *verbose
	vcs.Verbose = *verbose

	rr, err := repoForTool()
	if err != nil {
		log.Fatal("Error finding repository:", err)
	}
	rootPath := filepath.Join(*buildroot, "goroot")
	goroot := &Repo{
		Path:   rootPath,
		Master: rr,
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
		goroot, err = RemoteRepo(goroot.Master.Root, rootPath)
		if err != nil {
			log.Fatalf("Error creating repository with url (%s): %s", goroot.Master.Root, err)
		}

		goroot, err = goroot.Clone(goroot.Path, "tip")
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

	// get builderEnv for this tool
	var err error
	if b.env, err = b.builderEnv(name); err != nil {
		return nil, err
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

// builderEnv returns the builderEnv for this buildTool.
func (b *Builder) builderEnv(name string) (builderEnv, error) {
	// get goos/goarch from builder string
	s := strings.SplitN(b.name, "-", 3)
	if len(s) < 2 {
		return nil, fmt.Errorf("unsupported builder form: %s", name)
	}
	b.goos = s[0]
	b.goarch = s[1]

	switch *buildTool {
	case "go":
		return &goEnv{
			goos:   s[0],
			goarch: s[1],
		}, nil
	case "gccgo":
		return &gccgoEnv{}, nil
	default:
		return nil, fmt.Errorf("unsupported build tool: %s", *buildTool)
	}
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

	// set up builder's environment.
	srcDir, err := b.env.setup(b.goroot, workpath, hash, b.envv())
	if err != nil {
		return err
	}

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

	// go's build command is a script relative to the srcDir, whereas
	// gccgo's build command is usually "make check-go" in the srcDir.
	if *buildTool == "go" {
		if !filepath.IsAbs(cmd) {
			cmd = filepath.Join(srcDir, cmd)
		}
	}

	// make sure commands with extra arguments are handled properly
	splitCmd := strings.Split(cmd, " ")
	startTime := time.Now()
	ok, err := runOutput(*buildTimeout, b.envv(), w, srcDir, splitCmd...)
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

	// build sub-repositories
	goRoot := filepath.Join(workpath, *buildTool)
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
	pkgmaster, err := vcs.RepoRootForImportPath(pkg, *verbose)
	if err != nil {
		return "", fmt.Errorf("Error finding subrepo (%s): %s", pkg, err)
	}
	repo := &Repo{
		Path:   filepath.Join(goPath, "src", pkg),
		Master: pkgmaster,
	}
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

// repoForTool returns the correct RepoRoot for the buildTool, or an error if
// the tool is unknown.
func repoForTool() (*vcs.RepoRoot, error) {
	switch *buildTool {
	case "go":
		return vcs.RepoRootForImportPath(goImportPath, *verbose)
	case "gccgo":
		return vcs.RepoRootForImportPath(gofrontendImportPath, *verbose)
	default:
		return nil, fmt.Errorf("unknown build tool: %s", *buildTool)
	}
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
			pkgmaster, err := vcs.RepoRootForImportPath(pkg, *verbose)
			if err != nil {
				log.Fatalf("Error finding subrepo (%s): %s", pkg, err)
			}
			pkgroot := &Repo{
				Path:   filepath.Join(*buildroot, pkg),
				Master: pkgmaster,
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
	pkgPath := filepath.Join(*buildroot, repo.Master.Root)
	if !repo.Exists() {
		var err error
		repo, err = RemoteRepo(pkg, pkgPath)
		if err != nil {
			log.Printf("Error cloning package (%s): %s", pkg, err)
		}

		repo, err = repo.Clone(repo.Path, "tip")
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

// defaultSuffix returns file extension used for command files in
// current os environment.
func defaultSuffix() string {
	switch runtime.GOOS {
	case "windows":
		return ".bat"
	case "plan9":
		return ".rc"
	default:
		return ".bash"
	}
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
