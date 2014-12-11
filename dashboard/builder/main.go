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
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"golang.org/x/tools/go/vcs"
)

const (
	codeProject          = "go"
	codePyScript         = "misc/dashboard/googlecode_upload.py"
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
	// Last benchmarking workpath. We reuse it, if do successive benchmarks on the same commit.
	lastWorkpath string
}

var (
	doBuild       = flag.Bool("build", true, "Build and test packages")
	doBench       = flag.Bool("bench", false, "Run benchmarks")
	buildroot     = flag.String("buildroot", defaultBuildRoot(), "Directory under which to build")
	dashboard     = flag.String("dashboard", "https://build.golang.org", "Dashboard app base path")
	buildRelease  = flag.Bool("release", false, "Build and upload binary release archives")
	buildRevision = flag.String("rev", "", "Build specified revision and exit")
	buildCmd      = flag.String("cmd", filepath.Join(".", allCmd), "Build command (specify relative to go/src/)")
	buildTool     = flag.String("tool", "go", "Tool to build.")
	gcPath        = flag.String("gcpath", "go.googlesource.com/go", "Path to download gc from")
	gccPath       = flag.String("gccpath", "https://github.com/mirrors/gcc.git", "Path to download gcc from")
	gccOpts       = flag.String("gccopts", "", "Command-line options to pass to `make` when building gccgo")
	benchPath     = flag.String("benchpath", "golang.org/x/benchmarks/bench", "Path to download benchmarks from")
	failAll       = flag.Bool("fail", false, "fail all builds")
	parallel      = flag.Bool("parallel", false, "Build multiple targets in parallel")
	buildTimeout  = flag.Duration("buildTimeout", 60*time.Minute, "Maximum time to wait for builds and tests")
	cmdTimeout    = flag.Duration("cmdTimeout", 10*time.Minute, "Maximum time to wait for an external command")
	benchNum      = flag.Int("benchnum", 5, "Run each benchmark that many times")
	benchTime     = flag.Duration("benchtime", 5*time.Second, "Benchmarking time for a single benchmark run")
	benchMem      = flag.Int("benchmem", 64, "Approx RSS value to aim at in benchmarks, in MB")
	fileLock      = flag.String("filelock", "", "File to lock around benchmaring (synchronizes several builders)")
	verbose       = flag.Bool("v", false, "verbose")
	report        = flag.Bool("report", true, "whether to report results to the dashboard")
)

var (
	binaryTagRe = regexp.MustCompile(`^(release\.r|weekly\.)[0-9\-.]+`)
	releaseRe   = regexp.MustCompile(`^release\.r[0-9\-.]+`)
	allCmd      = "all" + suffix
	makeCmd     = "make" + suffix
	raceCmd     = "race" + suffix
	cleanCmd    = "clean" + suffix
	suffix      = defaultSuffix()
	exeExt      = defaultExeExt()

	benchCPU      = CpuList([]int{1})
	benchAffinity = CpuList([]int{})
	benchMutex    *FileMutex // Isolates benchmarks from other activities
)

// CpuList is used as flag.Value for -benchcpu flag.
type CpuList []int

func (cl *CpuList) String() string {
	str := ""
	for _, cpu := range *cl {
		if str == "" {
			str = strconv.Itoa(cpu)
		} else {
			str += fmt.Sprintf(",%v", cpu)
		}
	}
	return str
}

func (cl *CpuList) Set(str string) error {
	*cl = []int{}
	for _, val := range strings.Split(str, ",") {
		val = strings.TrimSpace(val)
		if val == "" {
			continue
		}
		cpu, err := strconv.Atoi(val)
		if err != nil || cpu <= 0 {
			return fmt.Errorf("%v is a bad value for GOMAXPROCS", val)
		}
		*cl = append(*cl, cpu)
	}
	if len(*cl) == 0 {
		*cl = append(*cl, 1)
	}
	return nil
}

func main() {
	flag.Var(&benchCPU, "benchcpu", "Comma-delimited list of GOMAXPROCS values for benchmarking")
	flag.Var(&benchAffinity, "benchaffinity", "Comma-delimited list of affinity values for benchmarking")
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

	benchMutex = MakeFileMutex(*fileLock)

	rr, err := repoForTool()
	if err != nil {
		log.Fatal("Error finding repository:", err)
	}
	rootPath := filepath.Join(*buildroot, "goroot")
	goroot := &Repo{
		Path:   rootPath,
		Master: rr,
	}

	// set up work environment, use existing environment if possible
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

		goroot, err = goroot.Clone(goroot.Path, "")
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
		var exitErr error
		for _, b := range builders {
			if err := b.buildHash(hash); err != nil {
				log.Println(err)
				exitErr = err
			}
		}
		if exitErr != nil && !*report {
			// This mode (-report=false) is used for
			// testing Docker images, making sure the
			// environment is correctly configured. For
			// testing, we want a non-zero exit status, as
			// returned by log.Fatal:
			log.Fatal("Build error.")
		}
		return
	}

	if !*doBuild && !*doBench {
		fmt.Fprintf(os.Stderr, "Nothing to do, exiting (specify either -build or -bench or both)\n")
		os.Exit(2)
	}

	// go continuous build mode
	// check for new commits and build them
	benchMutex.RLock()
	for {
		built := false
		t := time.Now()
		if *parallel {
			done := make(chan bool)
			for _, b := range builders {
				go func(b *Builder) {
					done <- b.buildOrBench()
				}(b)
			}
			for _ = range builders {
				built = <-done || built
			}
		} else {
			for _, b := range builders {
				built = b.buildOrBench() || built
			}
		}
		// sleep if there was nothing to build
		benchMutex.RUnlock()
		if !built {
			time.Sleep(waitInterval)
		}
		benchMutex.RLock()
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
	if *report {
		err = b.setKey()
	}
	return b, err
}

func (b *Builder) setKey() error {
	// read keys from keyfile
	fn := ""
	switch runtime.GOOS {
	case "plan9":
		fn = os.Getenv("home")
	case "windows":
		fn = os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
	default:
		fn = os.Getenv("HOME")
	}
	fn = filepath.Join(fn, ".gobuildkey")
	if s := fn + "-" + b.name; isFile(s) { // builder-specific file
		fn = s
	}
	c, err := ioutil.ReadFile(fn)
	if err != nil {
		// If the on-disk file doesn't exist, also try the
		// Google Compute Engine metadata.
		if v := gceProjectMetadata("buildkey-" + b.name); v != "" {
			b.key = v
			return nil
		}
		return fmt.Errorf("readKeys %s (%s): %s", b.name, fn, err)
	}
	b.key = string(bytes.TrimSpace(bytes.SplitN(c, []byte("\n"), 2)[0]))
	return nil
}

func gceProjectMetadata(attr string) string {
	client := &http.Client{
		Transport: &http.Transport{
			Dial: (&net.Dialer{
				Timeout:   750 * time.Millisecond,
				KeepAlive: 30 * time.Second,
			}).Dial,
			ResponseHeaderTimeout: 750 * time.Millisecond,
		},
	}
	req, _ := http.NewRequest("GET", "http://metadata.google.internal/computeMetadata/v1/project/attributes/"+attr, nil)
	req.Header.Set("Metadata-Flavor", "Google")
	res, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return ""
	}
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return ""
	}
	return string(bytes.TrimSpace(slurp))
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

// buildOrBench checks for a new commit for this builder
// and builds or benchmarks it if one is found.
// It returns true if a build/benchmark was attempted.
func (b *Builder) buildOrBench() bool {
	var kinds []string
	if *doBuild {
		kinds = append(kinds, "build-go-commit")
	}
	if *doBench {
		kinds = append(kinds, "benchmark-go-commit")
	}
	kind, hash, benchs, err := b.todo(kinds, "", "")
	if err != nil {
		log.Println(err)
		return false
	}
	if hash == "" {
		return false
	}
	switch kind {
	case "build-go-commit":
		if err := b.buildHash(hash); err != nil {
			log.Println(err)
		}
		return true
	case "benchmark-go-commit":
		if err := b.benchHash(hash, benchs); err != nil {
			log.Println(err)
		}
		return true
	default:
		log.Printf("Unknown todo kind %v", kind)
		return false
	}
}

func (b *Builder) buildHash(hash string) error {
	log.Println(b.name, "building", hash)

	// create place in which to do work
	workpath := filepath.Join(*buildroot, b.name+"-"+hash[:12])
	if err := os.Mkdir(workpath, mkdirPerm); err != nil {
		if err2 := removePath(workpath); err2 != nil {
			return err
		}
		if err := os.Mkdir(workpath, mkdirPerm); err != nil {
			return err
		}
	}
	defer removePath(workpath)

	buildLog, runTime, err := b.buildRepoOnHash(workpath, hash, b.buildCmd())
	if err != nil {
		// record failure
		return b.recordResult(false, "", hash, "", buildLog, runTime)
	}

	// record success
	if err = b.recordResult(true, "", hash, "", "", runTime); err != nil {
		return fmt.Errorf("recordResult: %s", err)
	}

	if *buildTool == "go" {
		// build sub-repositories
		goRoot := filepath.Join(workpath, *buildTool)
		goPath := workpath
		b.buildSubrepos(goRoot, goPath, hash)
	}

	return nil
}

// buildRepoOnHash clones repo into workpath and builds it.
func (b *Builder) buildRepoOnHash(workpath, hash, cmd string) (buildLog string, runTime time.Duration, err error) {
	// Delete the previous workdir, if necessary
	// (benchmarking code can execute several benchmarks in the same workpath).
	if b.lastWorkpath != "" {
		if b.lastWorkpath == workpath {
			panic("workpath already exists: " + workpath)
		}
		removePath(b.lastWorkpath)
		b.lastWorkpath = ""
	}

	// pull before cloning to ensure we have the revision
	if err = b.goroot.Pull(); err != nil {
		buildLog = err.Error()
		return
	}

	// set up builder's environment.
	srcDir, err := b.env.setup(b.goroot, workpath, hash, b.envv())
	if err != nil {
		buildLog = err.Error()
		return
	}

	// build
	var buildbuf bytes.Buffer
	logfile := filepath.Join(workpath, "build.log")
	f, err := os.Create(logfile)
	if err != nil {
		return err.Error(), 0, err
	}
	defer f.Close()
	w := io.MultiWriter(f, &buildbuf)

	// go's build command is a script relative to the srcDir, whereas
	// gccgo's build command is usually "make check-go" in the srcDir.
	if *buildTool == "go" {
		if !filepath.IsAbs(cmd) {
			cmd = filepath.Join(srcDir, cmd)
		}
	}

	// naive splitting of command from its arguments:
	args := strings.Split(cmd, " ")
	c := exec.Command(args[0], args[1:]...)
	c.Dir = srcDir
	c.Env = b.envv()
	if *verbose {
		c.Stdout = io.MultiWriter(os.Stdout, w)
		c.Stderr = io.MultiWriter(os.Stderr, w)
	} else {
		c.Stdout = w
		c.Stderr = w
	}

	startTime := time.Now()
	err = run(c, runTimeout(*buildTimeout))
	runTime = time.Since(startTime)
	if err != nil {
		fmt.Fprintf(w, "Build complete, duration %v. Result: error: %v\n", runTime, err)
	} else {
		fmt.Fprintf(w, "Build complete, duration %v. Result: success\n", runTime)
	}
	return buildbuf.String(), runTime, err
}

// failBuild checks for a new commit for this builder
// and fails it if one is found.
// It returns true if a build was "attempted".
func (b *Builder) failBuild() bool {
	_, hash, _, err := b.todo([]string{"build-go-commit"}, "", "")
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
		_, hash, _, err := b.todo([]string{"build-package"}, pkg, goHash)
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
	goTool := filepath.Join(goRoot, "bin", "go") + exeExt
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

	// HACK: check out to new sub-repo location instead of old location.
	pkg = strings.Replace(pkg, "code.google.com/p/go.", "golang.org/x/", 1)

	// fetch package and dependencies
	var outbuf bytes.Buffer
	err := run(exec.Command(goTool, "get", "-d", pkg+"/..."), runEnv(env), allOutput(&outbuf), runDir(goPath))
	if err != nil {
		return outbuf.String(), err
	}
	outbuf.Reset()

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
	err = run(exec.Command(goTool, "test", "-short", pkg+"/..."),
		runTimeout(*buildTimeout), runEnv(env), allOutput(&outbuf), runDir(goPath))
	return outbuf.String(), err
}

// repoForTool returns the correct RepoRoot for the buildTool, or an error if
// the tool is unknown.
func repoForTool() (*vcs.RepoRoot, error) {
	switch *buildTool {
	case "go":
		return vcs.RepoRootForImportPath(*gcPath, *verbose)
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

func defaultExeExt() string {
	switch runtime.GOOS {
	case "windows":
		return ".exe"
	default:
		return ""
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

// removePath is a more robust version of os.RemoveAll.
// On windows, if remove fails (which can happen if test/benchmark timeouts
// and keeps some files open) it tries to rename the dir.
func removePath(path string) error {
	if err := os.RemoveAll(path); err != nil {
		if runtime.GOOS == "windows" {
			err = os.Rename(path, filepath.Clean(path)+"_remove_me")
		}
		return err
	}
	return nil
}
