package main

import (
	"container/vector"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	codeProject  = "go"
	codePyScript = "misc/dashboard/googlecode_upload.py"
	hgUrl        = "https://go.googlecode.com/hg/"
	waitInterval = 10e9 // time to wait before checking for new revs
	mkdirPerm    = 0750
)

type Builder struct {
	name         string
	goos, goarch string
	key          string
	codeUsername string
	codePassword string
}

type BenchRequest struct {
	builder *Builder
	commit  Commit
	path    string
}

var (
	dashboard     = flag.String("dashboard", "godashboard.appspot.com", "Go Dashboard Host")
	runBenchmarks = flag.Bool("bench", false, "Run benchmarks")
	buildRelease  = flag.Bool("release", false, "Build and upload binary release archives")
	buildRevision = flag.String("rev", "", "Build specified revision and exit")
	buildCmd      = flag.String("cmd", "./all.bash", "Build command (specify absolute or relative to go/src/)")
)

var (
	buildroot     = path.Join(os.TempDir(), "gobuilder")
	goroot        = path.Join(buildroot, "goroot")
	releaseRegexp = regexp.MustCompile(`^release\.[0-9\-]+`)
	benchRequests vector.Vector
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
	builders := make([]*Builder, len(flag.Args()))
	for i, builder := range flag.Args() {
		b, err := NewBuilder(builder)
		if err != nil {
			log.Exit(err)
		}
		builders[i] = b
	}
	if err := os.RemoveAll(buildroot); err != nil {
		log.Exitf("Error removing build root (%s): %s", buildroot, err)
	}
	if err := os.Mkdir(buildroot, mkdirPerm); err != nil {
		log.Exitf("Error making build root (%s): %s", buildroot, err)
	}
	if err := run(nil, buildroot, "hg", "clone", hgUrl, goroot); err != nil {
		log.Exit("Error cloning repository:", err)
	}
	// if specified, build revision and return
	if *buildRevision != "" {
		c, err := getCommit(*buildRevision)
		if err != nil {
			log.Exit("Error finding revision:", err)
		}
		for _, b := range builders {
			if err := b.buildCommit(c); err != nil {
				log.Println(err)
			}
			runQueuedBenchmark()
		}
		return
	}
	// check for new commits and build them
	for {
		err := run(nil, goroot, "hg", "pull", "-u")
		if err != nil {
			log.Println("hg pull failed:", err)
			time.Sleep(waitInterval)
			continue
		}
		built := false
		for _, b := range builders {
			if b.build() {
				built = true
			}
		}
		// only run benchmarks if we didn't build anything
		// so that they don't hold up the builder queue
		if !built {
			if !runQueuedBenchmark() {
				// if we have no benchmarks to do, pause
				time.Sleep(waitInterval)
			}
			// after running one benchmark, 
			// continue to find and build new revisions.
		}
	}
}

func runQueuedBenchmark() bool {
	if benchRequests.Len() == 0 {
		return false
	}
	runBenchmark(benchRequests.Pop().(BenchRequest))
	return true
}

func runBenchmark(r BenchRequest) {
	// run benchmarks and send to dashboard
	log.Println(r.builder.name, "benchmarking", r.commit.num)
	defer os.RemoveAll(r.path)
	pkg := path.Join(r.path, "go", "src", "pkg")
	bin := path.Join(r.path, "go", "bin")
	env := []string{
		"GOOS=" + r.builder.goos,
		"GOARCH=" + r.builder.goarch,
		"PATH=" + bin + ":" + os.Getenv("PATH"),
	}
	benchLog, _, err := runLog(env, pkg, "gomake", "bench")
	if err != nil {
		log.Println(r.builder.name, "gomake bench:", err)
		return
	}
	if err = r.builder.recordBenchmarks(benchLog, r.commit); err != nil {
		log.Println("recordBenchmarks:", err)
	}
}

func NewBuilder(builder string) (*Builder, os.Error) {
	b := &Builder{name: builder}

	// get goos/goarch from builder string
	s := strings.Split(builder, "-", 3)
	if len(s) == 2 {
		b.goos, b.goarch = s[0], s[1]
	} else {
		return nil, fmt.Errorf("unsupported builder form: %s", builder)
	}

	// read keys from keyfile
	fn := path.Join(os.Getenv("HOME"), ".gobuildkey")
	if s := fn + "-" + b.name; isFile(s) { // builder-specific file
		fn = s
	}
	c, err := ioutil.ReadFile(fn)
	if err != nil {
		return nil, fmt.Errorf("readKeys %s (%s): %s", b.name, fn, err)
	}
	v := strings.Split(string(c), "\n", -1)
	b.key = v[0]
	if len(v) >= 3 {
		b.codeUsername, b.codePassword = v[1], v[2]
	}

	return b, nil
}

// build checks for a new commit for this builder
// and builds it if one is found. 
// It returns true if a build was attempted.
func (b *Builder) build() bool {
	defer func() {
		err := recover()
		if err != nil {
			log.Println(b.name, "build:", err)
		}
	}()
	c, err := b.nextCommit()
	if err != nil {
		log.Println(err)
		return false
	}
	if c == nil {
		return false
	}
	log.Println(b.name, "building", c.num)
	err = b.buildCommit(*c)
	if err != nil {
		log.Println(err)
	}
	return true
}

// nextCommit returns the next unbuilt Commit for this builder
func (b *Builder) nextCommit() (nextC *Commit, err os.Error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("%s nextCommit: %s", b.name, err)
		}
	}()
	hw, err := b.getHighWater()
	if err != nil {
		return
	}
	c, err := getCommit(hw)
	if err != nil {
		return
	}
	next := c.num + 1
	c, err = getCommit(strconv.Itoa(next))
	if err == nil && c.num == next {
		return &c, nil
	}
	return nil, nil
}

func (b *Builder) buildCommit(c Commit) (err os.Error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("%s buildCommit: %d: %s", b.name, c.num, err)
		}
	}()

	// create place in which to do work
	workpath := path.Join(buildroot, b.name+"-"+strconv.Itoa(c.num))
	err = os.Mkdir(workpath, mkdirPerm)
	if err != nil {
		return
	}
	benchRequested := false
	defer func() {
		if !benchRequested {
			os.RemoveAll(workpath)
		}
	}()

	// clone repo
	err = run(nil, workpath, "hg", "clone", goroot, "go")
	if err != nil {
		return
	}

	// update to specified revision
	err = run(nil, path.Join(workpath, "go"),
		"hg", "update", "-r", strconv.Itoa(c.num))
	if err != nil {
		return
	}

	// set up environment for build/bench execution
	env := []string{
		"GOOS=" + b.goos,
		"GOARCH=" + b.goarch,
		"GOROOT_FINAL=/usr/local/go",
		"PATH=" + os.Getenv("PATH"),
	}
	srcDir := path.Join(workpath, "go", "src")

	// build
	buildLog, status, err := runLog(env, srcDir, *buildCmd)
	if err != nil {
		return fmt.Errorf("all.bash: %s", err)
	}
	if status != 0 {
		// record failure
		return b.recordResult(buildLog, c)
	}

	// record success
	if err = b.recordResult("", c); err != nil {
		return fmt.Errorf("recordResult: %s", err)
	}

	// send benchmark request if benchmarks are enabled
	if *runBenchmarks {
		benchRequests.Insert(0, BenchRequest{
			builder: b,
			commit:  c,
			path:    workpath,
		})
		benchRequested = true
	}

	// finish here if codeUsername and codePassword aren't set
	if b.codeUsername == "" || b.codePassword == "" || !*buildRelease {
		return
	}

	// if this is a release, create tgz and upload to google code
	if release := releaseRegexp.FindString(c.desc); release != "" {
		// clean out build state
		err = run(env, srcDir, "./clean.bash", "--nopkg")
		if err != nil {
			return fmt.Errorf("clean.bash: %s", err)
		}
		// upload binary release
		fn := fmt.Sprintf("%s.%s-%s.tar.gz", release, b.goos, b.goarch)
		err = run(nil, workpath, "tar", "czf", fn, "go")
		if err != nil {
			return fmt.Errorf("tar: %s", err)
		}
		err = run(nil, workpath, "python",
			path.Join(goroot, codePyScript),
			"-s", release,
			"-p", codeProject,
			"-u", b.codeUsername,
			"-w", b.codePassword,
			"-l", fmt.Sprintf("%s,%s", b.goos, b.goarch),
			fn)
	}

	return
}

func isDirectory(name string) bool {
	s, err := os.Stat(name)
	return err == nil && s.IsDirectory()
}

func isFile(name string) bool {
	s, err := os.Stat(name)
	return err == nil && (s.IsRegular() || s.IsSymlink())
}
