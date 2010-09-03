package main

import (
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
)

var (
	goroot        = path.Join(os.Getenv("PWD"), "goroot")
	releaseRegexp = regexp.MustCompile(`^release\.[0-9\-]+`)
	dashboardhost = flag.String("dashboard", "godashboard.appspot.com",
		"Godashboard Host")
)

type Builder struct {
	name         string
	goos, goarch string
	key          string
	codeUsername string
	codePassword string
}

func main() {
	flag.Parse()
	builders := make(map[string]*Builder)
	if len(flag.Args()) == 0{
		log.Exit("No builders specified.")
	}
	for _, builder := range flag.Args() {
		b, err := NewBuilder(builder)
		if err != nil {
			log.Exit(err)
		}
		builders[builder] = b
	}
	err := run(nil, "", "hg", "clone", hgUrl, goroot)
	if err != nil {
		log.Exit("Error cloning repository:", err)
	}
	// check for new commits and build them
	for {
		err := run(nil, goroot, "hg", "pull", "-u")
		if err != nil {
			log.Stderr("hg pull failed:", err)
			time.Sleep(waitInterval)
			continue
		}
		built := false
		for _, b := range builders {
			built = b.tryBuild() || built
		}
		// only wait if we didn't do anything
		if !built {
			time.Sleep(waitInterval)
		}
	}
}

func NewBuilder(builder string) (*Builder, os.Error) {
	b := &Builder{name:builder}

	// get goos/goarch from builder string
	s := strings.Split(builder, "-", 3)
	if len(s) == 2 {
		b.goos, b.goarch = s[0], s[1]
	} else {
		return nil, os.NewError(fmt.Sprintf(
			"unsupported builder form: %s", builder))
	}

	// read keys from keyfile
	fn := path.Join(os.Getenv("HOME"), ".gobuildkey")
	if isFile(fn + "-" + b.name) { // builder-specific file
		fn += "-" + b.name
	}
	c, err := ioutil.ReadFile(fn)
	if err != nil {
		return nil, os.NewError(fmt.Sprintf("readKeys %s (%s): %s",
			b.name, fn, err))
	}
	v := strings.Split(string(c), "\n", -1)
	b.key = v[0]
	if len(v) >= 3 {
		b.codeUsername, b.codePassword = v[1], v[2]
	}

	return b, nil
}

// tryBuild checks for a new commit for this builder, 
// and builds it if one is found. 
// Its return value indicates whether a build happened or not.
func (b *Builder) tryBuild() bool {
	c, err := b.nextCommit()
	if err != nil {
		log.Stderr(err)
		return false
	}
	if c == nil {
		return false
	}
	log.Stderr("Building new revision: ", c.num)
	err = b.build(*c)
	if err != nil {
		log.Stderr(err)
	}
	return true
}

// nextCommit returns the next unbuilt Commit for this builder
func (b *Builder) nextCommit() (nextC *Commit, err os.Error) {
	defer func() {
		if err != nil {
			err = os.NewError(fmt.Sprintf(
				"%s nextCommit: %s", b.name, err))
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
	if err == nil || c.num == next {
		return &c, nil
	}
	return nil, nil
}

func (b *Builder) build(c Commit) (err os.Error) {
	defer func() {
		if err != nil {
			err = os.NewError(fmt.Sprintf(
				"%s buildRev commit: %s: %s",
				b.name, c.num, err))
		}
	}()

	// destroy old build candidate
	err = run(nil, "", "rm", "-Rf", "go")
	if err != nil {
		return
	}

	// clone repo at revision num (new candidate)
	err = run(nil, "", 
		"hg", "clone", 
		"-r", strconv.Itoa(c.num), 
		goroot, "go")
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
	srcDir := path.Join("", "go", "src")

	// build the release candidate
	buildLog, status, err := runLog(env, srcDir, "./all.bash")
	if err != nil {
		return
	}
	if status != 0 {
		// record failure
		return b.recordResult(buildLog, c)
	}

	// record success
	if err = b.recordResult("", c); err != nil {
		return
	}

	// run benchmarks and send to dashboard
	pkgDir := path.Join(srcDir, "pkg")
	benchLog, _, err := runLog(env, pkgDir, "gomake", "bench")
	if err != nil {
		log.Stderr("gomake bench:", err)
	} else if err = b.recordBenchmarks(benchLog, c); err != nil {
		log.Stderr("recordBenchmarks:", err)
	}

	// finish here if codeUsername and codePassword aren't set
	if b.codeUsername == "" || b.codePassword == "" {
		return
	}

	// if this is a release, create tgz and upload to google code
	if release := releaseRegexp.FindString(c.desc); release != "" {
		// clean out build state
		err = run(env, srcDir, "sh", "clean.bash", "--nopkg")
		if err != nil {
			return
		}
		// upload binary release
		err = b.codeUpload(release)
		if err != nil {
			return
		}
	}

	return
}

func (b *Builder) codeUpload(release string) (err os.Error) {
	defer func() {
		if err != nil {
			err = os.NewError(fmt.Sprintf(
				"%s codeUpload release: %s: %s",
				b.name, release, err))
		}
	}()
	fn := fmt.Sprintf("%s.%s-%s.tar.gz", release, b.goos, b.goarch)
	err = run(nil, "", "tar", "czf", fn, "go")
	if err != nil {
		return
	}
	return run(nil, "", "python", 
		path.Join(goroot, codePyScript),
		"-s", release, 
		"-p", codeProject,
		"-u", b.codeUsername, 
		"-w", b.codePassword,
		"-l", fmt.Sprintf("%s,%s", b.goos, b.goarch),
		fn)
}

func isDirectory(name string) bool {
	s, err := os.Stat(name)
	return err == nil && s.IsDirectory()
}

func isFile(name string) bool {
	s, err := os.Stat(name)
	return err == nil && (s.IsRegular() || s.IsSymlink())
}
