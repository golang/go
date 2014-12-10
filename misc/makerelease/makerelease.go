// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a tool for packaging binary releases.
// It supports FreeBSD, Linux, NetBSD, OpenBSD, OS X, and Windows.
package main

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha1"
	"encoding/json"
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
	"regexp"
	"runtime"
	"strings"

	"code.google.com/p/goauth2/oauth"
	storage "code.google.com/p/google-api-go-client/storage/v1"
)

var (
	tag             = flag.String("tag", "release", "mercurial tag to check out")
	toolTag         = flag.String("tool", defaultToolTag, "go.tools tag to check out")
	tourTag         = flag.String("tour", defaultTourTag, "go-tour tag to check out")
	repo            = flag.String("repo", "https://code.google.com/p/go", "repo URL")
	verbose         = flag.Bool("v", false, "verbose output")
	upload          = flag.Bool("upload", false, "upload resulting files to Google Code")
	addLabel        = flag.String("label", "", "additional label to apply to file when uploading")
	includeRace     = flag.Bool("race", true, "build race detector packages")
	versionOverride = flag.String("version", "", "override version name")
	staticToolchain = flag.Bool("static", true, "try to build statically linked toolchain (only supported on ELF targets)")
	tokenCache      = flag.String("token", defaultCacheFile, "Authentication token cache file")
	storageBucket   = flag.String("bucket", "golang", "Cloud Storage Bucket")
	uploadURL       = flag.String("upload_url", defaultUploadURL, "Upload URL")

	defaultCacheFile = filepath.Join(os.Getenv("HOME"), ".makerelease-request-token")
	defaultUploadURL = "http://golang.org/dl/upload"
)

const (
	blogPath       = "golang.org/x/blog"
	toolPath       = "golang.org/x/tools"
	tourPath       = "code.google.com/p/go-tour"
	defaultToolTag = "release-branch.go1.4"
	defaultTourTag = "release-branch.go1.4"
)

// Import paths for tool commands.
// These must be the command that cmd/go knows to install to $GOROOT/bin
// or $GOROOT/pkg/tool.
var toolPaths = []string{
	"golang.org/x/tools/cmd/cover",
	"golang.org/x/tools/cmd/godoc",
	"golang.org/x/tools/cmd/vet",
}

var preBuildCleanFiles = []string{
	"lib/codereview",
	"misc/dashboard/godashboard",
	"src/cmd/cov",
	"src/cmd/prof",
	"src/exp",
	"src/old",
}

var cleanFiles = []string{
	".hg",
	".hgtags",
	".hgignore",
	"VERSION.cache",
}

var sourceCleanFiles = []string{
	"bin",
	"pkg",
}

var tourPackages = []string{
	"pic",
	"tree",
	"wc",
}

var tourContent = []string{
	"content",
	"solutions",
	"static",
	"template",
}

var blogContent = []string{
	"content",
	"template",
}

// The os-arches that support the race toolchain.
var raceAvailable = []string{
	"darwin-amd64",
	"linux-amd64",
	"windows-amd64",
}

// The OSes that support building statically linked toolchain
// Only ELF platforms are supported.
var staticLinkAvailable = []string{
	"linux",
	"freebsd",
	"openbsd",
	"netbsd",
}

var fileRe = regexp.MustCompile(`^(go[a-z0-9-.]+)\.(src|([a-z0-9]+)-([a-z0-9]+)(?:-([a-z0-9.]+))?)\.(tar\.gz|zip|pkg|msi)$`)

// OAuth2-authenticated HTTP client used to make calls to Cloud Storage.
var oauthClient *http.Client

// Builder key as specified in ~/.gobuildkey
var builderKey string

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s [flags] targets...\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(2)
	}
	flag.Parse()
	if flag.NArg() == 0 {
		flag.Usage()
	}
	if runtime.GOOS == "windows" {
		checkWindowsDeps()
	}

	if *upload {
		if err := readCredentials(); err != nil {
			log.Fatalln("readCredentials:", err)
		}
		if err := setupOAuthClient(); err != nil {
			log.Fatalln("setupOAuthClient:", err)
		}
	}
	ok := true
	for _, targ := range flag.Args() {
		var b Build
		if m := fileRe.FindStringSubmatch(targ); m != nil {
			// targ is a file name; upload it to googlecode.
			version := m[1]
			if m[2] == "src" {
				b.Source = true
			} else {
				b.OS = m[3]
				b.Arch = m[4]
				b.Label = m[5]
			}
			if !*upload {
				log.Printf("%s: -upload=false, skipping", targ)
				continue
			}
			if err := b.Upload(version, targ); err != nil {
				log.Printf("uploading %s: %v", targ, err)
			}
			continue
		}
		if targ == "source" {
			b.Source = true
		} else {
			p := strings.SplitN(targ, "-", 3)
			if len(p) < 2 {
				log.Println("Ignoring unrecognized target:", targ)
				continue
			}
			b.OS = p[0]
			b.Arch = p[1]
			if len(p) >= 3 {
				b.Label = p[2]
			}
			if *includeRace {
				for _, t := range raceAvailable {
					if t == targ || strings.HasPrefix(targ, t+"-") {
						b.Race = true
					}
				}
			}
			if *staticToolchain {
				for _, os := range staticLinkAvailable {
					if b.OS == os {
						b.static = true
					}
				}
			}
		}
		if err := b.Do(); err != nil {
			log.Printf("%s: %v", targ, err)
			ok = false
		}
	}
	if !ok {
		os.Exit(1)
	}
}

type Build struct {
	Source bool // if true, OS and Arch must be empty
	Race   bool // build race toolchain
	OS     string
	Arch   string
	Label  string
	root   string
	gopath string
	static bool // if true, build statically linked toolchain
}

func (b *Build) Do() error {
	work, err := ioutil.TempDir("", "makerelease")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	b.root = filepath.Join(work, "go")
	b.gopath = work

	// Clone Go distribution and update to tag.
	_, err = b.hgCmd(work, "clone", *repo, b.root)
	if err != nil {
		return err
	}
	_, err = b.hgCmd(b.root, "update", *tag)
	if err != nil {
		return err
	}

	// Remove exp and old packages.
	if err := b.clean(preBuildCleanFiles); err != nil {
		return err
	}

	src := filepath.Join(b.root, "src")
	if b.Source {
		if runtime.GOOS == "windows" {
			log.Print("Warning: running make.bash on Windows; source builds are intended to be run on a Unix machine")
		}
		// Build dist tool only.
		_, err = b.run(src, "bash", "make.bash", "--dist-tool")
	} else {
		// Build.
		if b.OS == "windows" {
			_, err = b.run(src, "cmd", "/C", "make.bat")
		} else {
			_, err = b.run(src, "bash", "make.bash")
		}
		if b.Race {
			if err != nil {
				return err
			}
			goCmd := filepath.Join(b.root, "bin", "go")
			if b.OS == "windows" {
				goCmd += ".exe"
			}
			_, err = b.run(src, goCmd, "install", "-race", "std")
			if err != nil {
				return err
			}
			// Re-install std without -race, so that we're not left
			// with a slower, race-enabled cmd/go, etc.
			_, err = b.run(src, goCmd, "install", "-a", "std")
			// Re-building go command leaves old versions of go.exe as go.exe~ on windows.
			// See (*builder).copyFile in $GOROOT/src/cmd/go/build.go for details.
			// Remove it manually.
			if b.OS == "windows" {
				os.Remove(goCmd + "~")
			}
		}
		if err != nil {
			return err
		}
		err = b.extras()
	}
	if err != nil {
		return err
	}

	// Get version strings.
	var (
		version     string // "weekly.2012-03-04"
		fullVersion []byte // "weekly.2012-03-04 9353aa1efdf3"
	)
	pat := filepath.Join(b.root, "pkg/tool/*/dist*") // trailing * for .exe
	m, err := filepath.Glob(pat)
	if err != nil {
		return err
	}
	if len(m) == 0 {
		return fmt.Errorf("couldn't find dist in %q", pat)
	}
	fullVersion, err = b.run("", m[0], "version")
	if err != nil {
		return err
	}
	fullVersion = bytes.TrimSpace(fullVersion)
	v := bytes.SplitN(fullVersion, []byte(" "), 2)
	version = string(v[0])
	if *versionOverride != "" {
		version = *versionOverride
	}

	// Write VERSION file.
	err = ioutil.WriteFile(filepath.Join(b.root, "VERSION"), fullVersion, 0644)
	if err != nil {
		return err
	}

	// Clean goroot.
	if err := b.clean(cleanFiles); err != nil {
		return err
	}
	if b.Source {
		if err := b.clean(sourceCleanFiles); err != nil {
			return err
		}
	}

	// Create packages.
	base := fmt.Sprintf("%s.%s-%s", version, b.OS, b.Arch)
	if b.Label != "" {
		base += "-" + b.Label
	}
	if !strings.HasPrefix(base, "go") {
		base = "go." + base
	}
	var targs []string
	switch b.OS {
	case "linux", "freebsd", "netbsd", "":
		// build tarball
		targ := base
		if b.Source {
			targ = fmt.Sprintf("%s.src", version)
			if !strings.HasPrefix(targ, "go") {
				targ = "go." + targ
			}
		}
		targ += ".tar.gz"
		err = makeTar(targ, work)
		targs = append(targs, targ)
	case "darwin":
		// build tarball
		targ := base + ".tar.gz"
		err = makeTar(targ, work)
		targs = append(targs, targ)

		makerelease := filepath.Join(runtime.GOROOT(), "misc/makerelease")

		// build pkg
		// arrange work so it's laid out as the dest filesystem
		etc := filepath.Join(makerelease, "darwin/etc")
		_, err = b.run(work, "cp", "-r", etc, ".")
		if err != nil {
			return err
		}
		localDir := filepath.Join(work, "usr/local")
		err = os.MkdirAll(localDir, 0755)
		if err != nil {
			return err
		}
		_, err = b.run(work, "mv", "go", localDir)
		if err != nil {
			return err
		}
		// build package
		pkgdest, err := ioutil.TempDir("", "pkgdest")
		if err != nil {
			return err
		}
		defer os.RemoveAll(pkgdest)
		_, err = b.run("", "pkgbuild",
			"--identifier", "com.googlecode.go",
			"--version", version,
			"--scripts", filepath.Join(makerelease, "darwin/scripts"),
			"--root", work,
			filepath.Join(pkgdest, "com.googlecode.go.pkg"))
		if err != nil {
			return err
		}
		targ = base + ".pkg"
		_, err = b.run("", "productbuild",
			"--distribution", filepath.Join(makerelease, "darwin/Distribution"),
			"--resources", filepath.Join(makerelease, "darwin/Resources"),
			"--package-path", pkgdest,
			targ)
		if err != nil {
			return err
		}
		targs = append(targs, targ)
	case "windows":
		// Create ZIP file.
		zip := filepath.Join(work, base+".zip")
		err = makeZip(zip, work)
		// Copy zip to target file.
		targ := base + ".zip"
		err = cp(targ, zip)
		if err != nil {
			return err
		}
		targs = append(targs, targ)

		// Create MSI installer.
		win := filepath.Join(runtime.GOROOT(), "misc/makerelease/windows")
		installer := filepath.Join(win, "installer.wxs")
		appfiles := filepath.Join(work, "AppFiles.wxs")
		msi := filepath.Join(work, "installer.msi")
		// Gather files.
		_, err = b.run(work, "heat", "dir", "go",
			"-nologo",
			"-gg", "-g1", "-srd", "-sfrag",
			"-cg", "AppFiles",
			"-template", "fragment",
			"-dr", "INSTALLDIR",
			"-var", "var.SourceDir",
			"-out", appfiles)
		if err != nil {
			return err
		}
		// Build package.
		_, err = b.run(work, "candle",
			"-nologo",
			"-dGoVersion="+version,
			"-dWixGoVersion="+wixVersion(version),
			"-dArch="+b.Arch,
			"-dSourceDir=go",
			installer, appfiles)
		if err != nil {
			return err
		}
		appfiles = filepath.Join(work, "AppFiles.wixobj")
		installer = filepath.Join(work, "installer.wixobj")
		_, err = b.run(win, "light",
			"-nologo",
			"-ext", "WixUIExtension",
			"-ext", "WixUtilExtension",
			installer, appfiles,
			"-o", msi)
		if err != nil {
			return err
		}
		// Copy installer to target file.
		targ = base + ".msi"
		err = cp(targ, msi)
		targs = append(targs, targ)
	}
	if err == nil && *upload {
		for _, targ := range targs {
			err = b.Upload(version, targ)
			if err != nil {
				return fmt.Errorf("uploading %s: %v", targ, err)
			}
		}
	}
	return err
}

var versionRe = regexp.MustCompile(`^go([0-9]+(\.[0-9]+)*)`)

// The Microsoft installer requires version format major.minor.build
// (http://msdn.microsoft.com/en-us/library/aa370859%28v=vs.85%29.aspx).
// Where the major and minor field has a maximum value of 255 and build 65535.
// The offical Go version format is goMAJOR.MINOR.PATCH at $GOROOT/VERSION.
// It's based on the Mercurial tag. Remove prefix and suffix to make the
// installer happy.
func wixVersion(v string) string {
	m := versionRe.FindStringSubmatch(v)
	if m == nil {
		return "0.0.0"
	}
	return m[1]
}

// extras fetches the go.tools, go.blog, and go-tour repositories,
// builds them and copies the resulting binaries and static assets
// to the new GOROOT.
func (b *Build) extras() error {
	defer b.cleanGopath()

	if err := b.tools(); err != nil {
		return err
	}
	if err := b.blog(); err != nil {
		return err
	}
	return b.tour()
}

func (b *Build) get(repoPath, revision string) error {
	dest := filepath.Join(b.gopath, "src", filepath.FromSlash(repoPath))

	if strings.HasPrefix(repoPath, "golang.org/x/") {
		// For sub-repos, fetch the old Mercurial repo; bypass "go get".
		// DO NOT import this special case into the git tree.

		if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
			return err
		}
		repo := strings.Replace(repoPath, "golang.org/x/", "https://code.google.com/p/go.", 1)
		if _, err := b.run(b.gopath, "hg", "clone", repo, dest); err != nil {
			return err
		}
	} else {
		// Fetch the packages (without building/installing).
		_, err := b.run(b.gopath, filepath.Join(b.root, "bin", "go"),
			"get", "-d", repoPath+"/...")
		if err != nil {
			return err
		}
	}

	// Update the repo to the specified revision.
	dest := filepath.Join(b.gopath, "src", filepath.FromSlash(repoPath))
	var err error
	switch {
	case exists(filepath.Join(dest, ".git")):
		_, err = b.run(dest, "git", "checkout", revision)
	case exists(filepath.Join(dest, ".hg")):
		_, err = b.run(dest, "hg", "update", revision)
	default:
		err = errors.New("unknown version control system")
	}
	return err
}

func (b *Build) tools() error {
	// Fetch the go.tools repository.
	if err := b.get(toolPath, *toolTag); err != nil {
		return err
	}

	// Install tools.
	args := append([]string{"install"}, toolPaths...)
	_, err := b.run(b.gopath, filepath.Join(b.root, "bin", "go"), args...)
	if err != nil {
		return err
	}

	// Copy doc.go from go.tools/cmd/$CMD to $GOROOT/src/cmd/$CMD
	// while rewriting "package main" to "package documentation".
	for _, p := range toolPaths {
		d, err := ioutil.ReadFile(filepath.Join(b.gopath, "src",
			filepath.FromSlash(p), "doc.go"))
		if err != nil {
			return err
		}
		d = bytes.Replace(d, []byte("\npackage main\n"),
			[]byte("\npackage documentation\n"), 1)
		cmdDir := filepath.Join(b.root, "src", "cmd", path.Base(p))
		if err := os.MkdirAll(cmdDir, 0755); err != nil {
			return err
		}
		docGo := filepath.Join(cmdDir, "doc.go")
		if err := ioutil.WriteFile(docGo, d, 0644); err != nil {
			return err
		}
	}

	return nil
}

func (b *Build) blog() error {
	// Fetch the blog repository.
	_, err := b.run(b.gopath, filepath.Join(b.root, "bin", "go"), "get", "-d", blogPath+"/blog")
	if err != nil {
		return err
	}

	// Copy blog content to $GOROOT/blog.
	blogSrc := filepath.Join(b.gopath, "src", filepath.FromSlash(blogPath))
	contentDir := filepath.Join(b.root, "blog")
	return cpAllDir(contentDir, blogSrc, blogContent...)
}

func (b *Build) tour() error {
	// Fetch the go-tour repository.
	if err := b.get(tourPath, *tourTag); err != nil {
		return err
	}

	// Build tour binary.
	_, err := b.run(b.gopath, filepath.Join(b.root, "bin", "go"),
		"install", tourPath+"/gotour")
	if err != nil {
		return err
	}

	// Copy all the tour content to $GOROOT/misc/tour.
	importPath := filepath.FromSlash(tourPath)
	tourSrc := filepath.Join(b.gopath, "src", importPath)
	contentDir := filepath.Join(b.root, "misc", "tour")
	if err = cpAllDir(contentDir, tourSrc, tourContent...); err != nil {
		return err
	}

	// Copy the tour source code so it's accessible with $GOPATH pointing to $GOROOT/misc/tour.
	if err = cpAllDir(filepath.Join(contentDir, "src", importPath), tourSrc, tourPackages...); err != nil {
		return err
	}

	// Copy gotour binary to tool directory as "tour"; invoked as "go tool tour".
	return cp(
		filepath.Join(b.root, "pkg", "tool", b.OS+"_"+b.Arch, "tour"+ext()),
		filepath.Join(b.gopath, "bin", "gotour"+ext()),
	)
}

func (b *Build) cleanGopath() {
	for _, d := range []string{"bin", "pkg", "src"} {
		os.RemoveAll(filepath.Join(b.gopath, d))
	}
}

func ext() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

func (b *Build) hgCmd(dir string, args ...string) ([]byte, error) {
	return b.run(dir, "hg", append([]string{"--config", "extensions.codereview=!"}, args...)...)
}

func (b *Build) run(dir, name string, args ...string) ([]byte, error) {
	buf := new(bytes.Buffer)
	absName, err := lookPath(name)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(absName, args...)
	var output io.Writer = buf
	if *verbose {
		log.Printf("Running %q %q", absName, args)
		output = io.MultiWriter(buf, os.Stdout)
	}
	cmd.Stdout = output
	cmd.Stderr = output
	cmd.Dir = dir
	cmd.Env = b.env()
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s", buf.Bytes())
		return nil, fmt.Errorf("%s %s: %v", name, strings.Join(args, " "), err)
	}
	return buf.Bytes(), nil
}

var cleanEnv = []string{
	"GOARCH",
	"GOBIN",
	"GOHOSTARCH",
	"GOHOSTOS",
	"GOOS",
	"GOROOT",
	"GOROOT_FINAL",
	"GOPATH",
}

func (b *Build) env() []string {
	env := os.Environ()
	for i := 0; i < len(env); i++ {
		for _, c := range cleanEnv {
			if strings.HasPrefix(env[i], c+"=") {
				env = append(env[:i], env[i+1:]...)
			}
		}
	}
	final := "/usr/local/go"
	if b.OS == "windows" {
		final = `c:\go`
	}
	env = append(env,
		"GOARCH="+b.Arch,
		"GOHOSTARCH="+b.Arch,
		"GOHOSTOS="+b.OS,
		"GOOS="+b.OS,
		"GOROOT="+b.root,
		"GOROOT_FINAL="+final,
		"GOPATH="+b.gopath,
	)
	if b.static {
		env = append(env, "GO_DISTFLAGS=-s")
	}
	return env
}

func (b *Build) Upload(version string, filename string) error {
	file, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	svc, err := storage.New(oauthClient)
	if err != nil {
		return err
	}
	obj := &storage.Object{
		Acl:  []*storage.ObjectAccessControl{{Entity: "allUsers", Role: "READER"}},
		Name: filename,
	}
	_, err = svc.Objects.Insert(*storageBucket, obj).Media(bytes.NewReader(file)).Do()
	if err != nil {
		return err
	}

	sum := fmt.Sprintf("%x", sha1.Sum(file))
	kind := "unknown"
	switch {
	case b.Source:
		kind = "source"
	case strings.HasSuffix(filename, ".tar.gz"), strings.HasSuffix(filename, ".zip"):
		kind = "archive"
	case strings.HasSuffix(filename, ".msi"), strings.HasSuffix(filename, ".pkg"):
		kind = "installer"
	}
	req, err := json.Marshal(File{
		Filename: filename,
		Version:  version,
		OS:       b.OS,
		Arch:     b.Arch,
		Checksum: sum,
		Kind:     kind,
	})
	if err != nil {
		return err
	}
	u := fmt.Sprintf("%s?%s", *uploadURL, url.Values{"key": []string{builderKey}}.Encode())
	resp, err := http.Post(u, "application/json", bytes.NewReader(req))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("upload status: %v", resp.Status)
	}

	return nil
}

type File struct {
	Filename string
	OS       string
	Arch     string
	Version  string
	Checksum string `datastore:",noindex"`
	Kind     string // "archive", "installer", "source"
}

func setupOAuthClient() error {
	config := &oauth.Config{
		ClientId:     "999119582588-h7kpj5pcm6d9solh5lgrbusmvvk4m9dn.apps.googleusercontent.com",
		ClientSecret: "8YLFgOhXIELWbO-NtF3iqIQz",
		Scope:        storage.DevstorageRead_writeScope,
		AuthURL:      "https://accounts.google.com/o/oauth2/auth",
		TokenURL:     "https://accounts.google.com/o/oauth2/token",
		TokenCache:   oauth.CacheFile(*tokenCache),
		RedirectURL:  "oob",
	}
	transport := &oauth.Transport{Config: config}
	if token, err := config.TokenCache.Token(); err != nil {
		url := transport.Config.AuthCodeURL("")
		fmt.Println("Visit the following URL, obtain an authentication" +
			"code, and enter it below.")
		fmt.Println(url)
		fmt.Print("Enter authentication code: ")
		code := ""
		if _, err := fmt.Scan(&code); err != nil {
			return err
		}
		if _, err := transport.Exchange(code); err != nil {
			return err
		}
	} else {
		transport.Token = token
	}
	oauthClient = transport.Client()
	return nil
}

func (b *Build) clean(files []string) error {
	for _, name := range files {
		err := os.RemoveAll(filepath.Join(b.root, name))
		if err != nil {
			return err
		}
	}
	return nil
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func readCredentials() error {
	name := os.Getenv("HOME")
	if runtime.GOOS == "windows" {
		name = os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
	}
	name = filepath.Join(name, ".gobuildkey")
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	if s.Scan() {
		builderKey = s.Text()
	}
	return s.Err()
}

func cp(dst, src string) error {
	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()
	fi, err := sf.Stat()
	if err != nil {
		return err
	}
	df, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer df.Close()
	// Windows doesn't currently implement Fchmod
	if runtime.GOOS != "windows" {
		if err := df.Chmod(fi.Mode()); err != nil {
			return err
		}
	}
	_, err = io.Copy(df, sf)
	return err
}

func cpDir(dst, src string) error {
	walk := func(srcPath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		dstPath := filepath.Join(dst, srcPath[len(src):])
		if info.IsDir() {
			return os.MkdirAll(dstPath, 0755)
		}
		return cp(dstPath, srcPath)
	}
	return filepath.Walk(src, walk)
}

func cpAllDir(dst, basePath string, dirs ...string) error {
	for _, dir := range dirs {
		if err := cpDir(filepath.Join(dst, dir), filepath.Join(basePath, dir)); err != nil {
			return err
		}
	}
	return nil
}

func makeTar(targ, workdir string) error {
	f, err := os.Create(targ)
	if err != nil {
		return err
	}
	zout := gzip.NewWriter(f)
	tw := tar.NewWriter(zout)

	err = filepath.Walk(workdir, func(path string, fi os.FileInfo, err error) error {
		if !strings.HasPrefix(path, workdir) {
			log.Panicf("walked filename %q doesn't begin with workdir %q", path, workdir)
		}
		name := path[len(workdir):]

		// Chop of any leading / from filename, leftover from removing workdir.
		if strings.HasPrefix(name, "/") {
			name = name[1:]
		}
		// Don't include things outside of the go subdirectory (for instance,
		// the zip file that we're currently writing here.)
		if !strings.HasPrefix(name, "go/") {
			return nil
		}
		if *verbose {
			log.Printf("adding to tar: %s", name)
		}
		target, _ := os.Readlink(path)
		hdr, err := tar.FileInfoHeader(fi, target)
		if err != nil {
			return err
		}
		hdr.Name = name
		hdr.Uname = "root"
		hdr.Gname = "root"
		hdr.Uid = 0
		hdr.Gid = 0

		// Force permissions to 0755 for executables, 0644 for everything else.
		if fi.Mode().Perm()&0111 != 0 {
			hdr.Mode = hdr.Mode&^0777 | 0755
		} else {
			hdr.Mode = hdr.Mode&^0777 | 0644
		}

		err = tw.WriteHeader(hdr)
		if err != nil {
			return fmt.Errorf("Error writing file %q: %v", name, err)
		}
		if fi.IsDir() {
			return nil
		}
		r, err := os.Open(path)
		if err != nil {
			return err
		}
		defer r.Close()
		_, err = io.Copy(tw, r)
		return err
	})
	if err != nil {
		return err
	}
	if err := tw.Close(); err != nil {
		return err
	}
	if err := zout.Close(); err != nil {
		return err
	}
	return f.Close()
}

func makeZip(targ, workdir string) error {
	f, err := os.Create(targ)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)

	err = filepath.Walk(workdir, func(path string, fi os.FileInfo, err error) error {
		if !strings.HasPrefix(path, workdir) {
			log.Panicf("walked filename %q doesn't begin with workdir %q", path, workdir)
		}
		name := path[len(workdir):]

		// Convert to Unix-style named paths, as that's the
		// type of zip file that archive/zip creates.
		name = strings.Replace(name, "\\", "/", -1)
		// Chop of any leading / from filename, leftover from removing workdir.
		if strings.HasPrefix(name, "/") {
			name = name[1:]
		}
		// Don't include things outside of the go subdirectory (for instance,
		// the zip file that we're currently writing here.)
		if !strings.HasPrefix(name, "go/") {
			return nil
		}
		if *verbose {
			log.Printf("adding to zip: %s", name)
		}
		fh, err := zip.FileInfoHeader(fi)
		if err != nil {
			return err
		}
		fh.Name = name
		fh.Method = zip.Deflate
		if fi.IsDir() {
			fh.Name += "/"        // append trailing slash
			fh.Method = zip.Store // no need to deflate 0 byte files
		}
		w, err := zw.CreateHeader(fh)
		if err != nil {
			return err
		}
		if fi.IsDir() {
			return nil
		}
		r, err := os.Open(path)
		if err != nil {
			return err
		}
		defer r.Close()
		_, err = io.Copy(w, r)
		return err
	})
	if err != nil {
		return err
	}
	if err := zw.Close(); err != nil {
		return err
	}
	return f.Close()
}

type tool struct {
	name       string
	commonDirs []string
}

var wixTool = tool{
	"http://wix.sourceforge.net/, version 3.5",
	[]string{`C:\Program Files\Windows Installer XML v3.5\bin`,
		`C:\Program Files (x86)\Windows Installer XML v3.5\bin`},
}

var hgTool = tool{
	"http://mercurial.selenic.com/wiki/WindowsInstall",
	[]string{`C:\Program Files\Mercurial`,
		`C:\Program Files (x86)\Mercurial`,
	},
}

var gccTool = tool{
	"Mingw gcc; http://sourceforge.net/projects/mingw/files/Installer/mingw-get-inst/",
	[]string{`C:\Mingw\bin`},
}

var windowsDeps = map[string]tool{
	"gcc":    gccTool,
	"heat":   wixTool,
	"candle": wixTool,
	"light":  wixTool,
	"cmd":    {"Windows cmd.exe", nil},
	"hg":     hgTool,
}

func checkWindowsDeps() {
	for prog, help := range windowsDeps {
		absPath, err := lookPath(prog)
		if err != nil {
			log.Fatalf("Failed to find necessary binary %q in path or common locations; %s", prog, help)
		}
		if *verbose {
			log.Printf("found windows dep %s at %s", prog, absPath)
		}
	}
}

func lookPath(prog string) (absPath string, err error) {
	absPath, err = exec.LookPath(prog)
	if err == nil {
		return
	}
	t, ok := windowsDeps[prog]
	if !ok {
		return
	}
	for _, dir := range t.commonDirs {
		for _, ext := range []string{"exe", "bat"} {
			absPath = filepath.Join(dir, prog+"."+ext)
			if _, err1 := os.Stat(absPath); err1 == nil {
				err = nil
				os.Setenv("PATH", os.Getenv("PATH")+";"+dir)
				return
			}
		}
	}
	return
}
