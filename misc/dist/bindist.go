// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a tool for packaging binary releases.
// It supports FreeBSD, Linux, OS X, and Windows.
package main

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

var (
	tag      = flag.String("tag", "release", "mercurial tag to check out")
	repo     = flag.String("repo", "https://code.google.com/p/go", "repo URL")
	verbose  = flag.Bool("v", false, "verbose output")
	upload   = flag.Bool("upload", true, "upload resulting files to Google Code")
	wxsFile  = flag.String("wxs", "", "path to custom installer.wxs")
	addLabel = flag.String("label", "", "additional label to apply to file when uploading")

	username, password string // for Google Code upload
)

const (
	uploadURL = "https://go.googlecode.com/files"
)

var preBuildCleanFiles = []string{
	"lib/codereview",
	"misc/dashboard/godashboard",
	"src/cmd/cov",
	"src/cmd/prof",
	"src/pkg/exp",
	"src/pkg/old",
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

var fileRe = regexp.MustCompile(`^go\.([a-z0-9-.]+)\.(src|([a-z0-9]+)-([a-z0-9]+))\.`)

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
			log.Println("readCredentials:", err)
		}
	}
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
			}
			if !*upload {
				log.Printf("%s: -upload=false, skipping", targ)
				continue
			}
			if err := b.Upload(version, targ); err != nil {
				log.Printf("%s: %v", targ, err)
			}
			continue
		}
		if targ == "source" {
			b.Source = true
		} else {
			p := strings.SplitN(targ, "-", 2)
			if len(p) != 2 {
				log.Println("Ignoring unrecognized target:", targ)
				continue
			}
			b.OS = p[0]
			b.Arch = p[1]
		}
		if err := b.Do(); err != nil {
			log.Printf("%s: %v", targ, err)
		}
	}
}

type Build struct {
	Source bool // if true, OS and Arch must be empty
	OS     string
	Arch   string
	root   string
}

func (b *Build) Do() error {
	work, err := ioutil.TempDir("", "bindist")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	b.root = filepath.Join(work, "go")

	// Clone Go distribution and update to tag.
	_, err = b.run(work, "hg", "clone", "-q", *repo, b.root)
	if err != nil {
		return err
	}
	_, err = b.run(b.root, "hg", "update", *tag)
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
	if !strings.HasPrefix(base, "go") {
		base = "go." + base
	}
	var targs []string
	switch b.OS {
	case "linux", "freebsd", "":
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
		// arrange work so it's laid out as the dest filesystem
		etc := filepath.Join(b.root, "misc/dist/darwin/etc")
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
		dist := filepath.Join(runtime.GOROOT(), "misc/dist")
		_, err = b.run("", "pkgbuild",
			"--identifier", "com.googlecode.go",
			"--version", "1.0",
			"--scripts", filepath.Join(dist, "darwin/scripts"),
			"--root", work,
			filepath.Join(pkgdest, "com.googlecode.go.pkg"))
		if err != nil {
			return err
		}
		targ := base + ".pkg"
		_, err = b.run("", "productbuild",
			"--distribution", filepath.Join(dist, "darwin/Distribution"),
			"--resources", filepath.Join(dist, "darwin/Resources"),
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
		win := filepath.Join(b.root, "misc/dist/windows")
		installer := filepath.Join(win, "installer.wxs")
		if *wxsFile != "" {
			installer = *wxsFile
		}
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
			"-dVersion="+version,
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
				return err
			}
		}
	}
	return err
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
	)
	return env
}

func (b *Build) Upload(version string, filename string) error {
	// Prepare upload metadata.
	var labels []string
	os_, arch := b.OS, b.Arch
	switch b.Arch {
	case "386":
		arch = "x86 32-bit"
	case "amd64":
		arch = "x86 64-bit"
	}
	if arch != "" {
		labels = append(labels, "Arch-"+b.Arch)
	}
	switch b.OS {
	case "linux":
		os_ = "Linux"
		labels = append(labels, "Type-Archive", "OpSys-Linux")
	case "freebsd":
		os_ = "FreeBSD"
		labels = append(labels, "Type-Archive", "OpSys-FreeBSD")
	case "darwin":
		os_ = "Mac OS X"
		labels = append(labels, "Type-Installer", "OpSys-OSX")
	case "windows":
		os_ = "Windows"
		labels = append(labels, "OpSys-Windows")
	}
	summary := fmt.Sprintf("%s %s (%s)", version, os_, arch)
	if b.OS == "windows" {
		switch {
		case strings.HasSuffix(filename, ".msi"):
			labels = append(labels, "Type-Installer")
			summary += " MSI installer"
		case strings.HasSuffix(filename, ".zip"):
			labels = append(labels, "Type-Archive")
			summary += " ZIP archive"
		}
	}
	if b.Source {
		labels = append(labels, "Type-Source")
		summary = fmt.Sprintf("%s (source only)", version)
	}
	if *addLabel != "" {
		labels = append(labels, *addLabel)
	}
	// Put "Go" prefix on summary when it doesn't already begin with "go".
	if !strings.HasPrefix(strings.ToLower(summary), "go") {
		summary = "Go " + summary
	}

	// Open file to upload.
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Prepare multipart payload.
	body := new(bytes.Buffer)
	w := multipart.NewWriter(body)
	if err := w.WriteField("summary", summary); err != nil {
		return err
	}
	for _, l := range labels {
		if err := w.WriteField("label", l); err != nil {
			return err
		}
	}
	fw, err := w.CreateFormFile("filename", filename)
	if err != nil {
		return err
	}
	if _, err = io.Copy(fw, f); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}

	// Send the file to Google Code.
	req, err := http.NewRequest("POST", uploadURL, body)
	if err != nil {
		return err
	}
	token := fmt.Sprintf("%s:%s", username, password)
	token = base64.StdEncoding.EncodeToString([]byte(token))
	req.Header.Set("Authorization", "Basic "+token)
	req.Header.Set("Content-type", w.FormDataContentType())

	resp, err := http.DefaultTransport.RoundTrip(req)
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		fmt.Fprintln(os.Stderr, "upload failed")
		defer resp.Body.Close()
		io.Copy(os.Stderr, resp.Body)
		return fmt.Errorf("upload: %s", resp.Status)
	}
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
	r := bufio.NewReader(f)
	for i := 0; i < 3; i++ {
		b, _, err := r.ReadLine()
		if err != nil {
			return err
		}
		b = bytes.TrimSpace(b)
		switch i {
		case 1:
			username = string(b)
		case 2:
			password = string(b)
		}
	}
	return nil
}

func cp(dst, src string) error {
	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()
	df, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer df.Close()
	_, err = io.Copy(df, sf)
	return err
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
		hdr, err := tarFileInfoHeader(fi, path)
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

// sysStat, if non-nil, populates h from system-dependent fields of fi.
var sysStat func(fi os.FileInfo, h *tar.Header) error

// Mode constants from the tar spec.
const (
	c_ISDIR  = 040000
	c_ISFIFO = 010000
	c_ISREG  = 0100000
	c_ISLNK  = 0120000
	c_ISBLK  = 060000
	c_ISCHR  = 020000
	c_ISSOCK = 0140000
)

// tarFileInfoHeader creates a partially-populated Header from an os.FileInfo.
// The filename parameter is used only in the case of symlinks, to call os.Readlink.
// If fi is a symlink but filename is empty, an error is returned.
func tarFileInfoHeader(fi os.FileInfo, filename string) (*tar.Header, error) {
	h := &tar.Header{
		Name:    fi.Name(),
		ModTime: fi.ModTime(),
		Mode:    int64(fi.Mode().Perm()), // or'd with c_IS* constants later
	}
	switch {
	case fi.Mode()&os.ModeType == 0:
		h.Mode |= c_ISREG
		h.Typeflag = tar.TypeReg
		h.Size = fi.Size()
	case fi.IsDir():
		h.Typeflag = tar.TypeDir
		h.Mode |= c_ISDIR
	case fi.Mode()&os.ModeSymlink != 0:
		h.Typeflag = tar.TypeSymlink
		h.Mode |= c_ISLNK
		if filename == "" {
			return h, fmt.Errorf("archive/tar: unable to populate Header.Linkname of symlinks")
		}
		targ, err := os.Readlink(filename)
		if err != nil {
			return h, err
		}
		h.Linkname = targ
	case fi.Mode()&os.ModeDevice != 0:
		if fi.Mode()&os.ModeCharDevice != 0 {
			h.Mode |= c_ISCHR
			h.Typeflag = tar.TypeChar
		} else {
			h.Mode |= c_ISBLK
			h.Typeflag = tar.TypeBlock
		}
	case fi.Mode()&os.ModeSocket != 0:
		h.Mode |= c_ISSOCK
	default:
		return nil, fmt.Errorf("archive/tar: unknown file mode %v", fi.Mode())
	}
	if sysStat != nil {
		return h, sysStat(fi, h)
	}
	return h, nil
}
