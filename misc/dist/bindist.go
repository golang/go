// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a tool for packaging binary releases.
// It supports FreeBSD, Linux, and OS X.
package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"errors"
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
	"strings"
)

var (
	tag  = flag.String("tag", "weekly", "mercurial tag to check out")
	repo = flag.String("repo", "https://code.google.com/p/go", "repo URL")

	username, password string // for Google Code upload
)

const (
	packageMaker = "/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker"
	uploadURL    = "https://go.googlecode.com/files"
)

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
	if err := readCredentials(); err != nil {
		log.Println("readCredentials:", err)
	}
	for _, targ := range flag.Args() {
		var b Build
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

	src := filepath.Join(b.root, "src")
	if b.Source {
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
	pat := filepath.Join(b.root, "pkg/tool/*/dist")
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
	targ := fmt.Sprintf("go.%s.%s-%s", version, b.OS, b.Arch)
	switch b.OS {
	case "linux", "freebsd", "":
		// build tarball
		if b.Source {
			targ = fmt.Sprintf("go.%s.src", version)
		}
		targ += ".tar.gz"
		_, err = b.run("", "tar", "czf", targ, "-C", work, "go")
	case "darwin":
		// arrange work so it's laid out as the dest filesystem
		etc := filepath.Join(b.root, "misc/dist/darwin/etc")
		_, err = b.run(work, "cp", "-r", etc, ".")
		if err != nil {
			return err
		}
		localDir := filepath.Join(work, "usr/local")
		err = os.MkdirAll(localDir, 0744)
		if err != nil {
			return err
		}
		_, err = b.run(work, "mv", "go", localDir)
		if err != nil {
			return err
		}
		// build package
		pm := packageMaker
		if !exists(pm) {
			pm = "/Developer" + pm
			if !exists(pm) {
				return errors.New("couldn't find PackageMaker")
			}
		}
		targ += ".pkg"
		scripts := filepath.Join(work, "usr/local/go/misc/dist/darwin/scripts")
		_, err = b.run("", pm, "-v",
			"-r", work,
			"-o", targ,
			"--scripts", scripts,
			"--id", "com.googlecode.go",
			"--title", "Go",
			"--version", "1.0",
			"--target", "10.5")
	case "windows":
		win := filepath.Join(b.root, "misc/dist/windows")
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
		targ += ".msi"
		err = cp(targ, msi)
	}
	if err == nil && password != "" {
		err = b.upload(version, targ)
	}
	return err
}

func (b *Build) run(dir, name string, args ...string) ([]byte, error) {
	buf := new(bytes.Buffer)
	cmd := exec.Command(name, args...)
	cmd.Stdout = buf
	cmd.Stderr = buf
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

func (b *Build) upload(version string, filename string) error {
	// Prepare upload metadata.
	var labels []string
	os_, arch := b.OS, b.Arch
	switch b.Arch {
	case "386":
		arch = "32-bit"
	case "amd64":
		arch = "64-bit"
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
		labels = append(labels, "Type-Installer", "OpSys-Windows")
	}
	summary := fmt.Sprintf("Go %s %s (%s)", version, os_, arch)
	if b.Source {
		labels = append(labels, "Type-Source")
		summary = fmt.Sprintf("Go %s (source only)", version)
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
	name := filepath.Join(os.Getenv("HOME"), ".gobuildkey")
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
