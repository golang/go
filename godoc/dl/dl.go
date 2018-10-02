// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package dl implements a simple downloads frontend server.
//
// It accepts HTTP POST requests to create a new download metadata entity, and
// lists entities with sorting and filtering.
// It is designed to run only on the instance of godoc that serves golang.org.
package dl

import (
	"fmt"
	"html/template"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	downloadBaseURL = "https://dl.google.com/go/"
	cacheKey        = "download_list_3" // increment if listTemplateData changes
	cacheDuration   = time.Hour
)

// File represents a file on the golang.org downloads page.
// It should be kept in sync with the upload code in x/build/cmd/release.
type File struct {
	Filename       string    `json:"filename"`
	OS             string    `json:"os"`
	Arch           string    `json:"arch"`
	Version        string    `json:"version"`
	Checksum       string    `json:"-" datastore:",noindex"` // SHA1; deprecated
	ChecksumSHA256 string    `json:"sha256" datastore:",noindex"`
	Size           int64     `json:"size" datastore:",noindex"`
	Kind           string    `json:"kind"` // "archive", "installer", "source"
	Uploaded       time.Time `json:"-"`
}

func (f File) ChecksumType() string {
	if f.ChecksumSHA256 != "" {
		return "SHA256"
	}
	return "SHA1"
}

func (f File) PrettyChecksum() string {
	if f.ChecksumSHA256 != "" {
		return f.ChecksumSHA256
	}
	return f.Checksum
}

func (f File) PrettyOS() string {
	if f.OS == "darwin" {
		switch {
		case strings.Contains(f.Filename, "osx10.8"):
			return "OS X 10.8+"
		case strings.Contains(f.Filename, "osx10.6"):
			return "OS X 10.6+"
		}
	}
	return pretty(f.OS)
}

func (f File) PrettySize() string {
	const mb = 1 << 20
	if f.Size == 0 {
		return ""
	}
	if f.Size < mb {
		// All Go releases are >1mb, but handle this case anyway.
		return fmt.Sprintf("%v bytes", f.Size)
	}
	return fmt.Sprintf("%.0fMB", float64(f.Size)/mb)
}

var primaryPorts = map[string]bool{
	"darwin/amd64":  true,
	"linux/386":     true,
	"linux/amd64":   true,
	"linux/armv6l":  true,
	"windows/386":   true,
	"windows/amd64": true,
}

func (f File) PrimaryPort() bool {
	if f.Kind == "source" {
		return true
	}
	return primaryPorts[f.OS+"/"+f.Arch]
}

func (f File) Highlight() bool {
	switch {
	case f.Kind == "source":
		return true
	case f.Arch == "amd64" && f.OS == "linux":
		return true
	case f.Arch == "amd64" && f.Kind == "installer":
		switch f.OS {
		case "windows":
			return true
		case "darwin":
			if !strings.Contains(f.Filename, "osx10.6") {
				return true
			}
		}
	}
	return false
}

func (f File) URL() string {
	return downloadBaseURL + f.Filename
}

type Release struct {
	Version        string `json:"version"`
	Stable         bool   `json:"stable"`
	Files          []File `json:"files"`
	Visible        bool   `json:"-"` // show files on page load
	SplitPortTable bool   `json:"-"` // whether files should be split by primary/other ports.
}

type Feature struct {
	// The File field will be filled in by the first stable File
	// whose name matches the given fileRE.
	File
	fileRE *regexp.Regexp

	Platform     string // "Microsoft Windows", "Apple macOS", "Linux"
	Requirements string // "Windows XP and above, 64-bit Intel Processor"
}

// featuredFiles lists the platforms and files to be featured
// at the top of the downloads page.
var featuredFiles = []Feature{
	{
		Platform:     "Microsoft Windows",
		Requirements: "Windows 7 or later, Intel 64-bit processor",
		fileRE:       regexp.MustCompile(`\.windows-amd64\.msi$`),
	},
	{
		Platform:     "Apple macOS",
		Requirements: "macOS 10.10 or later, Intel 64-bit processor",
		fileRE:       regexp.MustCompile(`\.darwin-amd64(-osx10\.8)?\.pkg$`),
	},
	{
		Platform:     "Linux",
		Requirements: "Linux 2.6.23 or later, Intel 64-bit processor",
		fileRE:       regexp.MustCompile(`\.linux-amd64\.tar\.gz$`),
	},
	{
		Platform: "Source",
		fileRE:   regexp.MustCompile(`\.src\.tar\.gz$`),
	},
}

// data to send to the template; increment cacheKey if you change this.
type listTemplateData struct {
	Featured                  []Feature
	Stable, Unstable, Archive []Release
}

var (
	listTemplate  = template.Must(template.New("").Funcs(templateFuncs).Parse(templateHTML))
	templateFuncs = template.FuncMap{"pretty": pretty}
)

func filesToFeatured(fs []File) (featured []Feature) {
	for _, feature := range featuredFiles {
		for _, file := range fs {
			if feature.fileRE.MatchString(file.Filename) {
				feature.File = file
				featured = append(featured, feature)
				break
			}
		}
	}
	return
}

func filesToReleases(fs []File) (stable, unstable, archive []Release) {
	sort.Sort(fileOrder(fs))

	var r *Release
	var stableMaj, stableMin int
	add := func() {
		if r == nil {
			return
		}
		if !r.Stable {
			if len(unstable) != 0 {
				// Only show one (latest) unstable version.
				return
			}
			maj, min, _ := parseVersion(r.Version)
			if maj < stableMaj || maj == stableMaj && min <= stableMin {
				// Display unstable version only if newer than the
				// latest stable release.
				return
			}
			unstable = append(unstable, *r)
		}

		// Reports whether the release is the most recent minor version of the
		// two most recent major versions.
		shouldAddStable := func() bool {
			if len(stable) >= 2 {
				// Show up to two stable versions.
				return false
			}
			if len(stable) == 0 {
				// Most recent stable version.
				stableMaj, stableMin, _ = parseVersion(r.Version)
				return true
			}
			if maj, _, _ := parseVersion(r.Version); maj == stableMaj {
				// Older minor version of most recent major version.
				return false
			}
			// Second most recent stable version.
			return true
		}
		if !shouldAddStable() {
			archive = append(archive, *r)
			return
		}

		// Split the file list into primary/other ports for the stable releases.
		// NOTE(cbro): This is only done for stable releases because maintaining the historical
		// nature of primary/other ports for older versions is infeasible.
		// If freebsd is considered primary some time in the future, we'd not want to
		// mark all of the older freebsd binaries as "primary".
		// It might be better if we set that as a flag when uploading.
		r.SplitPortTable = true
		r.Visible = true // Toggle open all stable releases.
		stable = append(stable, *r)
	}
	for _, f := range fs {
		if r == nil || f.Version != r.Version {
			add()
			r = &Release{
				Version: f.Version,
				Stable:  isStable(f.Version),
			}
		}
		r.Files = append(r.Files, f)
	}
	add()
	return
}

// isStable reports whether the version string v is a stable version.
func isStable(v string) bool {
	return !strings.Contains(v, "beta") && !strings.Contains(v, "rc")
}

type fileOrder []File

func (s fileOrder) Len() int      { return len(s) }
func (s fileOrder) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s fileOrder) Less(i, j int) bool {
	a, b := s[i], s[j]
	if av, bv := a.Version, b.Version; av != bv {
		return versionLess(av, bv)
	}
	if a.OS != b.OS {
		return a.OS < b.OS
	}
	if a.Arch != b.Arch {
		return a.Arch < b.Arch
	}
	if a.Kind != b.Kind {
		return a.Kind < b.Kind
	}
	return a.Filename < b.Filename
}

func versionLess(a, b string) bool {
	// Put stable releases first.
	if isStable(a) != isStable(b) {
		return isStable(a)
	}
	maja, mina, ta := parseVersion(a)
	majb, minb, tb := parseVersion(b)
	if maja == majb {
		if mina == minb {
			return ta >= tb
		}
		return mina >= minb
	}
	return maja >= majb
}

func parseVersion(v string) (maj, min int, tail string) {
	if i := strings.Index(v, "beta"); i > 0 {
		tail = v[i:]
		v = v[:i]
	}
	if i := strings.Index(v, "rc"); i > 0 {
		tail = v[i:]
		v = v[:i]
	}
	p := strings.Split(strings.TrimPrefix(v, "go1."), ".")
	maj, _ = strconv.Atoi(p[0])
	if len(p) < 2 {
		return
	}
	min, _ = strconv.Atoi(p[1])
	return
}

func validUser(user string) bool {
	switch user {
	case "adg", "bradfitz", "cbro", "andybons", "valsorda", "dmitshur", "katiehockman":
		return true
	}
	return false
}

var (
	fileRe  = regexp.MustCompile(`^go[0-9a-z.]+\.[0-9a-z.-]+\.(tar\.gz|pkg|msi|zip)$`)
	goGetRe = regexp.MustCompile(`^go[0-9a-z.]+\.[0-9a-z.-]+$`)
)

// pretty returns a human-readable version of the given OS, Arch, or Kind.
func pretty(s string) string {
	t, ok := prettyStrings[s]
	if !ok {
		return s
	}
	return t
}

var prettyStrings = map[string]string{
	"darwin":  "macOS",
	"freebsd": "FreeBSD",
	"linux":   "Linux",
	"windows": "Windows",

	"386":    "x86",
	"amd64":  "x86-64",
	"armv6l": "ARMv6",
	"arm64":  "ARMv8",

	"archive":   "Archive",
	"installer": "Installer",
	"source":    "Source",
}
