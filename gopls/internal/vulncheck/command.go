// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/semver"
	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/vuln/client"
	gvcapi "golang.org/x/vuln/exp/govulncheck"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

func init() {
	VulnerablePackages = vulnerablePackages
}

func findGOVULNDB(env []string) []string {
	for _, kv := range env {
		if strings.HasPrefix(kv, "GOVULNDB=") {
			return strings.Split(kv[len("GOVULNDB="):], ",")
		}
	}
	if GOVULNDB := os.Getenv("GOVULNDB"); GOVULNDB != "" {
		return strings.Split(GOVULNDB, ",")
	}
	return []string{"https://vuln.go.dev"}
}

// GoVersionForVulnTest is an internal environment variable used in gopls
// testing to examine govulncheck behavior with a go version different
// than what `go version` returns in the system.
const GoVersionForVulnTest = "_GOPLS_TEST_VULNCHECK_GOVERSION"

func init() {
	Main = func(cfg packages.Config, patterns ...string) error {
		// Set the mode that Source needs.
		cfg.Mode = packages.NeedName | packages.NeedImports | packages.NeedTypes |
			packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedDeps |
			packages.NeedModule
		logf := log.New(os.Stderr, "", log.Ltime).Printf
		logf("Loading packages...")
		pkgs, err := packages.Load(&cfg, patterns...)
		if err != nil {
			logf("Failed to load packages: %v", err)
			return err
		}
		if n := packages.PrintErrors(pkgs); n > 0 {
			err := errors.New("failed to load packages due to errors")
			logf("%v", err)
			return err
		}
		logf("Loaded %d packages and their dependencies", len(pkgs))
		cache, err := govulncheck.DefaultCache()
		if err != nil {
			return err
		}
		cli, err := client.NewClient(findGOVULNDB(cfg.Env), client.Options{
			HTTPCache: cache,
		})
		if err != nil {
			return err
		}
		res, err := gvcapi.Source(context.Background(), &gvcapi.Config{
			Client:    cli,
			GoVersion: os.Getenv(GoVersionForVulnTest),
		}, vulncheck.Convert(pkgs))
		if err != nil {
			return err
		}
		affecting := 0
		for _, v := range res.Vulns {
			if v.IsCalled() {
				affecting++
			}
		}
		logf("Found %d affecting vulns and %d unaffecting vulns in imported packages", affecting, len(res.Vulns)-affecting)
		if err := json.NewEncoder(os.Stdout).Encode(res); err != nil {
			return err
		}
		return nil
	}
}

// semverToGoTag returns the Go standard library repository tag corresponding
// to semver, a version string without the initial "v".
// Go tags differ from standard semantic versions in a few ways,
// such as beginning with "go" instead of "v".
func semverToGoTag(v string) string {
	if strings.HasPrefix(v, "v0.0.0") {
		return "master"
	}
	// Special case: v1.0.0 => go1.
	if v == "v1.0.0" {
		return "go1"
	}
	if !semver.IsValid(v) {
		return fmt.Sprintf("<!%s:invalid semver>", v)
	}
	goVersion := semver.Canonical(v)
	prerelease := semver.Prerelease(goVersion)
	versionWithoutPrerelease := strings.TrimSuffix(goVersion, prerelease)
	patch := strings.TrimPrefix(versionWithoutPrerelease, semver.MajorMinor(goVersion)+".")
	if patch == "0" {
		versionWithoutPrerelease = strings.TrimSuffix(versionWithoutPrerelease, ".0")
	}
	goVersion = fmt.Sprintf("go%s", strings.TrimPrefix(versionWithoutPrerelease, "v"))
	if prerelease != "" {
		// Go prereleases look like  "beta1" instead of "beta.1".
		// "beta1" is bad for sorting (since beta10 comes before beta9), so
		// require the dot form.
		i := finalDigitsIndex(prerelease)
		if i >= 1 {
			if prerelease[i-1] != '.' {
				return fmt.Sprintf("<!%s:final digits in a prerelease must follow a period>", v)
			}
			// Remove the dot.
			prerelease = prerelease[:i-1] + prerelease[i:]
		}
		goVersion += strings.TrimPrefix(prerelease, "-")
	}
	return goVersion
}

// finalDigitsIndex returns the index of the first digit in the sequence of digits ending s.
// If s doesn't end in digits, it returns -1.
func finalDigitsIndex(s string) int {
	// Assume ASCII (since the semver package does anyway).
	var i int
	for i = len(s) - 1; i >= 0; i-- {
		if s[i] < '0' || s[i] > '9' {
			break
		}
	}
	if i == len(s)-1 {
		return -1
	}
	return i + 1
}

// vulnerablePackages queries the vulndb and reports which vulnerabilities
// apply to this snapshot. The result contains a set of packages,
// grouped by vuln ID and by module.
func vulnerablePackages(ctx context.Context, snapshot source.Snapshot, modfile source.FileHandle) (*govulncheck.Result, error) {
	// We want to report the intersection of vulnerable packages in the vulndb
	// and packages transitively imported by this module ('go list -deps all').
	// We use snapshot.AllMetadata to retrieve the list of packages
	// as an approximation.
	//
	// TODO(hyangah): snapshot.AllMetadata is a superset of
	// `go list all` - e.g. when the workspace has multiple main modules
	// (multiple go.mod files), that can include packages that are not
	// used by this module. Vulncheck behavior with go.work is not well
	// defined. Figure out the meaning, and if we decide to present
	// the result as if each module is analyzed independently, make
	// gopls track a separate build list for each module and use that
	// information instead of snapshot.AllMetadata.
	metadata, err := snapshot.AllMetadata(ctx)
	if err != nil {
		return nil, err
	}

	// TODO(hyangah): handle vulnerabilities in the standard library.

	// Group packages by modules since vuln db is keyed by module.
	metadataByModule := map[source.PackagePath][]*source.Metadata{}
	for _, md := range metadata {
		mi := md.Module
		modulePath := source.PackagePath("stdlib")
		if mi != nil {
			modulePath = source.PackagePath(mi.Path)
		}
		metadataByModule[modulePath] = append(metadataByModule[modulePath], md)
	}

	// Request vuln entries from remote service.
	fsCache, err := govulncheck.DefaultCache()
	if err != nil {
		return nil, err
	}
	cli, err := client.NewClient(
		findGOVULNDB(snapshot.View().Options().EnvSlice()),
		client.Options{HTTPCache: govulncheck.NewInMemoryCache(fsCache)})
	if err != nil {
		return nil, err
	}
	// Keys are osv.Entry.IDs
	vulnsResult := map[string]*govulncheck.Vuln{}
	var (
		group errgroup.Group
		mu    sync.Mutex
	)

	goVersion := snapshot.View().Options().Env[GoVersionForVulnTest]
	if goVersion == "" {
		goVersion = snapshot.View().GoVersionString()
	}
	group.SetLimit(10)
	stdlibModule := &packages.Module{
		Path:    "stdlib",
		Version: goVersion,
	}
	for path, mds := range metadataByModule {
		path, mds := path, mds
		group.Go(func() error {
			effectiveModule := stdlibModule
			if m := mds[0].Module; m != nil {
				effectiveModule = m
			}
			for effectiveModule.Replace != nil {
				effectiveModule = effectiveModule.Replace
			}
			ver := effectiveModule.Version

			// TODO(go.dev/issues/56312): batch these requests for efficiency.
			vulns, err := cli.GetByModule(ctx, effectiveModule.Path)
			if err != nil {
				return err
			}
			if len(vulns) == 0 { // No known vulnerability.
				return nil
			}

			// set of packages in this module known to gopls.
			// This will be lazily initialized when we need it.
			var knownPkgs map[source.PackagePath]bool

			// Report vulnerabilities that affect packages of this module.
			for _, entry := range vulns {
				var vulnerablePkgs []*govulncheck.Package

				for _, a := range entry.Affected {
					if a.Package.Ecosystem != osv.GoEcosystem || a.Package.Name != effectiveModule.Path {
						continue
					}
					if !a.Ranges.AffectsSemver(ver) {
						continue
					}
					for _, imp := range a.EcosystemSpecific.Imports {
						if knownPkgs == nil {
							knownPkgs = toPackagePathSet(mds)
						}
						if knownPkgs[source.PackagePath(imp.Path)] {
							vulnerablePkgs = append(vulnerablePkgs, &govulncheck.Package{
								Path: imp.Path,
							})
						}
					}
				}
				if len(vulnerablePkgs) == 0 {
					continue
				}
				mu.Lock()
				vuln, ok := vulnsResult[entry.ID]
				if !ok {
					vuln = &govulncheck.Vuln{OSV: entry}
					vulnsResult[entry.ID] = vuln
				}
				vuln.Modules = append(vuln.Modules, &govulncheck.Module{
					Path:         string(path),
					FoundVersion: ver,
					FixedVersion: fixedVersion(effectiveModule.Path, entry.Affected),
					Packages:     vulnerablePkgs,
				})
				mu.Unlock()
			}
			return nil
		})
	}
	if err := group.Wait(); err != nil {
		return nil, err
	}

	vulns := make([]*govulncheck.Vuln, 0, len(vulnsResult))
	for _, v := range vulnsResult {
		vulns = append(vulns, v)
	}
	// Sort so the results are deterministic.
	sort.Slice(vulns, func(i, j int) bool {
		return vulns[i].OSV.ID < vulns[j].OSV.ID
	})
	ret := &govulncheck.Result{
		Vulns: vulns,
		Mode:  govulncheck.ModeImports,
	}
	return ret, nil
}

// toPackagePathSet transforms the metadata to a set of package paths.
func toPackagePathSet(mds []*source.Metadata) map[source.PackagePath]bool {
	pkgPaths := make(map[source.PackagePath]bool, len(mds))
	for _, md := range mds {
		pkgPaths[md.PkgPath] = true
	}
	return pkgPaths
}

func fixedVersion(modulePath string, affected []osv.Affected) string {
	fixed := govulncheck.LatestFixed(modulePath, affected)
	if fixed != "" {
		fixed = versionString(modulePath, fixed)
	}
	return fixed
}

// versionString prepends a version string prefix (`v` or `go`
// depending on the modulePath) to the given semver-style version string.
func versionString(modulePath, version string) string {
	if version == "" {
		return ""
	}
	v := "v" + version
	// These are internal Go module paths used by the vuln DB
	// when listing vulns in standard library and the go command.
	if modulePath == "stdlib" || modulePath == "toolchain" {
		return semverToGoTag(v)
	}
	return v
}
