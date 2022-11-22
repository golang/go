// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/semver"
	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/vuln/client"
	gvcapi "golang.org/x/vuln/exp/govulncheck"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

func init() {
	Govulncheck = govulncheckFunc

	VulnerablePackages = vulnerablePackages
}

func govulncheckFunc(ctx context.Context, cfg *packages.Config, patterns string) (res command.VulncheckResult, _ error) {
	if patterns == "" {
		patterns = "."
	}

	dbClient, err := client.NewClient(findGOVULNDB(cfg.Env), client.Options{HTTPCache: govulncheck.DefaultCache()})
	if err != nil {
		return res, err
	}

	c := Cmd{Client: dbClient}
	vulns, err := c.Run(ctx, cfg, patterns)
	if err != nil {
		return res, err
	}

	res.Vuln = vulns
	return res, err
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

type Vuln = command.Vuln
type CallStack = command.CallStack
type StackEntry = command.StackEntry

// Cmd is an in-process govulncheck command runner
// that uses the provided client.Client.
type Cmd struct {
	Client client.Client
}

// Run runs the govulncheck after loading packages using the provided packages.Config.
func (c *Cmd) Run(ctx context.Context, cfg *packages.Config, patterns ...string) (_ []Vuln, err error) {
	logger := log.New(log.Default().Writer(), "", 0)
	cfg.Mode |= packages.NeedModule | packages.NeedName | packages.NeedFiles |
		packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedTypes |
		packages.NeedTypesSizes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedDeps

	logger.Println("loading packages...")
	loadedPkgs, err := govulncheck.LoadPackages(cfg, patterns...)
	if err != nil {
		logger.Printf("%v", err)
		return nil, fmt.Errorf("package load failed")
	}

	logger.Printf("analyzing %d packages...\n", len(loadedPkgs))

	r, err := vulncheck.Source(ctx, loadedPkgs, &vulncheck.Config{Client: c.Client, SourceGoVersion: goVersion()})
	if err != nil {
		return nil, err
	}

	logger.Printf("selecting affecting vulnerabilities from %d findings...\n", len(r.Vulns))
	unaffectedMods := filterUnaffected(r.Vulns)
	r.Vulns = filterCalled(r)

	logger.Printf("found %d vulnerabilities.\n", len(r.Vulns))
	callInfo := govulncheck.GetCallInfo(r, loadedPkgs)
	return toVulns(callInfo, unaffectedMods)
	// TODO: add import graphs.
}

// filterCalled returns vulnerabilities where the symbols are actually called.
func filterCalled(r *vulncheck.Result) []*vulncheck.Vuln {
	var vulns []*vulncheck.Vuln
	for _, v := range r.Vulns {
		if v.CallSink != 0 {
			vulns = append(vulns, v)
		}
	}
	return vulns
}

// filterUnaffected returns vulnerabilities where no symbols are called,
// grouped by module.
func filterUnaffected(vulns []*vulncheck.Vuln) map[string][]*osv.Entry {
	// It is possible that the same vuln.OSV.ID has vuln.CallSink != 0
	// for one symbol, but vuln.CallSink == 0 for a different one, so
	// we need to filter out ones that have been called.
	called := map[string]bool{}
	for _, vuln := range vulns {
		if vuln.CallSink != 0 {
			called[vuln.OSV.ID] = true
		}
	}

	modToIDs := map[string]map[string]*osv.Entry{}
	for _, vuln := range vulns {
		if !called[vuln.OSV.ID] {
			if _, ok := modToIDs[vuln.ModPath]; !ok {
				modToIDs[vuln.ModPath] = map[string]*osv.Entry{}
			}
			// keep only one vuln.OSV instance for the same ID.
			modToIDs[vuln.ModPath][vuln.OSV.ID] = vuln.OSV
		}
	}
	output := map[string][]*osv.Entry{}
	for m, vulnSet := range modToIDs {
		var vulns []*osv.Entry
		for _, vuln := range vulnSet {
			vulns = append(vulns, vuln)
		}
		sort.Slice(vulns, func(i, j int) bool { return vulns[i].ID < vulns[j].ID })
		output[m] = vulns
	}
	return output
}

func fixed(modPath string, v *osv.Entry) string {
	lf := govulncheck.LatestFixed(modPath, v.Affected)
	if lf != "" && lf[0] != 'v' {
		lf = "v" + lf
	}
	return lf
}

func toVulns(ci *govulncheck.CallInfo, unaffectedMods map[string][]*osv.Entry) ([]Vuln, error) {
	var vulns []Vuln

	for _, vg := range ci.VulnGroups {
		v0 := vg[0]
		vuln := Vuln{
			ID:             v0.OSV.ID,
			PkgPath:        v0.PkgPath,
			CurrentVersion: ci.ModuleVersions[v0.ModPath],
			FixedVersion:   fixed(v0.ModPath, v0.OSV),
			Details:        v0.OSV.Details,

			Aliases: v0.OSV.Aliases,
			Symbol:  v0.Symbol,
			ModPath: v0.ModPath,
			URL:     href(v0.OSV),
		}

		// Keep first call stack for each vuln.
		for _, v := range vg {
			if css := ci.CallStacks[v]; len(css) > 0 {
				vuln.CallStacks = append(vuln.CallStacks, toCallStack(css[0]))
				// TODO(hyangah):  https://go-review.googlesource.com/c/vuln/+/425183 added position info
				// in the summary but we don't need the info. Allow SummarizeCallStack to skip it optionally.
				sum := trimPosPrefix(govulncheck.SummarizeCallStack(css[0], ci.TopPackages, v.PkgPath))
				vuln.CallStackSummaries = append(vuln.CallStackSummaries, sum)
			}
		}
		vulns = append(vulns, vuln)
	}
	for m, vg := range unaffectedMods {
		for _, v0 := range vg {
			vuln := Vuln{
				ID:             v0.ID,
				Details:        v0.Details,
				Aliases:        v0.Aliases,
				ModPath:        m,
				URL:            href(v0),
				CurrentVersion: "",
				FixedVersion:   fixed(m, v0),
			}
			vulns = append(vulns, vuln)
		}
	}
	return vulns, nil
}

func trimPosPrefix(summary string) string {
	_, after, found := strings.Cut(summary, ": ")
	if !found {
		return summary
	}
	return after
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
		logf("Loaded %d packages and their dependencies", len(pkgs))
		cli, err := client.NewClient(findGOVULNDB(cfg.Env), client.Options{
			HTTPCache: govulncheck.DefaultCache(),
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
		logf("Found %d vulnerabilities", len(res.Vulns))
		if err := json.NewEncoder(os.Stdout).Encode(res); err != nil {
			return err
		}
		return nil
	}
}

var (
	// Regexp for matching go tags. The groups are:
	// 1  the major.minor version
	// 2  the patch version, or empty if none
	// 3  the entire prerelease, if present
	// 4  the prerelease type ("beta" or "rc")
	// 5  the prerelease number
	tagRegexp = regexp.MustCompile(`^go(\d+\.\d+)(\.\d+|)((beta|rc|-pre)(\d+))?$`)
)

// This is a modified copy of pkgsite/internal/stdlib:VersionForTag.
func GoTagToSemver(tag string) string {
	if tag == "" {
		return ""
	}

	tag = strings.Fields(tag)[0]
	// Special cases for go1.
	if tag == "go1" {
		return "v1.0.0"
	}
	if tag == "go1.0" {
		return ""
	}
	m := tagRegexp.FindStringSubmatch(tag)
	if m == nil {
		return ""
	}
	version := "v" + m[1]
	if m[2] != "" {
		version += m[2]
	} else {
		version += ".0"
	}
	if m[3] != "" {
		if !strings.HasPrefix(m[4], "-") {
			version += "-"
		}
		version += m[4] + "." + m[5]
	}
	return version
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
	metadata, err := snapshot.AllValidMetadata(ctx)
	if err != nil {
		return nil, err
	}

	// Group packages by modules since vuln db is keyed by module.
	metadataByModule := map[source.PackagePath][]*source.Metadata{}
	for _, md := range metadata {
		// TODO(hyangah): delete after go.dev/cl/452057 is merged.
		// After the cl, this becomes an impossible condition.
		if md == nil {
			continue
		}
		mi := md.Module
		if mi == nil {
			continue
		}
		modulePath := source.PackagePath(mi.Path)
		metadataByModule[modulePath] = append(metadataByModule[modulePath], md)
	}

	// Request vuln entries from remote service.
	cli, err := client.NewClient(
		findGOVULNDB(snapshot.View().Options().EnvSlice()),
		client.Options{HTTPCache: govulncheck.DefaultCache()})
	if err != nil {
		return nil, err
	}
	// Keys are osv.Entry.IDs
	vulnsResult := map[string]*govulncheck.Vuln{}
	var (
		group errgroup.Group
		mu    sync.Mutex
	)

	group.SetLimit(10)
	for path, mds := range metadataByModule {
		path, mds := path, mds
		group.Go(func() error {

			effectiveModule := mds[0].Module
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
