// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/gopathwalk"
	"golang.org/x/tools/internal/semver"
)

// debug controls verbose logging.
const debug = false

// A goTooOldError reports that the go command
// found by exec.LookPath is too old to use the new go list behavior.
type goTooOldError struct {
	error
}

// goListDriver uses the go list command to interpret the patterns and produce
// the build system package structure.
// See driver for more details.
func goListDriver(cfg *Config, patterns ...string) (*driverResponse, error) {
	var sizes types.Sizes
	var sizeserr error
	var sizeswg sync.WaitGroup
	if cfg.Mode >= LoadTypes {
		sizeswg.Add(1)
		go func() {
			sizes, sizeserr = getSizes(cfg)
			sizeswg.Done()
		}()
	}

	// Determine files requested in contains patterns
	var containFiles []string
	var packagesNamed []string
	restPatterns := make([]string, 0, len(patterns))
	// Extract file= and other [querytype]= patterns. Report an error if querytype
	// doesn't exist.
extractQueries:
	for _, pattern := range patterns {
		eqidx := strings.Index(pattern, "=")
		if eqidx < 0 {
			restPatterns = append(restPatterns, pattern)
		} else {
			query, value := pattern[:eqidx], pattern[eqidx+len("="):]
			switch query {
			case "file":
				containFiles = append(containFiles, value)
			case "pattern":
				restPatterns = append(restPatterns, value)
			case "name":
				packagesNamed = append(packagesNamed, value)
			case "": // not a reserved query
				restPatterns = append(restPatterns, pattern)
			default:
				for _, rune := range query {
					if rune < 'a' || rune > 'z' { // not a reserved query
						restPatterns = append(restPatterns, pattern)
						continue extractQueries
					}
				}
				// Reject all other patterns containing "="
				return nil, fmt.Errorf("invalid query type %q in query pattern %q", query, pattern)
			}
		}
	}
	patterns = restPatterns

	// TODO(matloob): Remove the definition of listfunc and just use golistPackages once go1.12 is released.
	var listfunc driver
	var isFallback bool
	listfunc = func(cfg *Config, words ...string) (*driverResponse, error) {
		response, err := golistDriverCurrent(cfg, words...)
		if _, ok := err.(goTooOldError); ok {
			isFallback = true
			listfunc = golistDriverFallback
			return listfunc(cfg, words...)
		}
		listfunc = golistDriverCurrent
		return response, err
	}

	var response *driverResponse
	var err error

	// see if we have any patterns to pass through to go list.
	if len(restPatterns) > 0 {
		response, err = listfunc(cfg, restPatterns...)
		if err != nil {
			return nil, err
		}
	} else {
		response = &driverResponse{}
	}

	sizeswg.Wait()
	if sizeserr != nil {
		return nil, sizeserr
	}
	// types.SizesFor always returns nil or a *types.StdSizes
	response.Sizes, _ = sizes.(*types.StdSizes)

	seenPkgs := make(map[string]*Package) // for deduplication. different containing queries could produce same packages
	for _, pkg := range response.Packages {
		seenPkgs[pkg.ID] = pkg
	}
	addPkg := func(p *Package) {
		if _, ok := seenPkgs[p.ID]; ok {
			return
		}
		seenPkgs[p.ID] = p
		response.Packages = append(response.Packages, p)
	}

	var containsCandidates []string

	if len(containFiles) != 0 {
		containsCandidates, err = runContainsQueries(cfg, listfunc, isFallback, addPkg, containFiles)
		if err != nil {
			return nil, err
		}
	}

	if len(packagesNamed) != 0 {
		namedResults, err := runNamedQueries(cfg, listfunc, addPkg, packagesNamed)
		if err != nil {
			return nil, err
		}
		response.Roots = append(response.Roots, namedResults...)
	}

	modifiedPkgs, needPkgs, err := processGolistOverlay(cfg, response)
	if err != nil {
		return nil, err
	}
	if len(containFiles) > 0 {
		containsCandidates = append(containsCandidates, modifiedPkgs...)
		containsCandidates = append(containsCandidates, needPkgs...)
	}

	if len(needPkgs) > 0 {
		addNeededOverlayPackages(cfg, listfunc, addPkg, needPkgs)
		if err != nil {
			return nil, err
		}
	}
	// Check candidate packages for containFiles.
	if len(containFiles) > 0 {
		for _, id := range containsCandidates {
			pkg := seenPkgs[id]
			for _, f := range containFiles {
				for _, g := range pkg.GoFiles {
					if sameFile(f, g) {
						response.Roots = append(response.Roots, id)
					}
				}
			}
		}
	}

	return response, nil
}

func addNeededOverlayPackages(cfg *Config, driver driver, addPkg func(*Package), pkgs []string) error {
	response, err := driver(cfg, pkgs...)
	if err != nil {
		return err
	}
	for _, pkg := range response.Packages {
		addPkg(pkg)
	}
	return nil
}

func runContainsQueries(cfg *Config, driver driver, isFallback bool, addPkg func(*Package), queries []string) ([]string, error) {
	var results []string
	for _, query := range queries {
		// TODO(matloob): Do only one query per directory.
		fdir := filepath.Dir(query)
		// Pass absolute path of directory to go list so that it knows to treat it as a directory,
		// not a package path.
		pattern, err := filepath.Abs(fdir)
		if err != nil {
			return nil, fmt.Errorf("could not determine absolute path of file= query path %q: %v", query, err)
		}
		if isFallback {
			pattern = "."
			cfg.Dir = fdir
		}

		dirResponse, err := driver(cfg, pattern)
		if err != nil {
			return nil, err
		}
		isRoot := make(map[string]bool, len(dirResponse.Roots))
		for _, root := range dirResponse.Roots {
			isRoot[root] = true
		}
		for _, pkg := range dirResponse.Packages {
			// Add any new packages to the main set
			// We don't bother to filter packages that will be dropped by the changes of roots,
			// that will happen anyway during graph construction outside this function.
			// Over-reporting packages is not a problem.
			addPkg(pkg)
			// if the package was not a root one, it cannot have the file
			if !isRoot[pkg.ID] {
				continue
			}
			for _, pkgFile := range pkg.GoFiles {
				if filepath.Base(query) == filepath.Base(pkgFile) {
					results = append(results, pkg.ID)
					break
				}
			}
		}
	}
	return results, nil
}

// modCacheRegexp splits a path in a module cache into module, module version, and package.
var modCacheRegexp = regexp.MustCompile(`(.*)@([^/\\]*)(.*)`)

func runNamedQueries(cfg *Config, driver driver, addPkg func(*Package), queries []string) ([]string, error) {
	// calling `go env` isn't free; bail out if there's nothing to do.
	if len(queries) == 0 {
		return nil, nil
	}
	// Determine which directories are relevant to scan.
	roots, modRoot, err := roots(cfg)
	if err != nil {
		return nil, err
	}

	// Scan the selected directories. Simple matches, from GOPATH/GOROOT
	// or the local module, can simply be "go list"ed. Matches from the
	// module cache need special treatment.
	var matchesMu sync.Mutex
	var simpleMatches, modCacheMatches []string
	add := func(root gopathwalk.Root, dir string) {
		// Walk calls this concurrently; protect the result slices.
		matchesMu.Lock()
		defer matchesMu.Unlock()

		path := dir[len(root.Path)+1:]
		if pathMatchesQueries(path, queries) {
			switch root.Type {
			case gopathwalk.RootModuleCache:
				modCacheMatches = append(modCacheMatches, path)
			case gopathwalk.RootCurrentModule:
				// We'd need to read go.mod to find the full
				// import path. Relative's easier.
				rel, err := filepath.Rel(cfg.Dir, dir)
				if err != nil {
					// This ought to be impossible, since
					// we found dir in the current module.
					panic(err)
				}
				simpleMatches = append(simpleMatches, "./"+rel)
			case gopathwalk.RootGOPATH, gopathwalk.RootGOROOT:
				simpleMatches = append(simpleMatches, path)
			}
		}
	}

	startWalk := time.Now()
	gopathwalk.Walk(roots, add, gopathwalk.Options{ModulesEnabled: modRoot != "", Debug: debug})
	if debug {
		log.Printf("%v for walk", time.Since(startWalk))
	}

	// Weird special case: the top-level package in a module will be in
	// whatever directory the user checked the repository out into. It's
	// more reasonable for that to not match the package name. So, if there
	// are any Go files in the mod root, query it just to be safe.
	if modRoot != "" {
		rel, err := filepath.Rel(cfg.Dir, modRoot)
		if err != nil {
			panic(err) // See above.
		}

		files, err := ioutil.ReadDir(modRoot)
		for _, f := range files {
			if strings.HasSuffix(f.Name(), ".go") {
				simpleMatches = append(simpleMatches, rel)
				break
			}
		}
	}

	var results []string
	addResponse := func(r *driverResponse) {
		for _, pkg := range r.Packages {
			addPkg(pkg)
			for _, name := range queries {
				if pkg.Name == name {
					results = append(results, pkg.ID)
					break
				}
			}
		}
	}

	if len(simpleMatches) != 0 {
		resp, err := driver(cfg, simpleMatches...)
		if err != nil {
			return nil, err
		}
		addResponse(resp)
	}

	// Module cache matches are tricky. We want to avoid downloading new
	// versions of things, so we need to use the ones present in the cache.
	// go list doesn't accept version specifiers, so we have to write out a
	// temporary module, and do the list in that module.
	if len(modCacheMatches) != 0 {
		// Collect all the matches, deduplicating by major version
		// and preferring the newest.
		type modInfo struct {
			mod   string
			major string
		}
		mods := make(map[modInfo]string)
		var imports []string
		for _, modPath := range modCacheMatches {
			matches := modCacheRegexp.FindStringSubmatch(modPath)
			mod, ver := filepath.ToSlash(matches[1]), matches[2]
			importPath := filepath.ToSlash(filepath.Join(matches[1], matches[3]))

			major := semver.Major(ver)
			if prevVer, ok := mods[modInfo{mod, major}]; !ok || semver.Compare(ver, prevVer) > 0 {
				mods[modInfo{mod, major}] = ver
			}

			imports = append(imports, importPath)
		}

		// Build the temporary module.
		var gomod bytes.Buffer
		gomod.WriteString("module modquery\nrequire (\n")
		for mod, version := range mods {
			gomod.WriteString("\t" + mod.mod + " " + version + "\n")
		}
		gomod.WriteString(")\n")

		tmpCfg := *cfg

		// We're only trying to look at stuff in the module cache, so
		// disable the network. This should speed things up, and has
		// prevented errors in at least one case, #28518.
		tmpCfg.Env = append(append([]string{"GOPROXY=off"}, cfg.Env...))

		var err error
		tmpCfg.Dir, err = ioutil.TempDir("", "gopackages-modquery")
		if err != nil {
			return nil, err
		}
		defer os.RemoveAll(tmpCfg.Dir)

		if err := ioutil.WriteFile(filepath.Join(tmpCfg.Dir, "go.mod"), gomod.Bytes(), 0777); err != nil {
			return nil, fmt.Errorf("writing go.mod for module cache query: %v", err)
		}

		// Run the query, using the import paths calculated from the matches above.
		resp, err := driver(&tmpCfg, imports...)
		if err != nil {
			return nil, fmt.Errorf("querying module cache matches: %v", err)
		}
		addResponse(resp)
	}

	return results, nil
}

func getSizes(cfg *Config) (types.Sizes, error) {
	stdout, err := invokeGo(cfg, "env", "GOARCH")
	if err != nil {
		return nil, err
	}

	goarch := strings.TrimSpace(stdout.String())
	// Assume "gc" because SizesFor doesn't respond to other compilers.
	// TODO(matloob): add support for gccgo as needed.
	return types.SizesFor("gc", goarch), nil
}

// roots selects the appropriate paths to walk based on the passed-in configuration,
// particularly the environment and the presence of a go.mod in cfg.Dir's parents.
func roots(cfg *Config) ([]gopathwalk.Root, string, error) {
	stdout, err := invokeGo(cfg, "env", "GOROOT", "GOPATH", "GOMOD")
	if err != nil {
		return nil, "", err
	}

	fields := strings.Split(stdout.String(), "\n")
	if len(fields) != 4 || len(fields[3]) != 0 {
		return nil, "", fmt.Errorf("go env returned unexpected output: %q", stdout.String())
	}
	goroot, gopath, gomod := fields[0], filepath.SplitList(fields[1]), fields[2]
	var modDir string
	if gomod != "" {
		modDir = filepath.Dir(gomod)
	}

	var roots []gopathwalk.Root
	// Always add GOROOT.
	roots = append(roots, gopathwalk.Root{filepath.Join(goroot, "/src"), gopathwalk.RootGOROOT})
	// If modules are enabled, scan the module dir.
	if modDir != "" {
		roots = append(roots, gopathwalk.Root{modDir, gopathwalk.RootCurrentModule})
	}
	// Add either GOPATH/src or GOPATH/pkg/mod, depending on module mode.
	for _, p := range gopath {
		if modDir != "" {
			roots = append(roots, gopathwalk.Root{filepath.Join(p, "/pkg/mod"), gopathwalk.RootModuleCache})
		} else {
			roots = append(roots, gopathwalk.Root{filepath.Join(p, "/src"), gopathwalk.RootGOPATH})
		}
	}

	return roots, modDir, nil
}

// These functions were copied from goimports. See further documentation there.

// pathMatchesQueries is adapted from pkgIsCandidate.
// TODO: is it reasonable to do Contains here, rather than an exact match on a path component?
func pathMatchesQueries(path string, queries []string) bool {
	lastTwo := lastTwoComponents(path)
	for _, query := range queries {
		if strings.Contains(lastTwo, query) {
			return true
		}
		if hasHyphenOrUpperASCII(lastTwo) && !hasHyphenOrUpperASCII(query) {
			lastTwo = lowerASCIIAndRemoveHyphen(lastTwo)
			if strings.Contains(lastTwo, query) {
				return true
			}
		}
	}
	return false
}

// lastTwoComponents returns at most the last two path components
// of v, using either / or \ as the path separator.
func lastTwoComponents(v string) string {
	nslash := 0
	for i := len(v) - 1; i >= 0; i-- {
		if v[i] == '/' || v[i] == '\\' {
			nslash++
			if nslash == 2 {
				return v[i:]
			}
		}
	}
	return v
}

func hasHyphenOrUpperASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b == '-' || ('A' <= b && b <= 'Z') {
			return true
		}
	}
	return false
}

func lowerASCIIAndRemoveHyphen(s string) (ret string) {
	buf := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		b := s[i]
		switch {
		case b == '-':
			continue
		case 'A' <= b && b <= 'Z':
			buf = append(buf, b+('a'-'A'))
		default:
			buf = append(buf, b)
		}
	}
	return string(buf)
}

// Fields must match go list;
// see $GOROOT/src/cmd/go/internal/load/pkg.go.
type jsonPackage struct {
	ImportPath      string
	Dir             string
	Name            string
	Export          string
	GoFiles         []string
	CompiledGoFiles []string
	CFiles          []string
	CgoFiles        []string
	CXXFiles        []string
	MFiles          []string
	HFiles          []string
	FFiles          []string
	SFiles          []string
	SwigFiles       []string
	SwigCXXFiles    []string
	SysoFiles       []string
	Imports         []string
	ImportMap       map[string]string
	Deps            []string
	TestGoFiles     []string
	TestImports     []string
	XTestGoFiles    []string
	XTestImports    []string
	ForTest         string // q in a "p [q.test]" package, else ""
	DepOnly         bool

	Error *jsonPackageError
}

type jsonPackageError struct {
	ImportStack []string
	Pos         string
	Err         string
}

func otherFiles(p *jsonPackage) [][]string {
	return [][]string{p.CFiles, p.CXXFiles, p.MFiles, p.HFiles, p.FFiles, p.SFiles, p.SwigFiles, p.SwigCXXFiles, p.SysoFiles}
}

// golistDriverCurrent uses the "go list" command to expand the
// pattern words and return metadata for the specified packages.
// dir may be "" and env may be nil, as per os/exec.Command.
func golistDriverCurrent(cfg *Config, words ...string) (*driverResponse, error) {
	// go list uses the following identifiers in ImportPath and Imports:
	//
	// 	"p"			-- importable package or main (command)
	//      "q.test"		-- q's test executable
	// 	"p [q.test]"		-- variant of p as built for q's test executable
	//	"q_test [q.test]"	-- q's external test package
	//
	// The packages p that are built differently for a test q.test
	// are q itself, plus any helpers used by the external test q_test,
	// typically including "testing" and all its dependencies.

	// Run "go list" for complete
	// information on the specified packages.
	buf, err := invokeGo(cfg, golistargs(cfg, words)...)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]*jsonPackage)
	// Decode the JSON and convert it to Package form.
	var response driverResponse
	for dec := json.NewDecoder(buf); dec.More(); {
		p := new(jsonPackage)
		if err := dec.Decode(p); err != nil {
			return nil, fmt.Errorf("JSON decoding failed: %v", err)
		}

		if p.ImportPath == "" {
			// The documentation for go list says that “[e]rroneous packages will have
			// a non-empty ImportPath”. If for some reason it comes back empty, we
			// prefer to error out rather than silently discarding data or handing
			// back a package without any way to refer to it.
			if p.Error != nil {
				return nil, Error{
					Pos: p.Error.Pos,
					Msg: p.Error.Err,
				}
			}
			return nil, fmt.Errorf("package missing import path: %+v", p)
		}

		if old, found := seen[p.ImportPath]; found {
			if !reflect.DeepEqual(p, old) {
				return nil, fmt.Errorf("go list repeated package %v with different values", p.ImportPath)
			}
			// skip the duplicate
			continue
		}
		seen[p.ImportPath] = p

		pkg := &Package{
			Name:            p.Name,
			ID:              p.ImportPath,
			GoFiles:         absJoin(p.Dir, p.GoFiles, p.CgoFiles),
			CompiledGoFiles: absJoin(p.Dir, p.CompiledGoFiles),
			OtherFiles:      absJoin(p.Dir, otherFiles(p)...),
		}

		// Workaround for https://golang.org/issue/28749.
		// TODO(adonovan): delete before go1.12 release.
		out := pkg.CompiledGoFiles[:0]
		for _, f := range pkg.CompiledGoFiles {
			if strings.HasSuffix(f, ".s") {
				continue
			}
			out = append(out, f)
		}
		pkg.CompiledGoFiles = out

		// Extract the PkgPath from the package's ID.
		if i := strings.IndexByte(pkg.ID, ' '); i >= 0 {
			pkg.PkgPath = pkg.ID[:i]
		} else {
			pkg.PkgPath = pkg.ID
		}

		if pkg.PkgPath == "unsafe" {
			pkg.GoFiles = nil // ignore fake unsafe.go file
		}

		// Assume go list emits only absolute paths for Dir.
		if p.Dir != "" && !filepath.IsAbs(p.Dir) {
			log.Fatalf("internal error: go list returned non-absolute Package.Dir: %s", p.Dir)
		}

		if p.Export != "" && !filepath.IsAbs(p.Export) {
			pkg.ExportFile = filepath.Join(p.Dir, p.Export)
		} else {
			pkg.ExportFile = p.Export
		}

		// imports
		//
		// Imports contains the IDs of all imported packages.
		// ImportsMap records (path, ID) only where they differ.
		ids := make(map[string]bool)
		for _, id := range p.Imports {
			ids[id] = true
		}
		pkg.Imports = make(map[string]*Package)
		for path, id := range p.ImportMap {
			pkg.Imports[path] = &Package{ID: id} // non-identity import
			delete(ids, id)
		}
		for id := range ids {
			if id == "C" {
				continue
			}

			pkg.Imports[id] = &Package{ID: id} // identity import
		}
		if !p.DepOnly {
			response.Roots = append(response.Roots, pkg.ID)
		}

		// Work around for pre-go.1.11 versions of go list.
		// TODO(matloob): they should be handled by the fallback.
		// Can we delete this?
		if len(pkg.CompiledGoFiles) == 0 {
			pkg.CompiledGoFiles = pkg.GoFiles
		}

		if p.Error != nil {
			pkg.Errors = append(pkg.Errors, Error{
				Pos: p.Error.Pos,
				Msg: p.Error.Err,
			})
		}

		response.Packages = append(response.Packages, pkg)
	}

	return &response, nil
}

// absJoin absolutizes and flattens the lists of files.
func absJoin(dir string, fileses ...[]string) (res []string) {
	for _, files := range fileses {
		for _, file := range files {
			if !filepath.IsAbs(file) {
				file = filepath.Join(dir, file)
			}
			res = append(res, file)
		}
	}
	return res
}

func golistargs(cfg *Config, words []string) []string {
	fullargs := []string{
		"list", "-e", "-json", "-compiled",
		fmt.Sprintf("-test=%t", cfg.Tests),
		fmt.Sprintf("-export=%t", usesExportData(cfg)),
		fmt.Sprintf("-deps=%t", cfg.Mode >= LoadImports),
	}
	fullargs = append(fullargs, cfg.BuildFlags...)
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	return fullargs
}

// invokeGo returns the stdout of a go command invocation.
func invokeGo(cfg *Config, args ...string) (*bytes.Buffer, error) {
	if debug {
		defer func(start time.Time) { log.Printf("%s for %v", time.Since(start), cmdDebugStr(cfg, args...)) }(time.Now())
	}
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)
	cmd := exec.CommandContext(cfg.Context, "go", args...)
	// On darwin the cwd gets resolved to the real path, which breaks anything that
	// expects the working directory to keep the original path, including the
	// go command when dealing with modules.
	// The Go stdlib has a special feature where if the cwd and the PWD are the
	// same node then it trusts the PWD, so by setting it in the env for the child
	// process we fix up all the paths returned by the go command.
	cmd.Env = append(append([]string{}, cfg.Env...), "PWD="+cfg.Dir)
	cmd.Dir = cfg.Dir
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			// Catastrophic error:
			// - executable not found
			// - context cancellation
			return nil, fmt.Errorf("couldn't exec 'go %v': %s %T", args, err, err)
		}

		// Old go version?
		if strings.Contains(stderr.String(), "flag provided but not defined") {
			return nil, goTooOldError{fmt.Errorf("unsupported version of go: %s: %s", exitErr, stderr)}
		}

		// Export mode entails a build.
		// If that build fails, errors appear on stderr
		// (despite the -e flag) and the Export field is blank.
		// Do not fail in that case.
		if !usesExportData(cfg) {
			return nil, fmt.Errorf("go %v: %s: %s", args, exitErr, stderr)
		}
	}

	// As of writing, go list -export prints some non-fatal compilation
	// errors to stderr, even with -e set. We would prefer that it put
	// them in the Package.Error JSON (see https://golang.org/issue/26319).
	// In the meantime, there's nowhere good to put them, but they can
	// be useful for debugging. Print them if $GOPACKAGESPRINTGOLISTERRORS
	// is set.
	if len(stderr.Bytes()) != 0 && os.Getenv("GOPACKAGESPRINTGOLISTERRORS") != "" {
		fmt.Fprintf(os.Stderr, "%s stderr: <<%s>>\n", cmdDebugStr(cfg, args...), stderr)
	}

	// debugging
	if false {
		fmt.Fprintf(os.Stderr, "%s stdout: <<%s>>\n", cmdDebugStr(cfg, args...), stdout)
	}

	return stdout, nil
}

func cmdDebugStr(cfg *Config, args ...string) string {
	env := make(map[string]string)
	for _, kv := range cfg.Env {
		split := strings.Split(kv, "=")
		k, v := split[0], split[1]
		env[k] = v
	}

	return fmt.Sprintf("GOROOT=%v GOPATH=%v GO111MODULE=%v PWD=%v go %v", env["GOROOT"], env["GOPATH"], env["GO111MODULE"], env["PWD"], args)
}
