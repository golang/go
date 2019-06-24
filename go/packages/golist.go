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
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/internal/packagesdriver"
	"golang.org/x/tools/internal/gopathwalk"
	"golang.org/x/tools/internal/semver"
)

// debug controls verbose logging.
var debug, _ = strconv.ParseBool(os.Getenv("GOPACKAGESDEBUG"))

// A goTooOldError reports that the go command
// found by exec.LookPath is too old to use the new go list behavior.
type goTooOldError struct {
	error
}

// responseDeduper wraps a driverResponse, deduplicating its contents.
type responseDeduper struct {
	seenRoots    map[string]bool
	seenPackages map[string]*Package
	dr           *driverResponse
}

// init fills in r with a driverResponse.
func (r *responseDeduper) init(dr *driverResponse) {
	r.dr = dr
	r.seenRoots = map[string]bool{}
	r.seenPackages = map[string]*Package{}
	for _, pkg := range dr.Packages {
		r.seenPackages[pkg.ID] = pkg
	}
	for _, root := range dr.Roots {
		r.seenRoots[root] = true
	}
}

func (r *responseDeduper) addPackage(p *Package) {
	if r.seenPackages[p.ID] != nil {
		return
	}
	r.seenPackages[p.ID] = p
	r.dr.Packages = append(r.dr.Packages, p)
}

func (r *responseDeduper) addRoot(id string) {
	if r.seenRoots[id] {
		return
	}
	r.seenRoots[id] = true
	r.dr.Roots = append(r.dr.Roots, id)
}

// goListDriver uses the go list command to interpret the patterns and produce
// the build system package structure.
// See driver for more details.
func goListDriver(cfg *Config, patterns ...string) (*driverResponse, error) {
	var sizes types.Sizes
	var sizeserr error
	var sizeswg sync.WaitGroup
	if cfg.Mode&NeedTypesSizes != 0 || cfg.Mode&NeedTypes != 0 {
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
			case "iamashamedtousethedisabledqueryname":
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

	response := &responseDeduper{}
	var err error

	// See if we have any patterns to pass through to go list. Zero initial
	// patterns also requires a go list call, since it's the equivalent of
	// ".".
	if len(restPatterns) > 0 || len(patterns) == 0 {
		dr, err := golistDriver(cfg, restPatterns...)
		if err != nil {
			return nil, err
		}
		response.init(dr)
	} else {
		response.init(&driverResponse{})
	}

	sizeswg.Wait()
	if sizeserr != nil {
		return nil, sizeserr
	}
	// types.SizesFor always returns nil or a *types.StdSizes
	response.dr.Sizes, _ = sizes.(*types.StdSizes)

	var containsCandidates []string

	if len(containFiles) != 0 {
		if err := runContainsQueries(cfg, golistDriver, response, containFiles); err != nil {
			return nil, err
		}
	}

	if len(packagesNamed) != 0 {
		if err := runNamedQueries(cfg, golistDriver, response, packagesNamed); err != nil {
			return nil, err
		}
	}

	modifiedPkgs, needPkgs, err := processGolistOverlay(cfg, response)
	if err != nil {
		return nil, err
	}
	if len(containFiles) > 0 {
		containsCandidates = append(containsCandidates, modifiedPkgs...)
		containsCandidates = append(containsCandidates, needPkgs...)
	}
	if err := addNeededOverlayPackages(cfg, golistDriver, response, needPkgs); err != nil {
		return nil, err
	}
	// Check candidate packages for containFiles.
	if len(containFiles) > 0 {
		for _, id := range containsCandidates {
			pkg, ok := response.seenPackages[id]
			if !ok {
				response.addPackage(&Package{
					ID: id,
					Errors: []Error{
						{
							Kind: ListError,
							Msg:  fmt.Sprintf("package %s expected but not seen", id),
						},
					},
				})
				continue
			}
			for _, f := range containFiles {
				for _, g := range pkg.GoFiles {
					if sameFile(f, g) {
						response.addRoot(id)
					}
				}
			}
		}
	}

	return response.dr, nil
}

func addNeededOverlayPackages(cfg *Config, driver driver, response *responseDeduper, pkgs []string) error {
	if len(pkgs) == 0 {
		return nil
	}
	dr, err := driver(cfg, pkgs...)
	if err != nil {
		return err
	}
	for _, pkg := range dr.Packages {
		response.addPackage(pkg)
	}
	_, needPkgs, err := processGolistOverlay(cfg, response)
	if err != nil {
		return err
	}
	addNeededOverlayPackages(cfg, driver, response, needPkgs)
	return nil
}

func runContainsQueries(cfg *Config, driver driver, response *responseDeduper, queries []string) error {
	for _, query := range queries {
		// TODO(matloob): Do only one query per directory.
		fdir := filepath.Dir(query)
		// Pass absolute path of directory to go list so that it knows to treat it as a directory,
		// not a package path.
		pattern, err := filepath.Abs(fdir)
		if err != nil {
			return fmt.Errorf("could not determine absolute path of file= query path %q: %v", query, err)
		}
		dirResponse, err := driver(cfg, pattern)
		if err != nil {
			// Couldn't find a package for the directory. Try to load the file as an ad-hoc package.
			var queryErr error
			dirResponse, queryErr = driver(cfg, query)
			if queryErr != nil {
				// Return the original error if the attempt to fall back failed.
				return err
			}
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
			response.addPackage(pkg)
			// if the package was not a root one, it cannot have the file
			if !isRoot[pkg.ID] {
				continue
			}
			for _, pkgFile := range pkg.GoFiles {
				if filepath.Base(query) == filepath.Base(pkgFile) {
					response.addRoot(pkg.ID)
					break
				}
			}
		}
	}
	return nil
}

// modCacheRegexp splits a path in a module cache into module, module version, and package.
var modCacheRegexp = regexp.MustCompile(`(.*)@([^/\\]*)(.*)`)

func runNamedQueries(cfg *Config, driver driver, response *responseDeduper, queries []string) error {
	// calling `go env` isn't free; bail out if there's nothing to do.
	if len(queries) == 0 {
		return nil
	}
	// Determine which directories are relevant to scan.
	roots, modRoot, err := roots(cfg)
	if err != nil {
		return err
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

		path := dir
		if dir != root.Path {
			path = dir[len(root.Path)+1:]
		}
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

	addResponse := func(r *driverResponse) {
		for _, pkg := range r.Packages {
			response.addPackage(pkg)
			for _, name := range queries {
				if pkg.Name == name {
					response.addRoot(pkg.ID)
					break
				}
			}
		}
	}

	if len(simpleMatches) != 0 {
		resp, err := driver(cfg, simpleMatches...)
		if err != nil {
			return err
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
			return err
		}
		defer os.RemoveAll(tmpCfg.Dir)

		if err := ioutil.WriteFile(filepath.Join(tmpCfg.Dir, "go.mod"), gomod.Bytes(), 0777); err != nil {
			return fmt.Errorf("writing go.mod for module cache query: %v", err)
		}

		// Run the query, using the import paths calculated from the matches above.
		resp, err := driver(&tmpCfg, imports...)
		if err != nil {
			return fmt.Errorf("querying module cache matches: %v", err)
		}
		addResponse(resp)
	}

	return nil
}

func getSizes(cfg *Config) (types.Sizes, error) {
	return packagesdriver.GetSizesGolist(cfg.Context, cfg.BuildFlags, cfg.Env, cfg.Dir, usesExportData(cfg))
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

// golistDriver uses the "go list" command to expand the pattern
// words and return metadata for the specified packages. dir may be
// "" and env may be nil, as per os/exec.Command.
func golistDriver(cfg *Config, words ...string) (*driverResponse, error) {
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
				return nil, fmt.Errorf("internal error: go list gives conflicting information for package %v", p.ImportPath)
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

		// Work around https://golang.org/issue/28749:
		// cmd/go puts assembly, C, and C++ files in CompiledGoFiles.
		// Filter out any elements of CompiledGoFiles that are also in OtherFiles.
		// We have to keep this workaround in place until go1.12 is a distant memory.
		if len(pkg.OtherFiles) > 0 {
			other := make(map[string]bool, len(pkg.OtherFiles))
			for _, f := range pkg.OtherFiles {
				other[f] = true
			}

			out := pkg.CompiledGoFiles[:0]
			for _, f := range pkg.CompiledGoFiles {
				if other[f] {
					continue
				}
				out = append(out, f)
			}
			pkg.CompiledGoFiles = out
		}

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
				Msg: strings.TrimSpace(p.Error.Err), // Trim to work around golang.org/issue/32363.
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
	const findFlags = NeedImports | NeedTypes | NeedSyntax | NeedTypesInfo
	fullargs := []string{
		"list", "-e", "-json",
		fmt.Sprintf("-compiled=%t", cfg.Mode&(NeedCompiledGoFiles|NeedSyntax|NeedTypesInfo|NeedTypesSizes) != 0),
		fmt.Sprintf("-test=%t", cfg.Tests),
		fmt.Sprintf("-export=%t", usesExportData(cfg)),
		fmt.Sprintf("-deps=%t", cfg.Mode&NeedDeps != 0),
		// go list doesn't let you pass -test and -find together,
		// probably because you'd just get the TestMain.
		fmt.Sprintf("-find=%t", !cfg.Tests && cfg.Mode&findFlags == 0),
	}
	fullargs = append(fullargs, cfg.BuildFlags...)
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	return fullargs
}

// invokeGo returns the stdout of a go command invocation.
func invokeGo(cfg *Config, args ...string) (*bytes.Buffer, error) {
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
	if debug {
		defer func(start time.Time) {
			log.Printf("%s for %v, stderr: <<%s>>\n", time.Since(start), cmdDebugStr(cmd, args...), stderr)
		}(time.Now())
	}

	if err := cmd.Run(); err != nil {
		// Check for 'go' executable not being found.
		if ee, ok := err.(*exec.Error); ok && ee.Err == exec.ErrNotFound {
			return nil, fmt.Errorf("'go list' driver requires 'go', but %s", exec.ErrNotFound)
		}

		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			// Catastrophic error:
			// - context cancellation
			return nil, fmt.Errorf("couldn't exec 'go %v': %s %T", args, err, err)
		}

		// Old go version?
		if strings.Contains(stderr.String(), "flag provided but not defined") {
			return nil, goTooOldError{fmt.Errorf("unsupported version of go: %s: %s", exitErr, stderr)}
		}

		// This error only appears in stderr. See golang.org/cl/166398 for a fix in go list to show
		// the error in the Err section of stdout in case -e option is provided.
		// This fix is provided for backwards compatibility.
		if len(stderr.String()) > 0 && strings.Contains(stderr.String(), "named files must be .go files") {
			output := fmt.Sprintf(`{"ImportPath": "command-line-arguments","Incomplete": true,"Error": {"Pos": "","Err": %q}}`,
				strings.Trim(stderr.String(), "\n"))
			return bytes.NewBufferString(output), nil
		}

		// Workaround for #29280: go list -e has incorrect behavior when an ad-hoc package doesn't exist.
		if len(stderr.String()) > 0 && strings.Contains(stderr.String(), "no such file or directory") {
			output := fmt.Sprintf(`{"ImportPath": "command-line-arguments","Incomplete": true,"Error": {"Pos": "","Err": %q}}`,
				strings.Trim(stderr.String(), "\n"))
			return bytes.NewBufferString(output), nil
		}

		// Workaround for an instance of golang.org/issue/26755: go list -e  will return a non-zero exit
		// status if there's a dependency on a package that doesn't exist. But it should return
		// a zero exit status and set an error on that package.
		if len(stderr.String()) > 0 && strings.Contains(stderr.String(), "no Go files in") {
			// try to extract package name from string
			stderrStr := stderr.String()
			var importPath string
			colon := strings.Index(stderrStr, ":")
			if colon > 0 && strings.HasPrefix(stderrStr, "go build ") {
				importPath = stderrStr[len("go build "):colon]
			}
			output := fmt.Sprintf(`{"ImportPath": %q,"Incomplete": true,"Error": {"Pos": "","Err": %q}}`,
				importPath, strings.Trim(stderrStr, "\n"))
			return bytes.NewBufferString(output), nil
		}

		// Export mode entails a build.
		// If that build fails, errors appear on stderr
		// (despite the -e flag) and the Export field is blank.
		// Do not fail in that case.
		// The same is true if an ad-hoc package given to go list doesn't exist.
		// TODO(matloob): Remove these once we can depend on go list to exit with a zero status with -e even when
		// packages don't exist or a build fails.
		if !usesExportData(cfg) && !containsGoFile(args) {
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
		fmt.Fprintf(os.Stderr, "%s stderr: <<%s>>\n", cmdDebugStr(cmd, args...), stderr)
	}

	// debugging
	if false {
		fmt.Fprintf(os.Stderr, "%s stdout: <<%s>>\n", cmdDebugStr(cmd, args...), stdout)
	}

	return stdout, nil
}

func containsGoFile(s []string) bool {
	for _, f := range s {
		if strings.HasSuffix(f, ".go") {
			return true
		}
	}
	return false
}

func cmdDebugStr(cmd *exec.Cmd, args ...string) string {
	env := make(map[string]string)
	for _, kv := range cmd.Env {
		split := strings.Split(kv, "=")
		k, v := split[0], split[1]
		env[k] = v
	}
	var quotedArgs []string
	for _, arg := range args {
		quotedArgs = append(quotedArgs, strconv.Quote(arg))
	}

	return fmt.Sprintf("GOROOT=%v GOPATH=%v GO111MODULE=%v PWD=%v go %s", env["GOROOT"], env["GOPATH"], env["GO111MODULE"], env["PWD"], strings.Join(quotedArgs, " "))
}
