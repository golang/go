// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

// See doc.go for package documentation and implementation notes.

import (
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/gcexportdata"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/typesinternal"
)

// A LoadMode controls the amount of detail to return when loading.
// The bits below can be combined to specify which fields should be
// filled in the result packages.
// The zero value is a special case, equivalent to combining
// the NeedName, NeedFiles, and NeedCompiledGoFiles bits.
// ID and Errors (if present) will always be filled.
// Load may return more information than requested.
type LoadMode int

// TODO(matloob): When a V2 of go/packages is released, rename NeedExportsFile to
// NeedExportFile to make it consistent with the Package field it's adding.

const (
	// NeedName adds Name and PkgPath.
	NeedName LoadMode = 1 << iota

	// NeedFiles adds GoFiles and OtherFiles.
	NeedFiles

	// NeedCompiledGoFiles adds CompiledGoFiles.
	NeedCompiledGoFiles

	// NeedImports adds Imports. If NeedDeps is not set, the Imports field will contain
	// "placeholder" Packages with only the ID set.
	NeedImports

	// NeedDeps adds the fields requested by the LoadMode in the packages in Imports.
	NeedDeps

	// NeedExportsFile adds ExportFile.
	NeedExportsFile

	// NeedTypes adds Types, Fset, and IllTyped.
	NeedTypes

	// NeedSyntax adds Syntax.
	NeedSyntax

	// NeedTypesInfo adds TypesInfo.
	NeedTypesInfo

	// NeedTypesSizes adds TypesSizes.
	NeedTypesSizes

	// typecheckCgo enables full support for type checking cgo. Requires Go 1.15+.
	// Modifies CompiledGoFiles and Types, and has no effect on its own.
	typecheckCgo

	// NeedModule adds Module.
	NeedModule
)

const (
	// Deprecated: LoadFiles exists for historical compatibility
	// and should not be used. Please directly specify the needed fields using the Need values.
	LoadFiles = NeedName | NeedFiles | NeedCompiledGoFiles

	// Deprecated: LoadImports exists for historical compatibility
	// and should not be used. Please directly specify the needed fields using the Need values.
	LoadImports = LoadFiles | NeedImports

	// Deprecated: LoadTypes exists for historical compatibility
	// and should not be used. Please directly specify the needed fields using the Need values.
	LoadTypes = LoadImports | NeedTypes | NeedTypesSizes

	// Deprecated: LoadSyntax exists for historical compatibility
	// and should not be used. Please directly specify the needed fields using the Need values.
	LoadSyntax = LoadTypes | NeedSyntax | NeedTypesInfo

	// Deprecated: LoadAllSyntax exists for historical compatibility
	// and should not be used. Please directly specify the needed fields using the Need values.
	LoadAllSyntax = LoadSyntax | NeedDeps
)

// A Config specifies details about how packages should be loaded.
// The zero value is a valid configuration.
// Calls to Load do not modify this struct.
type Config struct {
	// Mode controls the level of information returned for each package.
	Mode LoadMode

	// Context specifies the context for the load operation.
	// If the context is cancelled, the loader may stop early
	// and return an ErrCancelled error.
	// If Context is nil, the load cannot be cancelled.
	Context context.Context

	// Logf is the logger for the config.
	// If the user provides a logger, debug logging is enabled.
	// If the GOPACKAGESDEBUG environment variable is set to true,
	// but the logger is nil, default to log.Printf.
	Logf func(format string, args ...interface{})

	// Dir is the directory in which to run the build system's query tool
	// that provides information about the packages.
	// If Dir is empty, the tool is run in the current directory.
	Dir string

	// Env is the environment to use when invoking the build system's query tool.
	// If Env is nil, the current environment is used.
	// As in os/exec's Cmd, only the last value in the slice for
	// each environment key is used. To specify the setting of only
	// a few variables, append to the current environment, as in:
	//
	//	opt.Env = append(os.Environ(), "GOOS=plan9", "GOARCH=386")
	//
	Env []string

	// gocmdRunner guards go command calls from concurrency errors.
	gocmdRunner *gocommand.Runner

	// BuildFlags is a list of command-line flags to be passed through to
	// the build system's query tool.
	BuildFlags []string

	// modFile will be used for -modfile in go command invocations.
	modFile string

	// modFlag will be used for -modfile in go command invocations.
	modFlag string

	// Fset provides source position information for syntax trees and types.
	// If Fset is nil, Load will use a new fileset, but preserve Fset's value.
	Fset *token.FileSet

	// ParseFile is called to read and parse each file
	// when preparing a package's type-checked syntax tree.
	// It must be safe to call ParseFile simultaneously from multiple goroutines.
	// If ParseFile is nil, the loader will uses parser.ParseFile.
	//
	// ParseFile should parse the source from src and use filename only for
	// recording position information.
	//
	// An application may supply a custom implementation of ParseFile
	// to change the effective file contents or the behavior of the parser,
	// or to modify the syntax tree. For example, selectively eliminating
	// unwanted function bodies can significantly accelerate type checking.
	ParseFile func(fset *token.FileSet, filename string, src []byte) (*ast.File, error)

	// If Tests is set, the loader includes not just the packages
	// matching a particular pattern but also any related test packages,
	// including test-only variants of the package and the test executable.
	//
	// For example, when using the go command, loading "fmt" with Tests=true
	// returns four packages, with IDs "fmt" (the standard package),
	// "fmt [fmt.test]" (the package as compiled for the test),
	// "fmt_test" (the test functions from source files in package fmt_test),
	// and "fmt.test" (the test binary).
	//
	// In build systems with explicit names for tests,
	// setting Tests may have no effect.
	Tests bool

	// Overlay provides a mapping of absolute file paths to file contents.
	// If the file with the given path already exists, the parser will use the
	// alternative file contents provided by the map.
	//
	// Overlays provide incomplete support for when a given file doesn't
	// already exist on disk. See the package doc above for more details.
	Overlay map[string][]byte
}

// driver is the type for functions that query the build system for the
// packages named by the patterns.
type driver func(cfg *Config, patterns ...string) (*driverResponse, error)

// driverResponse contains the results for a driver query.
type driverResponse struct {
	// NotHandled is returned if the request can't be handled by the current
	// driver. If an external driver returns a response with NotHandled, the
	// rest of the driverResponse is ignored, and go/packages will fallback
	// to the next driver. If go/packages is extended in the future to support
	// lists of multiple drivers, go/packages will fall back to the next driver.
	NotHandled bool

	// Sizes, if not nil, is the types.Sizes to use when type checking.
	Sizes *types.StdSizes

	// Roots is the set of package IDs that make up the root packages.
	// We have to encode this separately because when we encode a single package
	// we cannot know if it is one of the roots as that requires knowledge of the
	// graph it is part of.
	Roots []string `json:",omitempty"`

	// Packages is the full set of packages in the graph.
	// The packages are not connected into a graph.
	// The Imports if populated will be stubs that only have their ID set.
	// Imports will be connected and then type and syntax information added in a
	// later pass (see refine).
	Packages []*Package
}

// Load loads and returns the Go packages named by the given patterns.
//
// Config specifies loading options;
// nil behaves the same as an empty Config.
//
// Load returns an error if any of the patterns was invalid
// as defined by the underlying build system.
// It may return an empty list of packages without an error,
// for instance for an empty expansion of a valid wildcard.
// Errors associated with a particular package are recorded in the
// corresponding Package's Errors list, and do not cause Load to
// return an error. Clients may need to handle such errors before
// proceeding with further analysis. The PrintErrors function is
// provided for convenient display of all errors.
func Load(cfg *Config, patterns ...string) ([]*Package, error) {
	l := newLoader(cfg)
	response, err := defaultDriver(&l.Config, patterns...)
	if err != nil {
		return nil, err
	}
	l.sizes = response.Sizes
	return l.refine(response.Roots, response.Packages...)
}

// defaultDriver is a driver that implements go/packages' fallback behavior.
// It will try to request to an external driver, if one exists. If there's
// no external driver, or the driver returns a response with NotHandled set,
// defaultDriver will fall back to the go list driver.
func defaultDriver(cfg *Config, patterns ...string) (*driverResponse, error) {
	driver := findExternalDriver(cfg)
	if driver == nil {
		driver = goListDriver
	}
	response, err := driver(cfg, patterns...)
	if err != nil {
		return response, err
	} else if response.NotHandled {
		return goListDriver(cfg, patterns...)
	}
	return response, nil
}

// A Package describes a loaded Go package.
type Package struct {
	// ID is a unique identifier for a package,
	// in a syntax provided by the underlying build system.
	//
	// Because the syntax varies based on the build system,
	// clients should treat IDs as opaque and not attempt to
	// interpret them.
	ID string

	// Name is the package name as it appears in the package source code.
	Name string

	// PkgPath is the package path as used by the go/types package.
	PkgPath string

	// Errors contains any errors encountered querying the metadata
	// of the package, or while parsing or type-checking its files.
	Errors []Error

	// GoFiles lists the absolute file paths of the package's Go source files.
	GoFiles []string

	// CompiledGoFiles lists the absolute file paths of the package's source
	// files that are suitable for type checking.
	// This may differ from GoFiles if files are processed before compilation.
	CompiledGoFiles []string

	// OtherFiles lists the absolute file paths of the package's non-Go source files,
	// including assembly, C, C++, Fortran, Objective-C, SWIG, and so on.
	OtherFiles []string

	// IgnoredFiles lists source files that are not part of the package
	// using the current build configuration but that might be part of
	// the package using other build configurations.
	IgnoredFiles []string

	// ExportFile is the absolute path to a file containing type
	// information for the package as provided by the build system.
	ExportFile string

	// Imports maps import paths appearing in the package's Go source files
	// to corresponding loaded Packages.
	Imports map[string]*Package

	// Types provides type information for the package.
	// The NeedTypes LoadMode bit sets this field for packages matching the
	// patterns; type information for dependencies may be missing or incomplete,
	// unless NeedDeps and NeedImports are also set.
	Types *types.Package

	// Fset provides position information for Types, TypesInfo, and Syntax.
	// It is set only when Types is set.
	Fset *token.FileSet

	// IllTyped indicates whether the package or any dependency contains errors.
	// It is set only when Types is set.
	IllTyped bool

	// Syntax is the package's syntax trees, for the files listed in CompiledGoFiles.
	//
	// The NeedSyntax LoadMode bit populates this field for packages matching the patterns.
	// If NeedDeps and NeedImports are also set, this field will also be populated
	// for dependencies.
	Syntax []*ast.File

	// TypesInfo provides type information about the package's syntax trees.
	// It is set only when Syntax is set.
	TypesInfo *types.Info

	// TypesSizes provides the effective size function for types in TypesInfo.
	TypesSizes types.Sizes

	// forTest is the package under test, if any.
	forTest string

	// depsErrors is the DepsErrors field from the go list response, if any.
	depsErrors []*packagesinternal.PackageError

	// module is the module information for the package if it exists.
	Module *Module
}

// Module provides module information for a package.
type Module struct {
	Path      string       // module path
	Version   string       // module version
	Replace   *Module      // replaced by this module
	Time      *time.Time   // time version was created
	Main      bool         // is this the main module?
	Indirect  bool         // is this module only an indirect dependency of main module?
	Dir       string       // directory holding files for this module, if any
	GoMod     string       // path to go.mod file used when loading this module, if any
	GoVersion string       // go version used in module
	Error     *ModuleError // error loading module
}

// ModuleError holds errors loading a module.
type ModuleError struct {
	Err string // the error itself
}

func init() {
	packagesinternal.GetForTest = func(p interface{}) string {
		return p.(*Package).forTest
	}
	packagesinternal.GetDepsErrors = func(p interface{}) []*packagesinternal.PackageError {
		return p.(*Package).depsErrors
	}
	packagesinternal.GetGoCmdRunner = func(config interface{}) *gocommand.Runner {
		return config.(*Config).gocmdRunner
	}
	packagesinternal.SetGoCmdRunner = func(config interface{}, runner *gocommand.Runner) {
		config.(*Config).gocmdRunner = runner
	}
	packagesinternal.SetModFile = func(config interface{}, value string) {
		config.(*Config).modFile = value
	}
	packagesinternal.SetModFlag = func(config interface{}, value string) {
		config.(*Config).modFlag = value
	}
	packagesinternal.TypecheckCgo = int(typecheckCgo)
}

// An Error describes a problem with a package's metadata, syntax, or types.
type Error struct {
	Pos  string // "file:line:col" or "file:line" or "" or "-"
	Msg  string
	Kind ErrorKind
}

// ErrorKind describes the source of the error, allowing the user to
// differentiate between errors generated by the driver, the parser, or the
// type-checker.
type ErrorKind int

const (
	UnknownError ErrorKind = iota
	ListError
	ParseError
	TypeError
)

func (err Error) Error() string {
	pos := err.Pos
	if pos == "" {
		pos = "-" // like token.Position{}.String()
	}
	return pos + ": " + err.Msg
}

// flatPackage is the JSON form of Package
// It drops all the type and syntax fields, and transforms the Imports
//
// TODO(adonovan): identify this struct with Package, effectively
// publishing the JSON protocol.
type flatPackage struct {
	ID              string
	Name            string            `json:",omitempty"`
	PkgPath         string            `json:",omitempty"`
	Errors          []Error           `json:",omitempty"`
	GoFiles         []string          `json:",omitempty"`
	CompiledGoFiles []string          `json:",omitempty"`
	OtherFiles      []string          `json:",omitempty"`
	IgnoredFiles    []string          `json:",omitempty"`
	ExportFile      string            `json:",omitempty"`
	Imports         map[string]string `json:",omitempty"`
}

// MarshalJSON returns the Package in its JSON form.
// For the most part, the structure fields are written out unmodified, and
// the type and syntax fields are skipped.
// The imports are written out as just a map of path to package id.
// The errors are written using a custom type that tries to preserve the
// structure of error types we know about.
//
// This method exists to enable support for additional build systems.  It is
// not intended for use by clients of the API and we may change the format.
func (p *Package) MarshalJSON() ([]byte, error) {
	flat := &flatPackage{
		ID:              p.ID,
		Name:            p.Name,
		PkgPath:         p.PkgPath,
		Errors:          p.Errors,
		GoFiles:         p.GoFiles,
		CompiledGoFiles: p.CompiledGoFiles,
		OtherFiles:      p.OtherFiles,
		IgnoredFiles:    p.IgnoredFiles,
		ExportFile:      p.ExportFile,
	}
	if len(p.Imports) > 0 {
		flat.Imports = make(map[string]string, len(p.Imports))
		for path, ipkg := range p.Imports {
			flat.Imports[path] = ipkg.ID
		}
	}
	return json.Marshal(flat)
}

// UnmarshalJSON reads in a Package from its JSON format.
// See MarshalJSON for details about the format accepted.
func (p *Package) UnmarshalJSON(b []byte) error {
	flat := &flatPackage{}
	if err := json.Unmarshal(b, &flat); err != nil {
		return err
	}
	*p = Package{
		ID:              flat.ID,
		Name:            flat.Name,
		PkgPath:         flat.PkgPath,
		Errors:          flat.Errors,
		GoFiles:         flat.GoFiles,
		CompiledGoFiles: flat.CompiledGoFiles,
		OtherFiles:      flat.OtherFiles,
		ExportFile:      flat.ExportFile,
	}
	if len(flat.Imports) > 0 {
		p.Imports = make(map[string]*Package, len(flat.Imports))
		for path, id := range flat.Imports {
			p.Imports[path] = &Package{ID: id}
		}
	}
	return nil
}

func (p *Package) String() string { return p.ID }

// loaderPackage augments Package with state used during the loading phase
type loaderPackage struct {
	*Package
	importErrors map[string]error // maps each bad import to its error
	loadOnce     sync.Once
	color        uint8 // for cycle detection
	needsrc      bool  // load from source (Mode >= LoadTypes)
	needtypes    bool  // type information is either requested or depended on
	initial      bool  // package was matched by a pattern
}

// loader holds the working state of a single call to load.
type loader struct {
	pkgs map[string]*loaderPackage
	Config
	sizes        types.Sizes
	parseCache   map[string]*parseValue
	parseCacheMu sync.Mutex
	exportMu     sync.Mutex // enforces mutual exclusion of exportdata operations

	// Config.Mode contains the implied mode (see impliedLoadMode).
	// Implied mode contains all the fields we need the data for.
	// In requestedMode there are the actually requested fields.
	// We'll zero them out before returning packages to the user.
	// This makes it easier for us to get the conditions where
	// we need certain modes right.
	requestedMode LoadMode
}

type parseValue struct {
	f     *ast.File
	err   error
	ready chan struct{}
}

func newLoader(cfg *Config) *loader {
	ld := &loader{
		parseCache: map[string]*parseValue{},
	}
	if cfg != nil {
		ld.Config = *cfg
		// If the user has provided a logger, use it.
		ld.Config.Logf = cfg.Logf
	}
	if ld.Config.Logf == nil {
		// If the GOPACKAGESDEBUG environment variable is set to true,
		// but the user has not provided a logger, default to log.Printf.
		if debug {
			ld.Config.Logf = log.Printf
		} else {
			ld.Config.Logf = func(format string, args ...interface{}) {}
		}
	}
	if ld.Config.Mode == 0 {
		ld.Config.Mode = NeedName | NeedFiles | NeedCompiledGoFiles // Preserve zero behavior of Mode for backwards compatibility.
	}
	if ld.Config.Env == nil {
		ld.Config.Env = os.Environ()
	}
	if ld.Config.gocmdRunner == nil {
		ld.Config.gocmdRunner = &gocommand.Runner{}
	}
	if ld.Context == nil {
		ld.Context = context.Background()
	}
	if ld.Dir == "" {
		if dir, err := os.Getwd(); err == nil {
			ld.Dir = dir
		}
	}

	// Save the actually requested fields. We'll zero them out before returning packages to the user.
	ld.requestedMode = ld.Mode
	ld.Mode = impliedLoadMode(ld.Mode)

	if ld.Mode&NeedTypes != 0 || ld.Mode&NeedSyntax != 0 {
		if ld.Fset == nil {
			ld.Fset = token.NewFileSet()
		}

		// ParseFile is required even in LoadTypes mode
		// because we load source if export data is missing.
		if ld.ParseFile == nil {
			ld.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
				const mode = parser.AllErrors | parser.ParseComments
				return parser.ParseFile(fset, filename, src, mode)
			}
		}
	}

	return ld
}

// refine connects the supplied packages into a graph and then adds type and
// and syntax information as requested by the LoadMode.
func (ld *loader) refine(roots []string, list ...*Package) ([]*Package, error) {
	rootMap := make(map[string]int, len(roots))
	for i, root := range roots {
		rootMap[root] = i
	}
	ld.pkgs = make(map[string]*loaderPackage)
	// first pass, fixup and build the map and roots
	var initial = make([]*loaderPackage, len(roots))
	for _, pkg := range list {
		rootIndex := -1
		if i, found := rootMap[pkg.ID]; found {
			rootIndex = i
		}

		// Overlays can invalidate export data.
		// TODO(matloob): make this check fine-grained based on dependencies on overlaid files
		exportDataInvalid := len(ld.Overlay) > 0 || pkg.ExportFile == "" && pkg.PkgPath != "unsafe"
		// This package needs type information if the caller requested types and the package is
		// either a root, or it's a non-root and the user requested dependencies ...
		needtypes := (ld.Mode&NeedTypes|NeedTypesInfo != 0 && (rootIndex >= 0 || ld.Mode&NeedDeps != 0))
		// This package needs source if the call requested source (or types info, which implies source)
		// and the package is either a root, or itas a non- root and the user requested dependencies...
		needsrc := ((ld.Mode&(NeedSyntax|NeedTypesInfo) != 0 && (rootIndex >= 0 || ld.Mode&NeedDeps != 0)) ||
			// ... or if we need types and the exportData is invalid. We fall back to (incompletely)
			// typechecking packages from source if they fail to compile.
			(ld.Mode&NeedTypes|NeedTypesInfo != 0 && exportDataInvalid)) && pkg.PkgPath != "unsafe"
		lpkg := &loaderPackage{
			Package:   pkg,
			needtypes: needtypes,
			needsrc:   needsrc,
		}
		ld.pkgs[lpkg.ID] = lpkg
		if rootIndex >= 0 {
			initial[rootIndex] = lpkg
			lpkg.initial = true
		}
	}
	for i, root := range roots {
		if initial[i] == nil {
			return nil, fmt.Errorf("root package %v is missing", root)
		}
	}

	// Materialize the import graph.

	const (
		white = 0 // new
		grey  = 1 // in progress
		black = 2 // complete
	)

	// visit traverses the import graph, depth-first,
	// and materializes the graph as Packages.Imports.
	//
	// Valid imports are saved in the Packages.Import map.
	// Invalid imports (cycles and missing nodes) are saved in the importErrors map.
	// Thus, even in the presence of both kinds of errors, the Import graph remains a DAG.
	//
	// visit returns whether the package needs src or has a transitive
	// dependency on a package that does. These are the only packages
	// for which we load source code.
	var stack []*loaderPackage
	var visit func(lpkg *loaderPackage) bool
	var srcPkgs []*loaderPackage
	visit = func(lpkg *loaderPackage) bool {
		switch lpkg.color {
		case black:
			return lpkg.needsrc
		case grey:
			panic("internal error: grey node")
		}
		lpkg.color = grey
		stack = append(stack, lpkg) // push
		stubs := lpkg.Imports       // the structure form has only stubs with the ID in the Imports
		// If NeedImports isn't set, the imports fields will all be zeroed out.
		if ld.Mode&NeedImports != 0 {
			lpkg.Imports = make(map[string]*Package, len(stubs))
			for importPath, ipkg := range stubs {
				var importErr error
				imp := ld.pkgs[ipkg.ID]
				if imp == nil {
					// (includes package "C" when DisableCgo)
					importErr = fmt.Errorf("missing package: %q", ipkg.ID)
				} else if imp.color == grey {
					importErr = fmt.Errorf("import cycle: %s", stack)
				}
				if importErr != nil {
					if lpkg.importErrors == nil {
						lpkg.importErrors = make(map[string]error)
					}
					lpkg.importErrors[importPath] = importErr
					continue
				}

				if visit(imp) {
					lpkg.needsrc = true
				}
				lpkg.Imports[importPath] = imp.Package
			}
		}
		if lpkg.needsrc {
			srcPkgs = append(srcPkgs, lpkg)
		}
		if ld.Mode&NeedTypesSizes != 0 {
			lpkg.TypesSizes = ld.sizes
		}
		stack = stack[:len(stack)-1] // pop
		lpkg.color = black

		return lpkg.needsrc
	}

	if ld.Mode&NeedImports == 0 {
		// We do this to drop the stub import packages that we are not even going to try to resolve.
		for _, lpkg := range initial {
			lpkg.Imports = nil
		}
	} else {
		// For each initial package, create its import DAG.
		for _, lpkg := range initial {
			visit(lpkg)
		}
	}
	if ld.Mode&NeedImports != 0 && ld.Mode&NeedTypes != 0 {
		for _, lpkg := range srcPkgs {
			// Complete type information is required for the
			// immediate dependencies of each source package.
			for _, ipkg := range lpkg.Imports {
				imp := ld.pkgs[ipkg.ID]
				imp.needtypes = true
			}
		}
	}
	// Load type data and syntax if needed, starting at
	// the initial packages (roots of the import DAG).
	if ld.Mode&NeedTypes != 0 || ld.Mode&NeedSyntax != 0 {
		var wg sync.WaitGroup
		for _, lpkg := range initial {
			wg.Add(1)
			go func(lpkg *loaderPackage) {
				ld.loadRecursive(lpkg)
				wg.Done()
			}(lpkg)
		}
		wg.Wait()
	}

	result := make([]*Package, len(initial))
	for i, lpkg := range initial {
		result[i] = lpkg.Package
	}
	for i := range ld.pkgs {
		// Clear all unrequested fields,
		// to catch programs that use more than they request.
		if ld.requestedMode&NeedName == 0 {
			ld.pkgs[i].Name = ""
			ld.pkgs[i].PkgPath = ""
		}
		if ld.requestedMode&NeedFiles == 0 {
			ld.pkgs[i].GoFiles = nil
			ld.pkgs[i].OtherFiles = nil
			ld.pkgs[i].IgnoredFiles = nil
		}
		if ld.requestedMode&NeedCompiledGoFiles == 0 {
			ld.pkgs[i].CompiledGoFiles = nil
		}
		if ld.requestedMode&NeedImports == 0 {
			ld.pkgs[i].Imports = nil
		}
		if ld.requestedMode&NeedExportsFile == 0 {
			ld.pkgs[i].ExportFile = ""
		}
		if ld.requestedMode&NeedTypes == 0 {
			ld.pkgs[i].Types = nil
			ld.pkgs[i].Fset = nil
			ld.pkgs[i].IllTyped = false
		}
		if ld.requestedMode&NeedSyntax == 0 {
			ld.pkgs[i].Syntax = nil
		}
		if ld.requestedMode&NeedTypesInfo == 0 {
			ld.pkgs[i].TypesInfo = nil
		}
		if ld.requestedMode&NeedTypesSizes == 0 {
			ld.pkgs[i].TypesSizes = nil
		}
		if ld.requestedMode&NeedModule == 0 {
			ld.pkgs[i].Module = nil
		}
	}

	return result, nil
}

// loadRecursive loads the specified package and its dependencies,
// recursively, in parallel, in topological order.
// It is atomic and idempotent.
// Precondition: ld.Mode&NeedTypes.
func (ld *loader) loadRecursive(lpkg *loaderPackage) {
	lpkg.loadOnce.Do(func() {
		// Load the direct dependencies, in parallel.
		var wg sync.WaitGroup
		for _, ipkg := range lpkg.Imports {
			imp := ld.pkgs[ipkg.ID]
			wg.Add(1)
			go func(imp *loaderPackage) {
				ld.loadRecursive(imp)
				wg.Done()
			}(imp)
		}
		wg.Wait()
		ld.loadPackage(lpkg)
	})
}

// loadPackage loads the specified package.
// It must be called only once per Package,
// after immediate dependencies are loaded.
// Precondition: ld.Mode & NeedTypes.
func (ld *loader) loadPackage(lpkg *loaderPackage) {
	if lpkg.PkgPath == "unsafe" {
		// Fill in the blanks to avoid surprises.
		lpkg.Types = types.Unsafe
		lpkg.Fset = ld.Fset
		lpkg.Syntax = []*ast.File{}
		lpkg.TypesInfo = new(types.Info)
		lpkg.TypesSizes = ld.sizes
		return
	}

	// Call NewPackage directly with explicit name.
	// This avoids skew between golist and go/types when the files'
	// package declarations are inconsistent.
	lpkg.Types = types.NewPackage(lpkg.PkgPath, lpkg.Name)
	lpkg.Fset = ld.Fset

	// Subtle: we populate all Types fields with an empty Package
	// before loading export data so that export data processing
	// never has to create a types.Package for an indirect dependency,
	// which would then require that such created packages be explicitly
	// inserted back into the Import graph as a final step after export data loading.
	// The Diamond test exercises this case.
	if !lpkg.needtypes && !lpkg.needsrc {
		return
	}
	if !lpkg.needsrc {
		ld.loadFromExportData(lpkg)
		return // not a source package, don't get syntax trees
	}

	appendError := func(err error) {
		// Convert various error types into the one true Error.
		var errs []Error
		switch err := err.(type) {
		case Error:
			// from driver
			errs = append(errs, err)

		case *os.PathError:
			// from parser
			errs = append(errs, Error{
				Pos:  err.Path + ":1",
				Msg:  err.Err.Error(),
				Kind: ParseError,
			})

		case scanner.ErrorList:
			// from parser
			for _, err := range err {
				errs = append(errs, Error{
					Pos:  err.Pos.String(),
					Msg:  err.Msg,
					Kind: ParseError,
				})
			}

		case types.Error:
			// from type checker
			errs = append(errs, Error{
				Pos:  err.Fset.Position(err.Pos).String(),
				Msg:  err.Msg,
				Kind: TypeError,
			})

		default:
			// unexpected impoverished error from parser?
			errs = append(errs, Error{
				Pos:  "-",
				Msg:  err.Error(),
				Kind: UnknownError,
			})

			// If you see this error message, please file a bug.
			log.Printf("internal error: error %q (%T) without position", err, err)
		}

		lpkg.Errors = append(lpkg.Errors, errs...)
	}

	if ld.Config.Mode&NeedTypes != 0 && len(lpkg.CompiledGoFiles) == 0 && lpkg.ExportFile != "" {
		// The config requested loading sources and types, but sources are missing.
		// Add an error to the package and fall back to loading from export data.
		appendError(Error{"-", fmt.Sprintf("sources missing for package %s", lpkg.ID), ParseError})
		ld.loadFromExportData(lpkg)
		return // can't get syntax trees for this package
	}

	files, errs := ld.parseFiles(lpkg.CompiledGoFiles)
	for _, err := range errs {
		appendError(err)
	}

	lpkg.Syntax = files
	if ld.Config.Mode&NeedTypes == 0 {
		return
	}

	lpkg.TypesInfo = &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	lpkg.TypesSizes = ld.sizes

	importer := importerFunc(func(path string) (*types.Package, error) {
		if path == "unsafe" {
			return types.Unsafe, nil
		}

		// The imports map is keyed by import path.
		ipkg := lpkg.Imports[path]
		if ipkg == nil {
			if err := lpkg.importErrors[path]; err != nil {
				return nil, err
			}
			// There was skew between the metadata and the
			// import declarations, likely due to an edit
			// race, or because the ParseFile feature was
			// used to supply alternative file contents.
			return nil, fmt.Errorf("no metadata for %s", path)
		}

		if ipkg.Types != nil && ipkg.Types.Complete() {
			return ipkg.Types, nil
		}
		log.Fatalf("internal error: package %q without types was imported from %q", path, lpkg)
		panic("unreachable")
	})

	// type-check
	tc := &types.Config{
		Importer: importer,

		// Type-check bodies of functions only in non-initial packages.
		// Example: for import graph A->B->C and initial packages {A,C},
		// we can ignore function bodies in B.
		IgnoreFuncBodies: ld.Mode&NeedDeps == 0 && !lpkg.initial,

		Error: appendError,
		Sizes: ld.sizes,
	}
	if (ld.Mode & typecheckCgo) != 0 {
		if !typesinternal.SetUsesCgo(tc) {
			appendError(Error{
				Msg:  "typecheckCgo requires Go 1.15+",
				Kind: ListError,
			})
			return
		}
	}
	types.NewChecker(tc, ld.Fset, lpkg.Types, lpkg.TypesInfo).Files(lpkg.Syntax)

	lpkg.importErrors = nil // no longer needed

	// If !Cgo, the type-checker uses FakeImportC mode, so
	// it doesn't invoke the importer for import "C",
	// nor report an error for the import,
	// or for any undefined C.f reference.
	// We must detect this explicitly and correctly
	// mark the package as IllTyped (by reporting an error).
	// TODO(adonovan): if these errors are annoying,
	// we could just set IllTyped quietly.
	if tc.FakeImportC {
	outer:
		for _, f := range lpkg.Syntax {
			for _, imp := range f.Imports {
				if imp.Path.Value == `"C"` {
					err := types.Error{Fset: ld.Fset, Pos: imp.Pos(), Msg: `import "C" ignored`}
					appendError(err)
					break outer
				}
			}
		}
	}

	// Record accumulated errors.
	illTyped := len(lpkg.Errors) > 0
	if !illTyped {
		for _, imp := range lpkg.Imports {
			if imp.IllTyped {
				illTyped = true
				break
			}
		}
	}
	lpkg.IllTyped = illTyped
}

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

// We use a counting semaphore to limit
// the number of parallel I/O calls per process.
var ioLimit = make(chan bool, 20)

func (ld *loader) parseFile(filename string) (*ast.File, error) {
	ld.parseCacheMu.Lock()
	v, ok := ld.parseCache[filename]
	if ok {
		// cache hit
		ld.parseCacheMu.Unlock()
		<-v.ready
	} else {
		// cache miss
		v = &parseValue{ready: make(chan struct{})}
		ld.parseCache[filename] = v
		ld.parseCacheMu.Unlock()

		var src []byte
		for f, contents := range ld.Config.Overlay {
			if sameFile(f, filename) {
				src = contents
			}
		}
		var err error
		if src == nil {
			ioLimit <- true // wait
			src, err = ioutil.ReadFile(filename)
			<-ioLimit // signal
		}
		if err != nil {
			v.err = err
		} else {
			v.f, v.err = ld.ParseFile(ld.Fset, filename, src)
		}

		close(v.ready)
	}
	return v.f, v.err
}

// parseFiles reads and parses the Go source files and returns the ASTs
// of the ones that could be at least partially parsed, along with a
// list of I/O and parse errors encountered.
//
// Because files are scanned in parallel, the token.Pos
// positions of the resulting ast.Files are not ordered.
//
func (ld *loader) parseFiles(filenames []string) ([]*ast.File, []error) {
	var wg sync.WaitGroup
	n := len(filenames)
	parsed := make([]*ast.File, n)
	errors := make([]error, n)
	for i, file := range filenames {
		if ld.Config.Context.Err() != nil {
			parsed[i] = nil
			errors[i] = ld.Config.Context.Err()
			continue
		}
		wg.Add(1)
		go func(i int, filename string) {
			parsed[i], errors[i] = ld.parseFile(filename)
			wg.Done()
		}(i, file)
	}
	wg.Wait()

	// Eliminate nils, preserving order.
	var o int
	for _, f := range parsed {
		if f != nil {
			parsed[o] = f
			o++
		}
	}
	parsed = parsed[:o]

	o = 0
	for _, err := range errors {
		if err != nil {
			errors[o] = err
			o++
		}
	}
	errors = errors[:o]

	return parsed, errors
}

// sameFile returns true if x and y have the same basename and denote
// the same file.
//
func sameFile(x, y string) bool {
	if x == y {
		// It could be the case that y doesn't exist.
		// For instance, it may be an overlay file that
		// hasn't been written to disk. To handle that case
		// let x == y through. (We added the exact absolute path
		// string to the CompiledGoFiles list, so the unwritten
		// overlay case implies x==y.)
		return true
	}
	if strings.EqualFold(filepath.Base(x), filepath.Base(y)) { // (optimisation)
		if xi, err := os.Stat(x); err == nil {
			if yi, err := os.Stat(y); err == nil {
				return os.SameFile(xi, yi)
			}
		}
	}
	return false
}

// loadFromExportData returns type information for the specified
// package, loading it from an export data file on the first request.
func (ld *loader) loadFromExportData(lpkg *loaderPackage) (*types.Package, error) {
	if lpkg.PkgPath == "" {
		log.Fatalf("internal error: Package %s has no PkgPath", lpkg)
	}

	// Because gcexportdata.Read has the potential to create or
	// modify the types.Package for each node in the transitive
	// closure of dependencies of lpkg, all exportdata operations
	// must be sequential. (Finer-grained locking would require
	// changes to the gcexportdata API.)
	//
	// The exportMu lock guards the Package.Pkg field and the
	// types.Package it points to, for each Package in the graph.
	//
	// Not all accesses to Package.Pkg need to be protected by exportMu:
	// graph ordering ensures that direct dependencies of source
	// packages are fully loaded before the importer reads their Pkg field.
	ld.exportMu.Lock()
	defer ld.exportMu.Unlock()

	if tpkg := lpkg.Types; tpkg != nil && tpkg.Complete() {
		return tpkg, nil // cache hit
	}

	lpkg.IllTyped = true // fail safe

	if lpkg.ExportFile == "" {
		// Errors while building export data will have been printed to stderr.
		return nil, fmt.Errorf("no export data file")
	}
	f, err := os.Open(lpkg.ExportFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read gc export data.
	//
	// We don't currently support gccgo export data because all
	// underlying workspaces use the gc toolchain. (Even build
	// systems that support gccgo don't use it for workspace
	// queries.)
	r, err := gcexportdata.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %v", lpkg.ExportFile, err)
	}

	// Build the view.
	//
	// The gcexportdata machinery has no concept of package ID.
	// It identifies packages by their PkgPath, which although not
	// globally unique is unique within the scope of one invocation
	// of the linker, type-checker, or gcexportdata.
	//
	// So, we must build a PkgPath-keyed view of the global
	// (conceptually ID-keyed) cache of packages and pass it to
	// gcexportdata. The view must contain every existing
	// package that might possibly be mentioned by the
	// current package---its transitive closure.
	//
	// In loadPackage, we unconditionally create a types.Package for
	// each dependency so that export data loading does not
	// create new ones.
	//
	// TODO(adonovan): it would be simpler and more efficient
	// if the export data machinery invoked a callback to
	// get-or-create a package instead of a map.
	//
	view := make(map[string]*types.Package) // view seen by gcexportdata
	seen := make(map[*loaderPackage]bool)   // all visited packages
	var visit func(pkgs map[string]*Package)
	visit = func(pkgs map[string]*Package) {
		for _, p := range pkgs {
			lpkg := ld.pkgs[p.ID]
			if !seen[lpkg] {
				seen[lpkg] = true
				view[lpkg.PkgPath] = lpkg.Types
				visit(lpkg.Imports)
			}
		}
	}
	visit(lpkg.Imports)

	viewLen := len(view) + 1 // adding the self package
	// Parse the export data.
	// (May modify incomplete packages in view but not create new ones.)
	tpkg, err := gcexportdata.Read(r, ld.Fset, view, lpkg.PkgPath)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %v", lpkg.ExportFile, err)
	}
	if viewLen != len(view) {
		log.Fatalf("Unexpected package creation during export data loading")
	}

	lpkg.Types = tpkg
	lpkg.IllTyped = false

	return tpkg, nil
}

// impliedLoadMode returns loadMode with its dependencies.
func impliedLoadMode(loadMode LoadMode) LoadMode {
	if loadMode&NeedTypesInfo != 0 && loadMode&NeedImports == 0 {
		// If NeedTypesInfo, go/packages needs to do typechecking itself so it can
		// associate type info with the AST. To do so, we need the export data
		// for dependencies, which means we need to ask for the direct dependencies.
		// NeedImports is used to ask for the direct dependencies.
		loadMode |= NeedImports
	}

	if loadMode&NeedDeps != 0 && loadMode&NeedImports == 0 {
		// With NeedDeps we need to load at least direct dependencies.
		// NeedImports is used to ask for the direct dependencies.
		loadMode |= NeedImports
	}

	return loadMode
}

func usesExportData(cfg *Config) bool {
	return cfg.Mode&NeedExportsFile != 0 || cfg.Mode&NeedTypes != 0 && cfg.Mode&NeedDeps == 0
}
