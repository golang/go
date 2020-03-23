// Package packagesinternal exposes internal-only fields from go/packages.
package packagesinternal

import (
	"time"

	"golang.org/x/tools/internal/gocommand"
)

// Fields must match go list;
type Module struct {
	Path      string       // module path
	Version   string       // module version
	Versions  []string     // available module versions (with -versions)
	Replace   *Module      // replaced by this module
	Time      *time.Time   // time version was created
	Update    *Module      // available update, if any (with -u)
	Main      bool         // is this the main module?
	Indirect  bool         // is this module only an indirect dependency of main module?
	Dir       string       // directory holding files for this module, if any
	GoMod     string       // path to go.mod file used when loading this module, if any
	GoVersion string       // go version used in module
	Error     *ModuleError // error loading module
}
type ModuleError struct {
	Err string // the error itself
}

var GetForTest = func(p interface{}) string { return "" }

var GetModule = func(p interface{}) *Module { return nil }

var GetGoCmdRunner = func(config interface{}) *gocommand.Runner { return nil }

var SetGoCmdRunner = func(config interface{}, runner *gocommand.Runner) {}
