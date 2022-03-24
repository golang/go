# Commands

This document describes the LSP-level commands supported by `gopls`. They cannot be invoked directly by users, and all the details are subject to change, so nobody should rely on this information.

<!-- BEGIN Commands: DO NOT MANUALLY EDIT THIS SECTION -->
### **Add a dependency**
Identifier: `gopls.add_dependency`

Adds a dependency to the go.mod file for a module.

Args:

```
{
	// The go.mod file URI.
	"URI": string,
	// Additional args to pass to the go command.
	"GoCmdArgs": []string,
	// Whether to add a require directive.
	"AddRequire": bool,
}
```

### **Add an import**
Identifier: `gopls.add_import`

Ask the server to add an import path to a given Go file.  The method will
call applyEdit on the client so that clients don't have to apply the edit
themselves.

Args:

```
{
	// ImportPath is the target import path that should
	// be added to the URI file
	"ImportPath": string,
	// URI is the file that the ImportPath should be
	// added to
	"URI": string,
}
```

### **Apply a fix**
Identifier: `gopls.apply_fix`

Applies a fix to a region of source code.

Args:

```
{
	// The fix to apply.
	"Fix": string,
	// The file URI for the document to fix.
	"URI": string,
	// The document range to scan for fixes.
	"Range": {
		"start": {
			"line": uint32,
			"character": uint32,
		},
		"end": {
			"line": uint32,
			"character": uint32,
		},
	},
}
```

### **Check for upgrades**
Identifier: `gopls.check_upgrades`

Checks for module upgrades.

Args:

```
{
	// The go.mod file URI.
	"URI": string,
	// The modules to check.
	"Modules": []string,
}
```

### **Run go mod edit -go=version**
Identifier: `gopls.edit_go_directive`

Runs `go mod edit -go=version` for a module.

Args:

```
{
	// Any document URI within the relevant module.
	"URI": string,
	// The version to pass to `go mod edit -go`.
	"Version": string,
}
```

### **Toggle gc_details**
Identifier: `gopls.gc_details`

Toggle the calculation of gc annotations.

Args:

```
string
```

### **Run go generate**
Identifier: `gopls.generate`

Runs `go generate` for a given directory.

Args:

```
{
	// URI for the directory to generate.
	"Dir": string,
	// Whether to generate recursively (go generate ./...)
	"Recursive": bool,
}
```

### **Generate gopls.mod**
Identifier: `gopls.generate_gopls_mod`

(Re)generate the gopls.mod file for a workspace.

Args:

```
{
	// The file URI.
	"URI": string,
}
```

### **go get a package**
Identifier: `gopls.go_get_package`

Runs `go get` to fetch a package.

Args:

```
{
	// Any document URI within the relevant module.
	"URI": string,
	// The package to go get.
	"Pkg": string,
	"AddRequire": bool,
}
```

### **List imports of a file and its package**
Identifier: `gopls.list_imports`

Retrieve a list of imports in the given Go file, and the package it
belongs to.

Args:

```
{
	// The file URI.
	"URI": string,
}
```

Result:

```
{
	// Imports is a list of imports in the requested file.
	"Imports": []{
		"Path": string,
		"Name": string,
	},
	// PackageImports is a list of all imports in the requested file's package.
	"PackageImports": []{
		"Path": string,
	},
}
```

### **List known packages**
Identifier: `gopls.list_known_packages`

Retrieve a list of packages that are importable from the given URI.

Args:

```
{
	// The file URI.
	"URI": string,
}
```

Result:

```
{
	// Packages is a list of packages relative
	// to the URIArg passed by the command request.
	// In other words, it omits paths that are already
	// imported or cannot be imported due to compiler
	// restrictions.
	"Packages": []string,
}
```

### **Regenerate cgo**
Identifier: `gopls.regenerate_cgo`

Regenerates cgo definitions.

Args:

```
{
	// The file URI.
	"URI": string,
}
```

### **Remove a dependency**
Identifier: `gopls.remove_dependency`

Removes a dependency from the go.mod file of a module.

Args:

```
{
	// The go.mod file URI.
	"URI": string,
	// The module path to remove.
	"ModulePath": string,
	"OnlyDiagnostic": bool,
}
```

### **Run test(s)**
Identifier: `gopls.run_tests`

Runs `go test` for a specific set of test or benchmark functions.

Args:

```
{
	// The test file containing the tests to run.
	"URI": string,
	// Specific test names to run, e.g. TestFoo.
	"Tests": []string,
	// Specific benchmarks to run, e.g. BenchmarkFoo.
	"Benchmarks": []string,
}
```

### **Run vulncheck (experimental)**
Identifier: `gopls.run_vulncheck_exp`

Run vulnerability check (`govulncheck`).

Args:

```
{
	// Dir is the directory from which vulncheck will run from.
	"Dir": string,
	// Package pattern. E.g. "", ".", "./...".
	"Pattern": string,
}
```

Result:

```
{
	"Vuln": []{
		"ID": string,
		"Details": string,
		"Aliases": []string,
		"Symbol": string,
		"PkgPath": string,
		"ModPath": string,
		"URL": string,
		"CurrentVersion": string,
		"FixedVersion": string,
		"CallStacks": [][]golang.org/x/tools/internal/lsp/command.StackEntry,
	},
}
```

### **Start the gopls debug server**
Identifier: `gopls.start_debugging`

Start the gopls debug server if it isn't running, and return the debug
address.

Args:

```
{
	// Optional: the address (including port) for the debug server to listen on.
	// If not provided, the debug server will bind to "localhost:0", and the
	// full debug URL will be contained in the result.
	// 
	// If there is more than one gopls instance along the serving path (i.e. you
	// are using a daemon), each gopls instance will attempt to start debugging.
	// If Addr specifies a port, only the daemon will be able to bind to that
	// port, and each intermediate gopls instance will fail to start debugging.
	// For this reason it is recommended not to specify a port (or equivalently,
	// to specify ":0").
	// 
	// If the server was already debugging this field has no effect, and the
	// result will contain the previously configured debug URL(s).
	"Addr": string,
}
```

Result:

```
{
	// The URLs to use to access the debug servers, for all gopls instances in
	// the serving path. For the common case of a single gopls instance (i.e. no
	// daemon), this will be exactly one address.
	// 
	// In the case of one or more gopls instances forwarding the LSP to a daemon,
	// URLs will contain debug addresses for each server in the serving path, in
	// serving order. The daemon debug address will be the last entry in the
	// slice. If any intermediate gopls instance fails to start debugging, no
	// error will be returned but the debug URL for that server in the URLs slice
	// will be empty.
	"URLs": []string,
}
```

### **Run test(s) (legacy)**
Identifier: `gopls.test`

Runs `go test` for a specific set of test or benchmark functions.

Args:

```
string,
[]string,
[]string
```

### **Run go mod tidy**
Identifier: `gopls.tidy`

Runs `go mod tidy` for a module.

Args:

```
{
	// The file URIs.
	"URIs": []string,
}
```

### **Toggle gc_details**
Identifier: `gopls.toggle_gc_details`

Toggle the calculation of gc annotations.

Args:

```
{
	// The file URI.
	"URI": string,
}
```

### **Update go.sum**
Identifier: `gopls.update_go_sum`

Updates the go.sum file for a module.

Args:

```
{
	// The file URIs.
	"URIs": []string,
}
```

### **Upgrade a dependency**
Identifier: `gopls.upgrade_dependency`

Upgrades a dependency in the go.mod file for a module.

Args:

```
{
	// The go.mod file URI.
	"URI": string,
	// Additional args to pass to the go command.
	"GoCmdArgs": []string,
	// Whether to add a require directive.
	"AddRequire": bool,
}
```

### **Run go mod vendor**
Identifier: `gopls.vendor`

Runs `go mod vendor` for a module.

Args:

```
{
	// The file URI.
	"URI": string,
}
```

<!-- END Commands: DO NOT MANUALLY EDIT THIS SECTION -->
