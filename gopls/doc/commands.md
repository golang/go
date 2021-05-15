# Commands

This document describes the LSP-level commands supported by `gopls`. They cannot be invoked directly by users, and all the details are subject to change, so nobody should rely on this information.

<!-- BEGIN Commands: DO NOT MANUALLY EDIT THIS SECTION -->
### **Add dependency**
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

### **asks the server to add an import path to a given Go file.**
Identifier: `gopls.add_import`

The method will call applyEdit on the client so that clients don't have
to apply the edit themselves.

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

### **go get package**
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

### **retrieves a list of packages**
Identifier: `gopls.list_known_packages`

that are importable from the given URI.

Args:

```
{
	// The file URI.
	"URI": string,
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

### **Remove dependency**
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

### ****
Identifier: `gopls.start_debugging`



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

### **Upgrade dependency**
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

### ****
Identifier: `gopls.workspace_metadata`



<!-- END Commands: DO NOT MANUALLY EDIT THIS SECTION -->
