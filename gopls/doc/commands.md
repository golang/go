# Commands

This document describes the LSP-level commands supported by `gopls`. They cannot be invoked directly by users, and all the details are subject to change, so nobody should rely on this information.

<!-- BEGIN Commands: DO NOT MANUALLY EDIT THIS SECTION -->
### **Add dependency**
Identifier: `gopls.add_dependency`

Adds a dependency to the go.mod file for a module.

### **Apply a fix**
Identifier: `gopls.apply_fix`

Applies a fix to a region of source code.

### **Check for upgrades**
Identifier: `gopls.check_upgrades`

Checks for module upgrades.

### **Toggle gc_details**
Identifier: `gopls.gc_details`

Toggle the calculation of gc annotations.

### **Run go generate**
Identifier: `gopls.generate`

Runs `go generate` for a given directory.

### **Generate gopls.mod**
Identifier: `gopls.generate_gopls_mod`

(Re)generate the gopls.mod file for a workspace.

### **go get package**
Identifier: `gopls.go_get_package`

Runs `go get` to fetch a package.

### **Regenerate cgo**
Identifier: `gopls.regenerate_cgo`

Regenerates cgo definitions.

### **Remove dependency**
Identifier: `gopls.remove_dependency`

Removes a dependency from the go.mod file of a module.

### **Run test(s)**
Identifier: `gopls.run_tests`

Runs `go test` for a specific set of test or benchmark functions.

### **Run test(s) (legacy)**
Identifier: `gopls.test`

Runs `go test` for a specific set of test or benchmark functions.

### **Run go mod tidy**
Identifier: `gopls.tidy`

Runs `go mod tidy` for a module.

### **Toggle gc_details**
Identifier: `gopls.toggle_gc_details`

Toggle the calculation of gc annotations.

### **Update go.sum**
Identifier: `gopls.update_go_sum`

Updates the go.sum file for a module.

### **Upgrade dependency**
Identifier: `gopls.upgrade_dependency`

Upgrades a dependency in the go.mod file for a module.

### **Run go mod vendor**
Identifier: `gopls.vendor`

Runs `go mod vendor` for a module.

<!-- END Commands: DO NOT MANUALLY EDIT THIS SECTION -->
