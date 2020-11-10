# Commands

This document describes the LSP-level commands supported by `gopls`. They cannot be invoked directly by users, and all the details are subject to change, so nobody should rely on this information.

<!-- BEGIN Commands: DO NOT MANUALLY EDIT THIS SECTION -->
### **Run go generate**
Identifier: `gopls.generate`

generate runs `go generate` for a given directory.


### **Fill struct**
Identifier: `gopls.fill_struct`

fill_struct is a gopls command to fill a struct with default
values.


### **Regenerate cgo**
Identifier: `gopls.regenerate_cgo`

regenerate_cgo regenerates cgo definitions.


### **Run test(s)**
Identifier: `gopls.test`

test runs `go test` for a specific test function.


### **Run go mod tidy**
Identifier: `gopls.tidy`

tidy runs `go mod tidy` for a module.


### **Update go.sum**
Identifier: `gopls.update_go_sum`

update_go_sum updates the go.sum file for a module.


### **Undeclared name**
Identifier: `gopls.undeclared_name`

undeclared_name adds a variable declaration for an undeclared
name.


### **go get package**
Identifier: `gopls.go_get_package`

go_get_package runs `go get` to fetch a package.


### **Add dependency**
Identifier: `gopls.add_dependency`

add_dependency adds a dependency.


### **Upgrade dependency**
Identifier: `gopls.upgrade_dependency`

upgrade_dependency upgrades a dependency.


### **Remove dependency**
Identifier: `gopls.remove_dependency`

remove_dependency removes a dependency.


### **Run go mod vendor**
Identifier: `gopls.vendor`

vendor runs `go mod vendor` for a module.


### **Extract to variable**
Identifier: `gopls.extract_variable`

extract_variable extracts an expression to a variable.


### **Extract to function**
Identifier: `gopls.extract_function`

extract_function extracts statements to a function.


### **Toggle gc_details**
Identifier: `gopls.gc_details`

gc_details controls calculation of gc annotations.


### **Generate gopls.mod**
Identifier: `gopls.generate_gopls_mod`

generate_gopls_mod (re)generates the gopls.mod file.


<!-- END Commands: DO NOT MANUALLY EDIT THIS SECTION -->
