module indirect

go 1.12
//@diag("// indirect", "go mod tidy", "example.com/extramodule should be a direct dependency.", "warning"),suggestedfix("// indirect")
require example.com/extramodule v1.0.0 // indirect
