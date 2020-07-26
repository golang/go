module unused

go 1.12

require example.com/extramodule v1.0.0 //@diag("require example.com/extramodule v1.0.0", "go mod tidy", "example.com/extramodule is not used in this module", "warning"),suggestedfix("require example.com/extramodule v1.0.0", "quickfix")
