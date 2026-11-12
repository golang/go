## Compiler {#compiler}

<!-- go.dev/issue/83032 -->

[Generic type aliases](/ref/spec#Alias_declarations) were introduced in
[Go 1.24](/doc/go1.24#language), but due to an oversight, the compiler and
[go/types] checked for language version Go 1.23 instead of Go 1.24.
Starting with Go 1.28, declaring a generic type alias requires language
version Go 1.24 or later. Packages targeting Go 1.23 (for instance, via
`go 1.23` in `go.mod`) that declare generic type aliases will now be
rejected unless updated to at least `go 1.24`.
See [issue #83032](/issue/83032).

## Assembler {#assembler}

## Linker {#linker}

## Bootstrap {#boostrap}

<!-- go.dev/issue/69315 -->
As mentioned in the [Go 1.26 release notes](/doc/go1.26#bootstrap), Go 1.28 now requires
Go 1.26.0 or later for bootstrap.
We expect that Go 1.30 will require a minor release of Go 1.28.0 or later for bootstrap.
