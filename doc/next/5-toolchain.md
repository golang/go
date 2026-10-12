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

<!-- go.dev/issue/62577 -->
<!-- go.dev/issue/82729 -->
On macOS and iOS, the linker now supports emitting DWARF debug information
into a separate file in `<out>.dSYM/Contents/Resources/DWARF/<out>`, instead
of combining it into the output binary.
By default, split DWARF is enabled for `-buildmode=c-shared` and
`-buildmode=plugin`, and disabled for executables.
(On macOS, `-buildmode=c-shared` previously defaulted to `-w` to omit DWARF;
it now emits split DWARF by default.)
When split DWARF is enabled, `go build` and `go test -c` output the `.dSYM`
directory alongside the built binary, whereas `go install` does not install
the `.dSYM` directory.
`go clean` removes it along with the binary.

The `.dSYM` directory follows the platform convention used by the C toolchain
and the `dsymutil` tool.
LLDB understands this convention and will load the debug information automatically
from the split DWARF file.
(Note: LLDB does not understand compressed DWARF, so the binary needs to be
built with the `-ldflags=-compressdwarf=0` build option.)
<!-- TODO: mention Delve once Delve supports it -->
The [`debug/macho.File.DWARF`](/pkg/debug/macho#File.DWARF) method is also
updated to load the debug information from the split DWARF file if available.

A linker command line option `-splitdwarf` can be used to enable or disable
split DWARF.
This option may be removed in a future Go release.

## Bootstrap {#boostrap}

<!-- go.dev/issue/69315 -->
As mentioned in the [Go 1.26 release notes](/doc/go1.26#bootstrap), Go 1.28 now requires
Go 1.26.0 or later for bootstrap.
We expect that Go 1.30 will require a minor release of Go 1.28.0 or later for bootstrap.
