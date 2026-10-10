# Shared-tool launcher

`launcher.c` is the native process boundary for a Go command built as a
`c-shared` library. It locates the matching library beside the launcher,
resolves `dll_<tool>_main`, forwards the process arguments and standard
streams, then returns the entry point's status as the process exit code.

The executable and library use the same basename:

| Platform | Executable | Library |
| --- | --- | --- |
| Windows | `go.exe` | `go.dll` |
| Linux | `compile` | `compile.so` |
| macOS | `asm` | `asm.dylib` |

The exported function uses the C ABI `int entry(int argc, char **argv)`.
Windows arguments are converted from UTF-16 to UTF-8; POSIX arguments are
passed through unchanged. The launcher does not redirect or buffer the standard
streams.

The Go runtime in a `c-shared` library is process-wide. The launcher therefore
keeps the library loaded until process exit rather than attempting to unload
the Go runtime.

Build the Linux launcher with:

```sh
cc -O2 -o go launcher.c -ldl
```

The tool libraries must export the symbol matching the executable basename.
The Go-side entry point and build integration are added separately.
