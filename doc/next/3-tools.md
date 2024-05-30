## Tools {#tools}

### Go command {#go-command}

### Cgo {#cgo}

Cgo currently refuses to compile calls to a C function which has multiple
incompatible declarations. For instance, if `f` is declared as both `void f(int)`
and `void f(double)`, cgo will report an error instead of possibly generating an
incorrect call sequence for `f(0)`. New in this release is a better detector for
this error condition when the incompatible declarations appear in different
files. See [#67699](https://go.dev/issue/67699).
