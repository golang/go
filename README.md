# loGo/s

Go whose errors the compiler checks, with an exit to plain Go.

loGos is one toolchain for Go and loGo/s. loGo/s adds `?` error propagation and `errdefer` to Go. You write `.lgo` files. The `logos` command checks them, lowers them to plain Go, and compiles them with the Go 1.27 backend. Plain `go` keeps Go semantics and its upstream command text (see [README.go.md](README.go.md)). Go packages and loGo/s packages import each other, per import. `logos eject` turns a loGo/s module back into plain Go with its comments.

* **Plain Go underneath** (`TestEjectMatchesLowering` checks the lowered output against what `logos eject` writes).
* **Go keeps its meaning** (`run.bash` passes with plain `go`).
* **Go and loGo/s import each other**, one kind per module (`logos_interop.txt` covers the per-import rule in both directions).

Contents:

* [What is this](#what-is-this)
* [Install](#install)
* [Quick start](#quick-start)
* [Language](#language)
* [Agent skills](#agent-skills)
* [Development](#development)
* [License](#license)

## What is this

```go
package main

import (
	"fmt"
	"os"
	"strconv"
)

func double(s string) (int, error) {
	n := strconv.Atoi(s)?
	return 2 * n, nil
}

func main() { fmt.Println(double(os.Args[1])) } // logos run . 21 -> 42 <nil>
```

Run it with `logos run . 21`. It prints `42 <nil>`. The full language is defined in [logos-docs/SPEC.md](logos-docs/SPEC.md).

## Install

You need Go 1.24.6 or newer to bootstrap. [src/make.bash](src/make.bash) sets this requirement (`bootgo=1.24.6`).

1. Clone the tree:

   ```sh
   git clone https://github.com/gosuda/loGos
   ```

2. Build the toolchain:

   ```sh
   cd loGos/src
   ```

   ```sh
   ./make.bash
   ```

3. Put `logos` on your PATH:

   ```sh
   export PATH="$PWD/../bin:$PATH"
   ```

4. Check the toolchain:

   ```sh
   logos version
   ```

## Quick start

```sh
mkdir hello && cd hello
logos mod init example.com/hello
cat > main.lgo <<'EOF'
package main

import (
	"fmt"
	"os"
	"strconv"
)

func double(s string) (int, error) {
	n := strconv.Atoi(s)?
	return 2 * n, nil
}

func main() { fmt.Println(double(os.Args[1])) }
EOF
logos run . 21
```

```sh
logos eject -o ../hello-go
cd ../hello-go && go run . 21
```

Both print `42 <nil>`. The ejected module is plain Go.

## Language

| Form | Meaning |
| ---- | ------- |
| `x := f()?` | Bind the call results except the error, and return the error from the caller when it is non-nil. |
| `f()?` | Discard the value results, and return the error from the caller when it is non-nil. |
| `defer f()?` | At function exit, join a non-nil deferred error into the caller's error result. |
| `errdefer f()` | Run `f` only when the frame exits with a non-nil error, a panic, or Goexit. |
| `errdefer f()?` | Run `f` only on failure, joining its error into the primary error. |

`?` applies to a call whose last result is `error`. The enclosing function must also return `error` last. `errdefer` is allowed only in such functions. A module holds either `.go` or `.lgo` packages, never both. Imports cross the boundary per import.

## Agent skills

The [plugins/logos/](plugins/logos/) plugin ships three skills:

* logos: write, build, test, and eject loGo/s code.
* logos-review: review `.lgo` code against the compiler rules.
* logos-from-go: migrate a Go module to loGo/s by hand.

Claude (clones the whole Go fork):

```sh
claude plugin marketplace add gosuda/loGos
claude plugin install logos@logos
```

Codex:

```sh
codex plugin marketplace add gosuda/loGos --sparse plugins/logos
codex plugin add logos@logos
```

Agy, from a clone of this repository:

```sh
agy plugin install ./plugins/logos
```

Copy [plugins/logos/templates/AGENTS.md](plugins/logos/templates/AGENTS.md) into a loGo/s repository so agents follow the module rules there.

## Development

```sh
cd src && ./make.bash
go test cmd/go -run 'Script/^logos_'
go test cmd/internal/testdir -run 'Test/logos/'
./run.bash --no-rebuild
```

## License

BSD-style, see [LICENSE](LICENSE) (The Go Authors). loGo/s changes are under the same license.
