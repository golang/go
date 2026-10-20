# loGo/s

Go whose errors the compiler checks, with an exit to plain Go, and closed variants.

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

Shipped forms, oldest first. Each row names the fixture that proves it: files under `test/logos/run/` run with `go test cmd/internal/testdir -run 'Test/logos/'`, files under `src/internal/types/testdata/logos/` run with `go test -short cmd/compile/internal/types2 -run TestLogos`, and `logos_*.txt` scripts run with `go test cmd/go -run 'Script/^logos_'`.

| Form | Meaning |
| ---- | ------- |
| `x := f()?` | Bind the call results except the error, and return the error from the caller when it is non-nil. |
| `f()?` | Discard the value results, and return the error from the caller when it is non-nil. |
| `defer f()?` | At function exit, join a non-nil deferred error into the caller's error result. |
| `errdefer f()` | Run `f` only when the frame exits with a non-nil error, a panic, or Goexit. |
| `errdefer f()?` | Run `f` only on failure, joining its error into the primary error. |
| `type U interface { int \| string }` | A closed union: a fixed set of member types. Proven by `test/logos/run/union_basic.lgo` and `src/internal/types/testdata/logos/union_members.lgo`. |
| `r := switch n { case 1 => "one"; default => "many" }` | A switch that returns a value; each arm uses `=>`. Proven by `test/logos/run/switch_expr.lgo` and `src/internal/types/testdata/logos/switch_expr_types.lgo`. |
| `switch x.(type) { case int => 1; case string => 2; case nil => 0 }` | A type switch over a union. Without `default`, it must cover every member and `nil`. Proven by `test/logos/run/union_basic.lgo` and `src/internal/types/testdata/logos/union_exhaustive.lgo`. |
| `x.(U)` | Assert membership at run time. A non-member gives `ok == false` with two results, or panics with `interface conversion: float64 is not a member of main.U`. Proven by `test/logos/run/union_assert.lgo`. |
| `type Person struct : Greeter { ... }` | Assert at the declaration that `*Person` implements `Greeter`. Proven by `test/logos/run/union_basic.lgo` and `src/internal/types/testdata/logos/struct_assert.lgo`. |
| `var x int?` | An option: either a value or none. `nil` and plain values fit; options never convert back. Proven by `test/logos/run/option_basic.lgo` and `src/internal/types/testdata/logos/option_types.lgo`. |
| `u?.Name` | Read through an option. None stays none instead of panicking. Proven by `test/logos/run/option_chain.lgo`. |
| `x ?? d` | `x` when present, `d` when none. Proven by `test/logos/run/option_basic.lgo`. |
| `*User?` | An option over a pointer, stored as the bare pointer with nil as none. Proven by `test/logos/run/option_size.lgo`. |
| `m[k]` | A single-value map read gives an option: none when the key is absent. `v, ok := m[k]` keeps the Go meaning. Proven by `test/logos/run/map_index.lgo` and `src/internal/types/testdata/logos/map_index.lgo`. |
| Narrowing with `!= nil` | After `if x != nil`, or after `if x == nil { return }`, the option acts as its plain value. Proven by `src/internal/types/testdata/logos/narrowing.lgo`. |
| Error-first results | Results of `.lgo` functions are read only after the error check. Calls into other packages keep their own rules. Proven by `src/internal/types/testdata/logos/error_first.lgo`. |
| Non-nil `*T` | A `.lgo` pointer holds a value, never nil. Storing nil, or leaving one without a value, fails to compile. Use `T?` for pointers that may be nil. Proven by `src/internal/types/testdata/logos/nonnil_zero.lgo` and `test/logos/run/platform.lgo`. |
| `type C interface { const Name string; HP int }` | An interface with properties: `const` is read-only, plain is read-write. Proven by `test/logos/run/properties.lgo` and `src/internal/types/testdata/logos/property_iface.lgo`. |
| `type Monster struct : Creature { ... }` | A struct meets a property interface only by listing it after `:`. Reads and writes through the interface reach the struct. Proven by `test/logos/run/properties.lgo` and `src/cmd/go/testdata/script/logos_interop_prop.txt`. |
| `const []int` parameter | A read-only view of a value. Writes through it fail to compile; the view erases to the plain type. Proven by `test/logos/run/const_views.lgo` and `src/internal/types/testdata/logos/const_views.lgo`. |
| `const []byte(s)` and `string(b)` | Conversions between strings and read-only bytes with zero allocations. A view of a mutable slice still copies. Proven by `test/logos/run/zerocopy.lgo`. |
| `rt.Freeze(&cfg)` | A read-only view over a deep copy. Later writes to the source leave the view unchanged. Proven by `test/logos/run/freeze.lgo` and `src/cmd/go/testdata/script/logos_interop_const.txt`. |
| `const func Fib(n int) int { ... }` | Compile-time evaluation: `const fib20 = Fib(20)` folds at build time, integers at target widths, within 100,000 steps and 1 MB. Proven by `test/logos/run/ctfe.lgo` and `src/internal/types/testdata/logos/ctfe.lgo`. |
| Package-level `const primes = []int{2, 3, 5}` | Read-only data: the backing symbol is `R` in `go tool nm` and a write through `unsafe` faults; function-scope consts stay writable. Proven by `test/logos/run/rodata_write.lgo`, `test/logos/run/rodata_reflect.lgo`, and `test/logos/run/constmap_init.lgo`. |
| `const m = map[string]int{...}` | Const maps: up to 8 keys lower to a `switch`, more to a perfect-hash table with one probed slot; `range` is sorted. Lookup cost is report-only and slower than a runtime map. Proven by `test/logos/run/const_map.lgo` and `test/logos/run/constmap_switch.lgo`. |
| `$"User {name} has {n} items"` | Interpolation over string, integer, bool, float, `[]byte`, `error`, and `fmt.Stringer`, with exactly one allocation; `{{` and `}}` are literal. Proven by `test/logos/run/interp.lgo` and `src/internal/types/testdata/logos/interp.lgo`. |
| `&100`, `&f()` | Address of any operand: a fresh variable per evaluation, lowered to `new(expr)`. Proven by `test/logos/run/addr_lit.lgo` and `src/internal/types/testdata/logos/addr_lit.lgo`. |
| `Port int = 8080` before the tag | Field defaults filled into `.lgo` literals; `var` and `new` stay zero with a warning; Go literals get zero. Proven by `test/logos/run/struct_default.lgo`, `src/internal/types/testdata/logos/struct_default.lgo`, and `src/cmd/go/testdata/script/logos_m6_addr_defaults.txt`. |
| `State{A: 10, ..src()}` | Spread: identical names and types copy, explicit fields win, the source runs once, self-update is one assignment. Proven by `test/logos/run/spread.lgo` and `src/internal/types/testdata/logos/spread.lgo`. |
| `var t (int, error)` then `a, b := t` | Tuples in non-result positions, packed from calls and unpacked by assignment, read as `t.x`. Proven by `test/logos/run/tuples.lgo`, `src/internal/types/testdata/logos/tuples.lgo`, and `src/cmd/go/testdata/script/logos_tuples.txt`. |
| `var f func(int) int = x => x + 1` | Lambdas typed against their target signature, including block bodies. Proven by `test/logos/run/lambda.lgo` and `src/internal/types/testdata/logos/lambdas.lgo`. |
| `xs.filter(...).map(...).take(5)` | Chains fuse into one loop with the receiver read once; `find` and friends give `T?`; `take` and friends clamp; `sortBy` is stable. Proven by `test/logos/run/chains.lgo` and `src/internal/types/testdata/logos/chains.lgo`. A `?` nested where it cannot propagate fails; mutating the receiver fails. |
| Leading-dot line break | A newline before `.filter(...)` continues the chain; `x` newline `.5` still breaks. Pinned by `src/cmd/compile/internal/syntax/scanner_leadingdot_logos_test.go`. |
| `for i := range n { yield i }` in `func Count(n int) iter.Seq[int]` | Generators lower to a hidden yield function; early `break` stops the body. Proven by `test/logos/run/yield.lgo` and `src/internal/types/testdata/logos/yield.lgo`. |
| `scope defer note("unlock")` | Runs at block exit on every path: last in first out, args at the statement, once per loop iteration. Proven by `test/logos/run/scope_defer.lgo` and `src/cmd/compile/internal/syntax/testdata/logos/scope_defer_parse.lgo`. |
| `[capabilities]` in `logos.toml` | Policy over dependencies: `deny`, `warn`, or `allow` per sink group (`transitive_unsafe`, `transitive_exec`, `transitive_net`, `direct_reflect`, `transitive_reflect`), with `*_allowlist` globs. Proven by `src/cmd/go/testdata/script/logos_cap_unsafe.txt` and `src/cmd/go/testdata/script/logos_cap_exec.txt`. |
| `GOEXPERIMENT=logosfastcgo` | Fast calls into C for functions marked `#cgo noescape` plus `#cgo nocallback` in Go, or `//logos:cgo_fastcall` in `.lgo`. Off by default (`src/internal/buildcfg/exp.go`). Proven by `src/cmd/cgo/internal/testlogos/fastcall_test.go`, run with and without the experiment. |
| `__.LOGOS` facts | Option positions in exported declarations travel with the archive; only `.lgo` importers apply them, and stale facts fail the import with a fingerprint mismatch. Proven by `src/cmd/go/testdata/script/logos_facts.txt` and `src/cmd/go/testdata/script/logos_facts_nil.txt`. |

`?` applies to a call whose last result is `error`. The enclosing function must also return `error` last. `errdefer` is allowed only in such functions. A module holds either `.go` or `.lgo` packages, never both. Imports cross the boundary per import.

Union example (adapted from `test/logos/run/union_basic.lgo`; the fixture prints `int`, `str`, `nil` for these lines):

```go
package main

import "fmt"

type NumOrStr interface {
	int | string
}

func describe(u NumOrStr) string {
	return switch u.(type) {
	case int => "int"
	case string => "str"
	case nil => "nil"
	}
}

func main() {
	var u NumOrStr = 7
	fmt.Println(describe(u)) // int
	u = "seven"
	fmt.Println(describe(u)) // str
	var z NumOrStr
	fmt.Println(describe(z)) // nil
}
```

Option example (adapted from `test/logos/run/option_chain.lgo`; the fixture prints `anon` then `ann`):

```go
package main

import "fmt"

type User struct {
	Name string
	Age  int
}

func greet(u *User?) string { return u?.Name ?? "anon" }

func main() {
	var u *User?
	fmt.Println(greet(u)) // anon
	u = &User{Name: "ann", Age: 3}
	fmt.Println(greet(u)) // ann
}
```

Property example (adapted from `test/logos/run/properties.lgo`; the fixture prints `gorg` then `90`):

```go
package main

import "fmt"

type Creature interface {
	const Name string
	HP int
	Start() error
}

type Monster struct : Creature {
	Name string
	HP   int
}

func (m Monster) Start() error { return nil }

func main() {
	var m Monster
	m.Name = "gorg"
	m.HP = 100
	var c Creature = &m
	fmt.Println(c.Name) // gorg
	c.HP -= 10
	fmt.Println(m.HP) // 90
}
```

Const-view example (adapted from `test/logos/run/zerocopy.lgo`; the fixture prints `true` for this comparison and `0` for the allocation counts of both directions):

```go
package main

import "fmt"

func main() {
	s := "hello world, zero copy me"
	var b const []byte = const []byte(s)
	fmt.Println(string(b) == s) // true
}
```

`logos` checks capabilities on every build: a dependency that imports `unsafe` or reaches process-spawn code fails with `ERROR [capabilities::unauthorized_unsafe]` naming the import chain, unless its module allowlists the dependency. Plain `go` runs no audit.

## Agent skills

The [plugins/logos/](plugins/logos/) plugin ships four skills:

* logos: write, build, test, and eject loGo/s code.
* logos-review: review `.lgo` code against the compiler rules.
* logos-from-go: migrate a Go module to loGo/s by hand.
* logos-performance: route marked C calls through the fast-cgo entry.

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
