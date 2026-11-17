# LoGOs

**Go with checked errors, closed unions, and a way back to plain Go.**

[![Go 1.27](https://img.shields.io/badge/go-1.27-00ADD8?logo=go)](https://go.dev/)
[![CI](https://github.com/gosuda/LoGOs/actions/workflows/logos.yml/badge.svg)](https://github.com/gosuda/LoGOs/actions/workflows/logos.yml)
[![License](https://img.shields.io/badge/license-BSD--3-blue)](LICENSE)

LoGOs is a Go toolchain fork that compiles both Go and extended `.lgo` source. You get compiler-checked error handling and nil safety, lowered to Go 1.27 by `logos`. Plain `go` keeps upstream language semantics and [command text](README.go.md). When you want out, `logos eject` returns standard Go with your comments preserved.

- **Checked errors**: `?` returns early on a non-nil error from a call returning `(T, error)`.
- **Closed unions**: the compiler checks that type switch expressions cover every variant and `nil`, unless you provide `default`.
- **Way back**: `logos eject` writes the module as standard Go that builds with upstream Go of the same version or newer ([walkthrough](#eject-to-standard-go)).
- **Shared toolchain**: Go and LoGOs modules can import each other under `logos`, with one language per module.
- **Comptime**: `const for` expands struct-field loops during compilation without run-time field lookup.
- **Diagnostics**: `logos lint` checks main-module packages without generating binaries.

```go
package main
import "fmt"
import "strconv"
func double() (int, error) {
	n := strconv.Atoi("21")?
	return 2 * n, nil
}
func main() { fmt.Println(double()) } // 42 <nil>
```

In `double()`, `?` gives `n` the value from `strconv.Atoi("21")` on success, or returns early with the error.

Contents: [Install](#install) · [Quick start](#quick-start) · [Features](#features) · [Usage](#usage) · [Language](#language) · [Agent skills](#agent-skills) · [Development](#development) · [License](#license)

## Install

Building LoGOs from source requires Bash and a bootstrap Go installation. Go 1.27.3 is known to work; Go 1.26.8 fails to compile the fork's compiler sources.

1. Clone the repository:

   ```sh
   git clone https://github.com/gosuda/LoGOs.git
   cd LoGOs
   ```

2. Capture your clean host bootstrap Go root and build the toolchain from source:

   ```sh
   unset GOROOT GOTOOLCHAIN
   upstream_go=$(command -v go)
   bootstrap_root=$("$upstream_go" env GOROOT)
   (cd src && GOROOT_BOOTSTRAP="$bootstrap_root" ./make.bash)
   ```

   If your bootstrap binary is not on your `PATH`, specify `GOROOT_BOOTSTRAP=/path/to/go` directly.

3. Configure your environment from the repository root:

   ```sh
   export GOROOT="$PWD"
   export PATH="$GOROOT/bin:$PATH"
   export GOTOOLCHAIN=local
   ```

4. Verify the installed binaries:

   ```sh
   logos version
   go version
   ```

   Both report the Go release the fork is based on, for example `logos version go1.27.4 linux/amd64` and `go version go1.27.4 linux/amd64`.

## Quick start

Create a fresh module outside the toolchain tree:

```sh
mkdir ~/logos-hello && cd ~/logos-hello
logos mod init example.com/hello
cat > main.lgo <<'EOF'
package main

import (
	"fmt"
	"strconv"
)

func double() (int, error) {
	n := strconv.Atoi("21")?
	return 2 * n, nil
}

func main() {
	fmt.Println(double())
}
EOF
logos run .
```

Output:

```text
42 <nil>
```

### Eject to standard Go

To build the ejected module, you need an upstream Go toolchain at least as new as the version `logos version` reports, because `logos mod init` writes that version into `go.mod`.

- Keep using the installation shell, where `upstream_go` holds the upstream executable.
- If that executable does not meet the requirement, select a suitable installation before continuing.
- In a new shell, set `upstream_go` to the absolute path of your upstream Go executable.

```sh
logos eject -o ../logos-hello-go
cd ../logos-hello-go
```

Check the saved upstream executable's version, then run the ejected module:

```sh
: "${upstream_go:?Set upstream_go to an upstream Go executable}"
env -u GOROOT -u GOTOOLCHAIN "$upstream_go" version
env -u GOROOT -u GOTOOLCHAIN "$upstream_go" run .
```

Output:

```text
42 <nil>
```

Ejection details:

- `logos eject [-o DIR]` requires a target directory that does not exist or is empty.
- Each `.lgo` source becomes a standard `.go` file, with source comments preserved.
- Dependency modules go under `DIR/_logos_deps`.
- Runtime support goes under `DIR/_logos_deps/rt` as module `github.com/gosuda/loGos/rt`.

## Features

### Modules and interoperability

You can import Go packages from LoGOs and LoGOs packages from Go when you build with `logos`. Exported LoGOs functions and types have standard Go signatures, with option types and closed unions represented as canonical Go types.

- Import standard library packages and external Go modules through standard Go `import` paths.
- Use `?` directly on calls returning `(T, error)`.
- Use `.lgo` for LoGOs source, or the aliases U+1F4DC, U+26A1, and the gopher sequence; use `.go` for plain Go.

Limits:
- Each module contains either `.lgo` packages or `.go` packages, never both.
- Plain `go` ignores `.lgo` files, so it cannot build a Go package that imports a LoGOs package; [eject first](#eject-to-standard-go).
- Property interfaces require an explicit assertion on the implementing struct declaration (`type S struct : I`), not automatic structural subtyping.

### Compile-time optimization

Use `logos build -O=3 ./...` or `go build -O=3 ./...` to enable stronger
scalar constant analysis. The compiler tracks known bits through wrapping
arithmetic and rotations, preserves relationships between values selected
by the same branch, and propagates constants through bit-count intrinsics
and conditional selections. These analyses can remove runtime work while
preserving integer overflow, side effects, and required panics.

The extra analysis has fixed work bounds and applies to both Go and loGo/s
packages. Lower tiers keep their existing behavior. See the
[20-technique comparison and implementation contract](logos-doc/SCALAR-OPTIMIZATION.md)
for the selected scope and measurement limits.

### Comptime

Use `const for` to unroll a loop across struct fields and `val.[f]` to read or write each field without run-time lookup. The compiler specializes the code for each struct type and folds metadata such as `f.Name` and `f.Tag.Get` into string constants without interface boxing.

`main.lgo`:

```go
package main

import (
	"comptime"
	"fmt"
)

type Server struct {
	Host string `json:"host"`
	Port int32  `json:"port"`
}

func main() {
	s := Server{Host: "db", Port: 5432}
	var copy Server
	const for _, f := range comptime.Fields[Server]() {
		fmt.Println(f.Name, f.Tag.Get("json"))
		copy.[f] = s.[f]
	}
	fmt.Println(copy)
}
```

Output:

```text
Host host
Port port
{db 5432}
```

Limits:
- Comptime inspection works only during compilation; referencing comptime APIs at run time is a compile error.
- Loop expansion accepts only `comptime.Fields[T]()` over a struct type.
- Fields follow declaration order; unexported fields from other packages are skipped.
- The default expansion limit is 4,096 nodes.

### Read-only views

Use a `const` view to read slices and pointers without defensive copies; the compiler rejects writes through that name and erases the qualifier before code generation. Converting a `string` to `const []byte` allocates nothing, and converting back is zero-copy when the bytes came from a string or constant.

`main.lgo`:

```go
package main

import (
	"fmt"
	"logos/rt"
)

type Config struct{ Port int }

func total(xs const []int) (t int) {
	for _, x := range xs {
		t += x
	}
	return
}

func main() {
	xs := []int{1, 2, 3}
	var view const []int = xs
	fmt.Println(total(view))
	xs[0] = 10
	fmt.Println(total(view))

	cfg := &Config{Port: 8080}
	snap := rt.Freeze(cfg)
	cfg.Port = 9090
	fmt.Println(snap.Port, cfg.Port)
}
```

Output:

```text
6
15
8080 9090
```

Limits:
- A view aliases existing memory; changes through the original mutable holder remain visible.
- For an isolated snapshot, `logos/rt.Freeze` deep-copies the value and returns a protected view, at the cost of heap allocations.

### Nil safety

Locally declared `*T` pointers cannot be nil, so you do not need defensive nil checks for them. Use `T?` for possible absence, then use safe navigation (`?.`), coalescing (`??`), or a nil guard before dereferencing.

`main.lgo`:

```go
package main

import "fmt"

type User struct{ Name string }

func find(name string) *User? {
	if name == "ada" {
		return &User{Name: "Ada"}
	}
	return nil
}

func describe(u *User?) string {
	if u == nil {
		return "unknown"
	}
	return u.Name
}

func main() {
	ada := find("ada")
	bob := find("bob")
	fmt.Println(ada?.Name ?? "anonymous")
	fmt.Println(bob?.Name ?? "anonymous")
	fmt.Println(describe(ada))
	fmt.Println(describe(bob))
}
```

Output:

```text
Ada
anonymous
Ada
unknown
```

Limits:
- The compiler rejects field access on optional pointers until you narrow them with a guard or use safe navigation.
- Pointers from plain Go packages are platform pointers and may be nil.
- When this toolchain compiles the Go caller, a direct call that passes nil to a non-nil LoGOs parameter gets a run-time `logos/rt.MustNonNil` check instead of a build failure.
- Calls through function values, method values, or interfaces are exceptions to that boundary check.

### Structured concurrency

Use `go group` to keep child goroutines within a block: it joins them before exit, propagates cancellation on failure, and caps concurrency with `limit`. Pair it with `logos/sync.Locked` to keep data and its mutex together and prevent unlocked field access.

`main.lgo`:

```go
package main

import (
	"context"
	"fmt"
	"logos/sync"
)

type Counter struct{ Total int }

func run(ctx context.Context) error {
	var c sync.Locked[Counter]
	go group (ctx, limit = 2) {
		for _, v := range []int{10, 20, 30} {
			go {
				c.WithLock(t => { t.Total += v })
			}
		}
	}
	var res int
	c.WithLock(t => { res = t.Total })
	fmt.Println("sum:", res)
	return nil
}

func main() {
	if err := run(context.Background()); err != nil {
		panic(err)
	}
}
```

Output:

```text
sum: 60
```

Limits:
- The compiler rejects unsupervised goroutines outside a group.
- Groups do not prevent all data races; concurrent writes to shared storage still need locks, channels, or atomic values.
- Cancellation unblocks plain channel operations written directly in a child and wakes netpoll-backed network I/O.
- Cancellation cannot interrupt cgo calls; those calls must finish before the group joins.

### Generic type switches

Use `const switch T` to choose a path by a generic function's type argument without writing reflection. Exact cases rebind the value to that type, `~basic` cases preserve the named type while matching its underlying type, and interface cases expose the interface method set.

`main.lgo`:

```go
package main

import "fmt"

type UserID int
type Word struct{ S string }
func (w Word) String() string { return w.S }

func Label[T any](v T) string {
	const switch T {
	case int:
		return fmt.Sprintf("int:%d", v+1)
	case ~int:
		return fmt.Sprintf("userid:%v", v)
	case fmt.Stringer:
		return "stringer:" + v.String()
	}
	return "other"
}

func main() {
	fmt.Println(Label(41))
	fmt.Println(Label(UserID(7)))
	fmt.Println(Label(Word{"hi"}))
	fmt.Println(Label(1.5))
}
```

Output:

```text
int:42
userid:7
stringer:hi
other
```

Limits:
- `const switch` is a statement, not an expression.
- It accepts neither a `default` arm nor multiple types per case; unhandled types fall through to statements after the block.
- The compiler folds branches only for concrete type tags and comptime generics; unmatched branches then cost nothing.
- Standard generic functions evaluate the switch at run time.

### Closed unions and property interfaces

Use a closed union to define a fixed set of variants and check type switch expressions for exhaustive coverage. Property interfaces give you field-style contracts through an explicit struct declaration assertion (`: I`); the compiler checks the cases and interface properties.

`main.lgo`:

```go
package main

import "fmt"

type Value interface{ int | string }

func tag(v Value) string {
	return switch v.(type) {
	case int => "int"
	case string => "str"
	case nil => "nil"
	}
}

type Creature interface { const Name string; HP int }

type Monster struct : Creature {
	Name string
	HP   int
}

func main() {
	m := Monster{Name: "gorg", HP: 100}
	var c Creature = &m
	c.HP -= 10
	var v Value = 7
	fmt.Println(tag(v), tag(nil), c.Name, m.HP)
}
```

Output:

```text
int nil gorg 90
```

Limits:
- Closed unions use Go's standard interface representation at run time, not a tagged-union layout.
- Writing to properties requires a pointer.
- A type switch expression over a union must handle every member and `nil`, unless it has a `default` case.

### Named tuples

Pack multi-value function results into a tuple without declaring a separate struct. You can pass tuples as arguments, store them in channels or slices, access named fields such as `t.host`, and unpack them with destructuring assignments.

`main.lgo`:

```go
package main

import "fmt"

func resolve(service string) (host string, port int) {
	if service == "api" {
		return "api.example.com", 443
	}
	return "localhost", 8080
}

func url(e (host string, port int)) string {
	return fmt.Sprintf("http://%s:%d", e.host, e.port)
}

func main() {
	api := resolve("api")
	fmt.Println(api.host, api.port)
	fmt.Println(url(api))

	host, port := api
	fmt.Println(host, port)

	queue := make(chan (host string, port int), 1)
	queue <- api
	next := <-queue
	fmt.Println(next.host, next.port)
}
```

Output:

```text
api.example.com 443
http://api.example.com:443
api.example.com 443
api.example.com 443
```

Limits:
- Field names come from the tuple type or function signature, not the literal expression.
- The compiler rejects positional indexing such as `t.0`.
- Tuples lower to generated structs; printing a tuple directly shows that struct representation.

### Element-wise array math

Use arithmetic operators (`+`, `-`, `*`, `/`, `&`, `|`, `^`) and reductions (`.sum()`, `.min()`, `.max()`, `.dot()`) on small fixed-size numeric arrays without writing loops. Reductions use a specified pairwise order, keeping numeric results identical across amd64 and arm64 vector lowerings and the scalar fallback.

`main.lgo`:

```go
package main

import "fmt"

func main() {
	a := [4]float64{1, 2, 3, 4}
	b := [4]float64{5, 6, 7, 8}

	add := a + b
	mul := a * b
	fmt.Println("add", add)
	fmt.Println("mul", mul)
	fmt.Println("sum", a.sum(), "min", a.min(), "max", a.max(), "dot", a.dot(b))

	p := [4]int32{17, -19, 100, 7}
	q := [4]int32{5, 4, -3, 2}
	fmt.Println("div", p/q)
	fmt.Println("intsum", p.sum(), "intdot", p.dot(q))
}
```

Output:

```text
add [6 8 10 12]
mul [5 12 21 32]
sum 10 min 1 max 4 dot 70
div [3 -4 -33 3]
intsum 105 intdot -277
```

Limits:
- Element-wise math works on arrays, not slices.
- The element count must be a power of two, and the total size must not exceed 64 bytes.
- Vectorized execution needs `GOEXPERIMENT=simd` and a supported architecture and instruction set.
- Otherwise, the compiler emits unrolled scalar instructions with identical numeric results but no throughput guarantee.

### Automatic loop vectorization

Ordinary Go and loGo/s integer loops can use SIMD without an import or annotation:

```go
func add(dst []int32, delta int32) {
	for i := range dst {
		dst[i] += delta
	}
}
```

Enable the existing SIMD experiment. On amd64, select an AVX2-capable target:

```sh
GOEXPERIMENT=simd GOAMD64=v3 go build ./...
GOEXPERIMENT=simd GOAMD64=v3 logos build ./...
```

Arm64 uses baseline NEON with `GOEXPERIMENT=simd`. The compiler emits 128-bit vector operations for eligible integer maps and handles the remaining elements with the original scalar loop. Eligible loops may start at a nonzero invariant index; the vector prefix and scalar tail preserve that start and leave earlier elements untouched. Supported operations are addition, subtraction, negation, bitwise complement, AND, OR, and XOR on 8-, 16-, 32-, and 64-bit integers, plus multiplication on 16- and 32-bit integers. Constant shifts smaller than the element width support left shifts and unsigned right shifts on 16-, 32-, and 64-bit integers, and signed right shifts on 16- and 32-bit integers. Byte doubling also vectorizes; other byte shifts and variable counts remain scalar.

Loops over several slices need bounds that the compiler can prove. An overlap guard keeps the original scalar behavior when their memory ranges are unsafe to vectorize. Loops containing calls, unresolved checks, reductions, or floating-point expressions remain scalar, as do instrumented builds, `//go:nosplit` functions, and unsupported loop shapes. When the current goroutine enables `runtime/debug.SetPanicOnFault`, eligible loops use the original scalar path to preserve writes before a recovered fault. `-gcflags=-d=ssa/autovec/debug=1` reports accepted loops; `-gcflags=-d=ssa/autovec/off` disables the pass. See [the exact eligibility rules](logos-doc/SPEC.md#134-automatic-loop-vectorization).

### Also in the language

You can also use these extensions; see [logos-doc/SPEC.md](logos-doc/SPEC.md) Sec 1.4:

- Error propagation (`n := f()?` or `f()?`): unwrap values or return early on a non-nil error from a function returning `(T, error)`.
- Error cleanup (`errdefer f()`): register cleanup that runs only when the enclosing function exits with an error, a panic, or Goexit.
- String interpolation (`$"user {name}: {count}"`): embed formatted expressions in string literals.
- Iterator chains (`xs.filter(...).map(...).take(n)`): transform slices with predicate and mapping closures.
- Literal addressing (`&100`, `&f()`): take the address of a scalar literal or function call result without a temporary variable.
- Struct field defaults (`type Config struct { Port int = 8080 }`): evaluate default values when struct literals omit fields.

## Usage

Use `logos run`, `logos build`, and `logos test` on LoGOs modules as you would use their `go` counterparts on Go modules.

Five commands are specific to LoGOs:
- `logos context`
- [`logos eject [-o DIR]`](#eject-to-standard-go)
- [`logos lint`](#lint)
- `logos lsp`
- `logos next`

### Lint

The `logos lint` command runs in-compiler checks through escape analysis and stops before code generation. It emits diagnostics without writing executable binaries:

```sh
logos lint [build flags] [--pedantic] [--json-diagnostics] [packages]
```

#### Scope and rules

- Target scope: `logos lint` checks packages in the main module only, never the standard library or external module dependencies.
- Language checks:
  - In `.lgo` packages, the compiler runs all registered compiler rules.
  - In standard `.go` packages, it runs only language-neutral rules.
- Directives:
  - `//logos:allow(<rule>)`: place directly above a declaration or statement to suppress a `deny` or `warn` diagnostic. In Go packages, `forbid` rules run at `deny` severity and this directive can silence them. In `.lgo` packages, it cannot silence `forbid` rules.
  - `//logos:zeroalloc`: place above a function declaration to require the compiler to reject local heap allocations at that escape-analysis site (see [logos-doc/SPEC.md](logos-doc/SPEC.md)).
- Flags:
  - `--pedantic`: raise the pedantic group from `allow` to `warn`. A stricter `logos.toml` severity stays in force. `--pedantic=false` turns the group off.
  - `--json-diagnostics`: emit one JSON object per diagnostic line on standard error.

#### Unsafe code

`logos build` allows `unsafe` imports and dependencies that contain assembly by default. To audit dependency use explicitly, set `transitive_unsafe = "deny"` or `"warn"` in the `[capabilities]` table of `logos.toml`.

`logos lint` checks risky pointer conversions in both `.go` and `.lgo` packages. The existing `correctness::arena_pointer_escape` rule now reports saved integer addresses converted back to pointers and standalone reflection-header misuse. Ordinary pointer returns and addresses used only as integers pass. Supported immediate pointer arithmetic and immediate reflection conversions also pass.

The rule defaults to `deny` and runs only under `logos lint`. Use `//logos:allow(correctness::arena_pointer_escape)` above a statement or declaration for a reviewed exception. A clean result does not prove allocation bounds or lifetime safety; see [the unsafe lint contract](logos-doc/SPEC.md#12-rfc-102-compiler-linter-and-rule-set).

## Language

One row per shipped form, with milestone IDs from [logos-doc/IMPLEMENTATION.md](logos-doc/IMPLEMENTATION.md). The full language definition is in [logos-doc/SPEC.md](logos-doc/SPEC.md).

| Milestone | Form | Summary |
|---|---|---|
| M1 | `?`, `errdefer`, eject | Checked error propagation with early return, failure-only cleanup, and error traces; see [Eject to standard Go](#eject-to-standard-go). |
| M2 | Closed unions | Fixed variant sets with exhaustive type switches. |
| M7 | `const if`, `const for`, `const switch T`, `comptime.ViewAs` | Compile-time branching, field unrolling, type dispatch, and read-only views. |
| M8 | `go group (ctx, limit = N)` | Structured concurrency with automatic joins, cancellation, and a concurrency cap. |
| M9 | `logos lint`, `--json-diagnostics` | In-compiler checks that stop before code generation, with one JSON object per diagnostic. |
| M10 | `logos next`, `logos context` | Rewrites a Go module toward LoGOs; prints the declaration and signatures for one symbol. |
| M11 | Generated `MarshalJSON` / `UnmarshalJSON` | Static `MarshalJSON` and `UnmarshalJSON` methods on structs that have JSON tags, defaults, or non-nil pointer fields and no embedded fields, with `encoding/json`-matching output. |
| M13 | `-O` | Optimization levels `-O=0` to `-O=3` and `-O=s` on the command line, with inline budgets and variable-make stack buffers. In `logos.toml`, `opt_level` takes `O0` to `O3` or `Os` (default `O2`). Backend, runtime, and tooling improvements apply to Go and LoGOs packages under `logos` and plain `go`, except the items SPEC §0.1 lists as LoGOs-only or logos-only. `target_cpu` applies to `logos` builds only; plain `go` keeps the upstream GOAMD64 and GOARM64 defaults. |
| M14 | Fixed-array SIMD | Element-wise operators and reductions on small fixed-size arrays. The vectorized lowering needs `GOEXPERIMENT=simd`; without it the compiler emits unrolled scalar code with the same results. |
| M16 | WASM profile | Automatic for GOOS=js and GOOS=wasip1 under `logos`; `--wasm-features=stack-switching`, `--wasm-loader`, and the `//logos:wasmimport` / `//logos:wasmexport` directives select options. |
| M17 | Smart stack | Flag only: `GOEXPERIMENT=logosstack` starts goroutines at per-call-site sizes. |
| M18 | Arena and bump allocation | On by default under `logos` (`[build] arena = true`); `GOEXPERIMENT=logosarena` under plain `go`. |
| M20 | Cgo optimization | Candidate `GOEXPERIMENT=logoscgoopt` removes return-frame relocation for ordinary `#cgo nocallback` calls while keeping `runtime.cgocall`. The separate `logosfastcgo` experiment selects the fast entry for `//logos:cgo_fastcall` or the `noescape`/`nocallback` pair. Both are off by default; see [SPEC Sec 3.6](logos-doc/SPEC.md#36-rfc-306-cgo-compiler-optimization) and the [validation status](logos-doc/benchmarks/M20-S2.md). |
| M21 | `logos lsp` | Editor setup: gopls proxy mapping `.lgo` positions, with inlay hints, hover, and code actions. |
| M22 | Verification suites | Directive only: `//logos:zeroalloc` rejects escape-analysis heap sites in the marked function, not run-time heap fallbacks. |

## Agent skills

The [plugins/logos/](plugins/logos/) directory packages skills for coding assistants:

- [`logos`](plugins/logos/skills/logos/SKILL.md): Author, build, test, and eject LoGOs modules.
- [`logos-review`](plugins/logos/skills/logos-review/SKILL.md): Review `.lgo` code against compiler rules, diagnostics, and language invariants.
- [`logos-from-go`](plugins/logos/skills/logos-from-go/SKILL.md): Guide migration of existing plain Go packages to LoGOs syntax.
- [`logos-performance`](plugins/logos/skills/logos-performance/SKILL.md): Assist with -O tiers, SIMD, WASM, stack, arena, fast cgo, and zero-alloc.
- [`logos-concurrency`](plugins/logos/skills/logos-concurrency/SKILL.md): Structure goroutine groups and shared state in LoGOs.

To provide coding agents with project conventions and compiler constraints, copy [plugins/logos/templates/AGENTS.md](plugins/logos/templates/AGENTS.md) into the target repository root. See [plugins/logos/](plugins/logos/) for plugin structure and usage documentation.

## Development

Follow the [Install](#install) instructions to compile the toolchain from source.
Run the commands below from the toolchain repository root, not the example module.

### Running tests

Each command enters `src` before it runs the tests:

```sh
(cd src && go test -count=1 cmd/internal/testdir -run 'Test/logos/')
(cd src && go test -count=1 -short cmd/compile/internal/types2 -run TestLogos)
(cd src && go test -count=1 cmd/go -run 'Script/^logos_')
```

### Contributing

1. Fork the repository on GitHub.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Add test fixtures under `test/logos/run/` or `src/internal/types/testdata/logos/` covering new syntax or compiler behavior.
4. Verify that targeted tests pass before submitting changes.
5. Commit your changes with concise, descriptive commit messages adhering to standard Go conventions.
6. Open a pull request on GitHub.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [README.go.md](README.go.md) for contribution guidelines.

## License

LoGOs additions and toolchain modifications use the same BSD-3-Clause license as the standard Go source. See [LICENSE](LICENSE) for the terms.
