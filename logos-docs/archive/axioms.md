# Section 0: architectural axioms (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 10-21 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 0: ARCHITECTURAL AXIOMS & TRANSPILATION CONTRACT:**

* **0.1. Upstream Compatibility:** The compiler SHALL compile all valid upstream Go source files (`.go`) and module dependencies (`go.mod` / `GOPATH/pkg/mod`) without syntax or runtime behavior modification.
* **0.2. Zero New Global Keywords:** The language SHALL NOT introduce any new globally reserved keywords that invalidate existing Go identifiers. All extensions SHALL reuse Go's 25 reserved keywords (`const`, `interface`, `struct`, `switch`, `defer`, `for`, `go`, `func`, `type`) or contextual statements (`errdefer`, `yield`).
* **0.3. Bidirectional Pure Go Transpilation (`--emit-go`):**
  * The command `logos build --emit-go ./...` (or `logos eject`) SHALL lower all Tier 1 and Tier 2 `.lgo`, `.📜`, `.⚡`, and `.ʕ◔ϖ◔ʔ` source files into 100% standard, `gofmt`-compliant `.gen.go` files.
  * Every emitted `.gen.go` file SHALL include `//go:build !logos` at line 1 so that standard `go build` compiles `.gen.go` while `logos build` compiles the native `.lgo` source without symbol redeclaration conflicts.
  * The transpiler SHALL inject collision-proof helper imports (`import _logos_strconv "strconv"`, `import _logos_errors "errors"`, `import _logos_unsafe "unsafe"`) and strip virtual `import "comptime"`.
* **0.4. AI-First Generation & 1-Second Human Supervision:** Every side effect, mutability contract, nilability state, and error path SHALL be explicitly visible in function signatures and call sites. Dangerous edge cases SHALL fail at compile time via `go lint` (`forbid` / `deny`).

---

