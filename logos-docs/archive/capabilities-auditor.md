# Section 13: capabilities auditor (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 826-864 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 13: LINKER REACHABILITY & CAPABILITIES AUDITOR (`src/cmd/link/internal/ld/deadcode.go`):**

To prevent malicious or bloated transitive dependencies from secretly invoking `reflect`, `unsafe`, `cgo`, `os/exec`, or `net` (Resolution 25, 162), the linker integrates a zero-cost call-graph audit directly into the Dead-Code Elimination (`deadcode`) mark phase.

**13.1. Protected Runtime / Stdlib Sink Symbols:**
```go
var capabilitySinks = map[string]CapabilityKind{
    // Reflection Sinks
    "reflect.ValueOf":               CapReflect,
    "reflect.TypeOf":                CapReflect,
    "reflect.New":                   CapReflect,
    // Subprocess / Execution Sinks
    "syscall.Exec":                  CapExec,
    "syscall.ForkExec":              CapExec,
    "os.startProcess":               CapExec,
    // Network Sinks
    "net.sysSocket":                 CapNet,
    // CGo Sinks
    "runtime.cgocall":               CapCgo,
    "runtime.logosFastCgoCall":      CapCgo,
}
```

**13.2. Reachability Backtrace Algorithm:**
1. During `deadcode(ctxt)`, the linker traverses symbol relocations starting from `main.main` and ` main.init`, recording `parentSym[childIdx] = callerIdx`.
2. Whenever a `childIdx` matches a symbol in `capabilitySinks`, the linker walks `parentSym` backward to find the first package outside `GOROOT/src/` (the originating package `P`).
3. If `P` is not permitted by `[capabilities]` in `bettergo.toml`, linking aborts with the exact shortest call chain:
   ```
   ERROR [capabilities::unauthorized_exec]: Transitive dependency 'github.com/evil/logger' reaches 'os.startProcess'
     --> Reachability Call Chain (cmd/link deadcode audit):
      1. main.main()                              at cmd/server/main.lgo:18
      2.   -> github.com/evil/logger.Init()       at pkg/mod/github.com/evil/logger@v1.0.2/init.go:44
      3.   -> os/exec.(*Cmd).Start()              at GOROOT/src/os/exec/exec.go:692
      4.   -> os.startProcess()                   at GOROOT/src/os/exec_posix.go:54
     = Help: Remove the dependency or explicitly authorize "github.com/evil/logger/**" under 'transitive_exec_allowlist' in bettergo.toml.
   ```

---

