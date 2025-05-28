<!--
Output from relnote todo that was generated and reviewed on 2025-05-23, plus summary info from bug/CL: -->

### TODO

**Please turn these into proper release notes**

<!-- TODO: CL 420114 has a RELNOTE comment without a suggested text (from RELNOTE comment in https://go.dev/cl/420114) -->
all: implement plugin build mode for riscv64

<!-- TODO: CL 660996 has a RELNOTE comment without a suggested text (from RELNOTE comment in https://go.dev/cl/660996) -->
cmd/link/internal/ld: introduce -funcalign=N option  
This patch adds linker option -funcalign=N that allows to set alignment
for function entries.  
For \#72130.

<!-- TODO: accepted proposal https://go.dev/issue/32816 (from https://go.dev/cl/645155, https://go.dev/cl/645455, https://go.dev/cl/645955, https://go.dev/cl/646255, https://go.dev/cl/646455, https://go.dev/cl/646495, https://go.dev/cl/646655, https://go.dev/cl/646875, https://go.dev/cl/647298, https://go.dev/cl/647299, https://go.dev/cl/647736, https://go.dev/cl/648581, https://go.dev/cl/648715, https://go.dev/cl/648976, https://go.dev/cl/648995, https://go.dev/cl/649055, https://go.dev/cl/649056, https://go.dev/cl/649057, https://go.dev/cl/649456, https://go.dev/cl/649476, https://go.dev/cl/650755, https://go.dev/cl/651615, https://go.dev/cl/651617, https://go.dev/cl/651655, https://go.dev/cl/653436) -->
cmd/fix: automate migrations for simple deprecations

<!-- TODO: accepted proposal https://go.dev/issue/34055 (from https://go.dev/cl/625577) -->
cmd/go: allow serving module under the subdirectory of git repository  
cmd/go: add subdirectory support to go-import meta tag  
This CL adds ability to specify a subdirectory in the go-import meta tag.
A go-import meta tag now will support:
\<meta name="go-import" content="root-path vcs repo-url subdir">  
Fixes: \#34055

<!-- TODO: accepted proposal https://go.dev/issue/42965 (from https://go.dev/cl/643355, https://go.dev/cl/670656, https://go.dev/cl/670975, https://go.dev/cl/674076) -->
cmd/go: add global ignore mechanism for Go tooling ecosystem

<!-- TODO: accepted proposal https://go.dev/issue/51430 (from https://go.dev/cl/644997, https://go.dev/cl/646355) -->
cmd/cover: extend coverage testing to include applications

<!-- TODO: accepted proposal https://go.dev/issue/60905 (from https://go.dev/cl/645795) -->
all: add GOARM64=v8.1 and so on  
runtime: check LSE support on ARM64 at runtime init  
Check presence of LSE support on ARM64 chip if we targeted it at compile
time.  
Related to \#69124  
Updates \#60905  
Fixes \#71411

<!-- TODO: accepted proposal https://go.dev/issue/61476 (from https://go.dev/cl/633417) -->
all: add GORISCV64 environment variable  
cmd/go: add rva23u64 as a valid value for GORISCV64  
The RVA23 profile was ratified on the 21st of October 2024.
https://riscv.org/announcements/2024/10/risc-v-announces-ratification-of-the-rva23-profile-standard/
Now that it's ratified we can add rva23u64 as a valid value for the
GORISCV64 environment variable. This will allow the compiler and
assembler to generate instructions made mandatory by the new profile
without a runtime check.  Examples of such instructions include those
introduced by the Vector and Zicond extensions.
Setting GORISCV64=rva23u64 defines the riscv64.rva20u64,
riscv64.rva22u64 and riscv64.rva23u64 build tags, sets the internal
variable buildcfg.GORISCV64 to 23 and defines the macros
GORISCV64_rva23u64, hasV, hasZba, hasZbb, hasZbs, hasZfa, and
hasZicond for use in assembly language code.  
Updates \#61476

<!-- TODO: accepted proposal https://go.dev/issue/61716 (from https://go.dev/cl/644475) -->
math/rand/v2: revised API for math/rand  
rand: deprecate in favor of math/rand/v2  
For golang/go#61716  
Fixes golang/go#71373

<!-- TODO: accepted proposal https://go.dev/issue/64876 (from https://go.dev/cl/649435) -->
cmd/go: enable GOCACHEPROG by default   
cmd/go/internal/cacheprog: drop Request.ObjectID  
ObjectID was a misnaming of OutputID from cacheprog's initial
implementation. It was maintained for compatibility with existing
cacheprog users in 1.24 but can be removed in 1.25.

<!-- TODO: accepted proposal https://go.dev/issue/68106 (from https://go.dev/cl/628175, https://go.dev/cl/674158, https://go.dev/cl/674436, https://go.dev/cl/674437, https://go.dev/cl/674555, https://go.dev/cl/674556, https://go.dev/cl/674575, https://go.dev/cl/675075, https://go.dev/cl/675076, https://go.dev/cl/675155, https://go.dev/cl/675235) -->
cmd/go: doc -http should start a pkgsite instance and open a browser

<!-- TODO: accepted proposal https://go.dev/issue/69712 (from https://go.dev/cl/619955) -->
cmd/go: -json flag for go version -m  
cmd/go: support -json flag in go version  
It supports features described in the issue:
- add -json flag for 'go version -m' to print json encoding of
  runtime/debug.BuildSetting to standard output.
- report an error when specifying -json flag without -m.
- print build settings on seperated line for each binary  
Fixes \#69712

<!-- TODO: accepted proposal https://go.dev/issue/70123 (from https://go.dev/cl/657116) -->
crypto: mechanism to enable FIPS mode

<!-- TODO: accepted proposal https://go.dev/issue/70128 (from https://go.dev/cl/645716, https://go.dev/cl/647455, https://go.dev/cl/651215, https://go.dev/cl/651256, https://go.dev/cl/652136, https://go.dev/cl/652215, https://go.dev/cl/653095, https://go.dev/cl/653139, https://go.dev/cl/653156, https://go.dev/cl/654395) -->
spec: remove notion of core types

<!-- TODO: accepted proposal https://go.dev/issue/70200 (from https://go.dev/cl/674916) -->
cmd/go: add fips140 module selection mechanism  
lib/fips140: set inprocess.txt to v1.0.0

<!-- TODO: accepted proposal https://go.dev/issue/70464 (from https://go.dev/cl/630137) -->
testing: panic in AllocsPerRun during parallel test  
testing: panic in AllocsPerRun if parallel tests are running  
If other tests are running, AllocsPerRun's result will be inherently flaky.
Saw this with CL 630136 and \#70327.  
Proposed in \#70464.  
Fixes \#70464.

<!-- TODO: accepted proposal https://go.dev/issue/71845 (from https://go.dev/cl/665796, https://go.dev/cl/666935) -->
encoding/json/v2: add new JSON API behind a GOEXPERIMENT=jsonv2 guard

<!-- TODO: accepted proposal https://go.dev/issue/71867 (from https://go.dev/cl/666476, https://go.dev/cl/666755, https://go.dev/cl/673119, https://go.dev/cl/673696) -->
cmd/go, cmd/distpack: build and run tools that are not necessary for builds as needed and don't include in binary distribution

<!-- Items that don't need to be mentioned in Go 1.25 release notes but are picked up by relnote todo 

TODO: accepted proposal https://go.dev/issue/30999 (from https://go.dev/cl/671795)
net: reject leading zeros in IP address parsers  
net: don't test with leading 0 in ipv4 addresses  
Updates \#30999
Fixes \#73378

TODO: accepted proposal https://go.dev/issue/36532 (from https://go.dev/cl/647555)
testing: reconsider adding Context method to testing.T  
database/sql: use t.Context in tests  
Replace "context.WithCancel(context.Background())" with "t.Context()".  
Updates \#36532

TODO: accepted proposal https://go.dev/issue/48429 (from https://go.dev/cl/648577)
cmd/go: track tool dependencies in go.mod  
cmd/go: document -modfile and other flags for 'go tool'  
Mention -modfile, -C, -overlay, and -modcacherw in the 'go tool'
documentation. We let a reference to 'go help build' give a pointer to
more detailed information.
The -modfile flag in particular is newly useful with the Go 1.24 support
for user-defined tools with 'go tool'.  
Updates \#48429  
Updates \#33926  
Updates \#71663  
Fixes \#71502

TODO: accepted proposal https://go.dev/issue/51572 (from https://go.dev/cl/651996)
cmd/go: add 'unix' build tag but not \*\_unix.go file support  
os, syscall: use unix build tag where appropriate  
These newly added files may use the unix build tag instead of explitly
listing all unix-like GOOS values.  
For \#51572

TODO: accepted proposal https://go.dev/issue/53757 (from https://go.dev/cl/644575)
x/sync/errgroup: propagate panics and Goexits through Wait  
errgroup: propagate panic and Goexit through Wait  
Recovered panic values are wrapped and saved in Group.
Goexits are detected by a sentinel value set after the given function
returns normally. Wait propagates the first instance of a panic or
Goexit.
According to the runtime.Goexit after the code will not be executed,
with a bool, if f not call runtime.Goexit, is true,
determine whether to propagate runtime.Goexit.  
Fixes golang/go#53757

TODO: accepted proposal https://go.dev/issue/54743 (from https://go.dev/cl/532415)
ssh: add server side support for Diffie Hellman Group Exchange

TODO: accepted proposal https://go.dev/issue/57792 (from https://go.dev/cl/649716, https://go.dev/cl/651737)
x/crypto/x509roots: new module

TODO: accepted proposal https://go.dev/issue/58523 (from https://go.dev/cl/538235)
ssh: expose negotiated algorithms  
Fixes golang/go#58523  
Fixes golang/go#46638

TODO: accepted proposal https://go.dev/issue/61537 (from https://go.dev/cl/531935)
ssh: export supported algorithms  
Fixes golang/go#61537

TODO: accepted proposal https://go.dev/issue/61901 (from https://go.dev/cl/647875)
bytes, strings: add iterator forms of existing functions

TODO: accepted proposal https://go.dev/issue/61940 (from https://go.dev/cl/650235)
all: fix links to Go wiki  
The Go wiki on GitHub has moved to go.dev in golang/go#61940.

TODO: accepted proposal https://go.dev/issue/64207 (from https://go.dev/cl/647015, https://go.dev/cl/652235)
all: end support for macOS 10.15 in Go 1.23

TODO: accepted proposal https://go.dev/issue/67839 (from https://go.dev/cl/646535)
x/sys/unix: access to ELF auxiliary vector  
runtime: adjust comments for auxv getAuxv  
github.com/cilium/ebpf no longer accesses getAuxv using linkname but now
uses the golang.org/x/sys/unix.Auxv wrapper introduced in
go.dev/cl/644295.
Also adjust the list of users to include x/sys/unix.  
Updates \#67839  
Updates \#67401

TODO: accepted proposal https://go.dev/issue/68780 (from https://go.dev/cl/659835)
x/term: support pluggable history  
term: support pluggable history  
Expose a new History interface that allows replacement of the default
ring buffer to customize what gets added or not; as well as to allow
saving/restoring history on either the default ringbuffer or a custom
replacement.  
Fixes golang/go#68780

TODO: accepted proposal https://go.dev/issue/69095 (from https://go.dev/cl/649320, https://go.dev/cl/649321, https://go.dev/cl/649337, https://go.dev/cl/649376, https://go.dev/cl/649377, https://go.dev/cl/649378, https://go.dev/cl/649379, https://go.dev/cl/649380, https://go.dev/cl/649397, https://go.dev/cl/649398, https://go.dev/cl/649419, https://go.dev/cl/649497, https://go.dev/cl/649498, https://go.dev/cl/649618, https://go.dev/cl/649675, https://go.dev/cl/649676, https://go.dev/cl/649677, https://go.dev/cl/649695, https://go.dev/cl/649696, https://go.dev/cl/649697, https://go.dev/cl/649698, https://go.dev/cl/649715, https://go.dev/cl/649717, https://go.dev/cl/649718, https://go.dev/cl/649755, https://go.dev/cl/649775, https://go.dev/cl/649795, https://go.dev/cl/649815, https://go.dev/cl/649835, https://go.dev/cl/651336, https://go.dev/cl/651736, https://go.dev/cl/651737, https://go.dev/cl/658018)
all, x/build/cmd/relui: automate go directive maintenance in golang.org/x repositories

TODO: accepted proposal https://go.dev/issue/70859 (from https://go.dev/cl/666056, https://go.dev/cl/670835, https://go.dev/cl/672015, https://go.dev/cl/672016, https://go.dev/cl/672017)
x/tools/go/ast/inspector: add Cursor, to enable partial and multi-level traversals

-->
