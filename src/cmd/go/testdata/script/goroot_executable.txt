[compiler:gccgo] skip
[short] skip 'builds and links another cmd/go'

mkdir $WORK/new/bin

# $GOROOT/bin/go is whatever the user has already installed
# (using make.bash or similar). We can't make assumptions about what
# options it may have been built with, such as -trimpath or not.
# Instead, we build a fresh copy of the binary with known settings.
go build -o $WORK/new/bin/go$GOEXE cmd/go &
go build -trimpath -o $WORK/bin/check$GOEXE check.go &
wait

env TESTGOROOT=$GOROOT
env GOROOT=

# Relocated Executable
exec $WORK/bin/check$GOEXE $WORK/new/bin/go$GOEXE $TESTGOROOT

# Relocated Tree:
# If the binary is sitting in a bin dir next to ../pkg/tool, that counts as a GOROOT,
# so it should find the new tree.
mkdir $WORK/new/pkg/tool
exec $WORK/bin/check$GOEXE $WORK/new/bin/go$GOEXE $WORK/new

[!symlink] stop 'The rest of the test cases require symlinks'

# Symlinked Executable:
# With a symlink into go tree, we should still find the go tree.
mkdir $WORK/other/bin
symlink $WORK/other/bin/go$GOEXE -> $WORK/new/bin/go$GOEXE
exec $WORK/bin/check$GOEXE $WORK/new/bin/go$GOEXE $WORK/new

rm $WORK/new/pkg

# Runtime GOROOT:
# Binaries built in the new tree should report the
# new tree when they call runtime.GOROOT.
symlink $WORK/new/src -> $TESTGOROOT/src
symlink $WORK/new/pkg -> $TESTGOROOT/pkg
exec $WORK/new/bin/go$GOEXE run check_runtime_goroot.go $WORK/new

-- check.go --
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	exe := os.Args[1]
	want := os.Args[2]
	cmd := exec.Command(exe, "env", "GOROOT")
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s env GOROOT: %v, %s\n", exe, err, out)
		os.Exit(1)
	}
	goroot, err := filepath.EvalSymlinks(strings.TrimSpace(string(out)))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	want, err = filepath.EvalSymlinks(want)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if !strings.EqualFold(goroot, want) {
		fmt.Fprintf(os.Stderr, "go env GOROOT:\nhave %s\nwant %s\n", goroot, want)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "go env GOROOT: %s\n", goroot)

}
-- check_runtime_goroot.go --
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func main() {
	goroot, err := filepath.EvalSymlinks(runtime.GOROOT())
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	want, err := filepath.EvalSymlinks(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if !strings.EqualFold(goroot, want) {
		fmt.Fprintf(os.Stderr, "go env GOROOT:\nhave %s\nwant %s\n", goroot, want)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "go env GOROOT: %s\n", goroot)

}
