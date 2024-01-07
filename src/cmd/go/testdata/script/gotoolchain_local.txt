# This test uses the fake toolchain switch support in cmd/go/internal/toolchain.Switch
# to exercise all the version selection logic without needing actual toolchains.
# See gotoolchain_net.txt and gotoolchain_path.txt for tests of network and PATH toolchains.

env TESTGO_VERSION=go1.500
env TESTGO_VERSION_SWITCH=switch

# GOTOOLCHAIN=auto runs default toolchain without a go.mod or go.work
env GOTOOLCHAIN=auto
go version
stdout go1.500

# GOTOOLCHAIN=path runs default toolchain without a go.mod or go.work
env GOTOOLCHAIN=path
go version
stdout go1.500

# GOTOOLCHAIN=asdf is a syntax error
env GOTOOLCHAIN=asdf
! go version
stderr '^go: invalid GOTOOLCHAIN "asdf"$'

# GOTOOLCHAIN=version is used directly.
env GOTOOLCHAIN=go1.600
go version
stdout go1.600

env GOTOOLCHAIN=go1.400
go version
stdout go1.400

# GOTOOLCHAIN=version+auto sets a minimum.
env GOTOOLCHAIN=go1.600+auto
go version
stdout go1.600

env GOTOOLCHAIN=go1.400.0+auto
go version
stdout go1.400.0

# GOTOOLCHAIN=version+path sets a minimum too.
env GOTOOLCHAIN=go1.600+path
go version
stdout go1.600

env GOTOOLCHAIN=go1.400+path
go version
stdout go1.400

# Create a go.mod file and test interactions with auto and path.

# GOTOOLCHAIN=auto uses go line if newer than local toolchain.
env GOTOOLCHAIN=auto
go mod init m
go mod edit -go=1.700 -toolchain=none
go version
stdout 1.700

go mod edit -go=1.300 -toolchain=none
go version
stdout 1.500 # local toolchain is newer

go mod edit -go=1.700 -toolchain=go1.300
go version
stdout go1.700 # toolchain too old, ignored

go mod edit -go=1.300 -toolchain=default
go version
stdout go1.500

go mod edit -go=1.700 -toolchain=default
go version
stdout go1.500 # toolchain local is like GOTOOLCHAIN=local and wins
! go build
stderr '^go: go.mod requires go >= 1.700 \(running go 1.500; go.mod sets toolchain default\)'

# GOTOOLCHAIN=path does the same.
env GOTOOLCHAIN=path
go mod edit -go=1.700 -toolchain=none
go version
stdout 1.700

go mod edit -go=1.300 -toolchain=none
go version
stdout 1.500 # local toolchain is newer

go mod edit -go=1.700 -toolchain=go1.300
go version
stdout go1.700 # toolchain too old, ignored

go mod edit -go=1.300 -toolchain=default
go version
stdout go1.500

go mod edit -go=1.700 -toolchain=default
go version
stdout go1.500 # toolchain default applies even if older than go line
! go build
stderr '^go: go.mod requires go >= 1.700 \(running go 1.500; GOTOOLCHAIN=path; go.mod sets toolchain default\)'

# GOTOOLCHAIN=min+auto with toolchain default uses min, not local

env GOTOOLCHAIN=go1.400+auto
go mod edit -go=1.300 -toolchain=default
go version
stdout 1.400 # not 1.500 local toolchain

env GOTOOLCHAIN=go1.600+auto
go mod edit -go=1.300 -toolchain=default
go version
stdout 1.600 # not 1.500 local toolchain

# GOTOOLCHAIN names can have -suffix
env GOTOOLCHAIN=go1.800-bigcorp
go version
stdout go1.800-bigcorp

env GOTOOLCHAIN=auto
go mod edit -go=1.999 -toolchain=go1.800-bigcorp
go version
stdout go1.999

go mod edit -go=1.777 -toolchain=go1.800-bigcorp
go version
stdout go1.800-bigcorp

# go.work takes priority over go.mod
go mod edit -go=1.700 -toolchain=go1.999-wrong
go work init
go work edit -go=1.400 -toolchain=go1.600-right
go version
stdout go1.600-right

go work edit -go=1.400 -toolchain=default
go version
stdout go1.500

# go.work misconfiguration does not break go work edit
# ('go 1.600 / toolchain local' forces use of 1.500 which can't normally load that go.work; allow work edit to fix it.)
go work edit -go=1.600 -toolchain=default
go version
stdout go1.500

go work edit -toolchain=none
go version
stdout go1.600

rm go.work

# go.mod misconfiguration does not break go mod edit
go mod edit -go=1.600 -toolchain=default
go version
stdout go1.500

go mod edit -toolchain=none
go version
stdout go1.600

# toolchain built with a custom version should know how it compares to others

env TESTGO_VERSION=go1.500-bigcorp
go mod edit -go=1.499 -toolchain=none
go version
stdout go1.500-bigcorp

go mod edit -go=1.499 -toolchain=go1.499
go version
stdout go1.500-bigcorp

go mod edit -go=1.500 -toolchain=none
go version
stdout go1.500-bigcorp

go mod edit -go=1.500 -toolchain=go1.500
go version
stdout go1.500-bigcorp

go mod edit -go=1.501 -toolchain=none
go version
stdout go1.501

	# If toolchain > go, we must upgrade to the indicated toolchain (not just the go version).
go mod edit -go=1.499 -toolchain=go1.501
go version
stdout go1.501

env TESTGO_VERSION='go1.500 (bigcorp)'
go mod edit -go=1.499 -toolchain=none
go version
stdout 'go1.500 \(bigcorp\)'

go mod edit -go=1.500 -toolchain=none
go version
stdout 'go1.500 \(bigcorp\)'

go mod edit -go=1.501 -toolchain=none
go version
stdout go1.501

# go install m@v and go run m@v should ignore go.mod and use m@v
env TESTGO_VERSION=go1.2.3
go mod edit -go=1.999 -toolchain=go1.998

! go install rsc.io/fortune/nonexist@v0.0.1
stderr '^go: rsc.io/fortune@v0.0.1 requires go >= 1.21rc999; switching to go1.22.9$'
stderr '^go: rsc.io/fortune/nonexist@v0.0.1: module rsc.io/fortune@v0.0.1 found, but does not contain package rsc.io/fortune/nonexist'

! go run rsc.io/fortune/nonexist@v0.0.1
stderr '^go: rsc.io/fortune@v0.0.1 requires go >= 1.21rc999; switching to go1.22.9$'
stderr '^go: rsc.io/fortune/nonexist@v0.0.1: module rsc.io/fortune@v0.0.1 found, but does not contain package rsc.io/fortune/nonexist'

# go install should handle unknown flags to find m@v
! go install -unknownflag rsc.io/fortune/nonexist@v0.0.1
stderr '^go: rsc.io/fortune@v0.0.1 requires go >= 1.21rc999; switching to go1.22.9$'
stderr '^flag provided but not defined: -unknownflag'

! go install -unknownflag arg rsc.io/fortune/nonexist@v0.0.1
stderr '^go: rsc.io/fortune@v0.0.1 requires go >= 1.21rc999; switching to go1.22.9$'
stderr '^flag provided but not defined: -unknownflag'

# go run cannot handle unknown boolean flags
! go run -unknownflag rsc.io/fortune/nonexist@v0.0.1
! stderr switching
stderr '^flag provided but not defined: -unknownflag'

! go run -unknownflag oops rsc.io/fortune/nonexist@v0.0.1
! stderr switching
stderr '^flag provided but not defined: -unknownflag'

# go run can handle unknown flag with argument.
! go run -unknown=flag rsc.io/fortune/nonexist@v0.0.1
stderr '^go: rsc.io/fortune@v0.0.1 requires go >= 1.21rc999; switching to go1.22.9$'
stderr '^flag provided but not defined: -unknown'

# go install m@v should handle queries
! go install rsc.io/fortune/nonexist@v0.0
stderr '^go: rsc.io/fortune@v0.0.1 requires go >= 1.21rc999; switching to go1.22.9$'
stderr '^go: rsc.io/fortune/nonexist@v0.0: module rsc.io/fortune@v0.0 found \(v0.0.1\), but does not contain package rsc.io/fortune/nonexist'

# go run m@v should handle queries
! go install rsc.io/fortune/nonexist@v0
stderr '^go: rsc.io/fortune@v0.0.1 requires go >= 1.21rc999; switching to go1.22.9$'
stderr '^go: rsc.io/fortune/nonexist@v0: module rsc.io/fortune@v0 found \(v0.0.1\), but does not contain package rsc.io/fortune/nonexist'

# go install m@v should use local toolchain if not upgrading
! go install rsc.io/fortune/nonexist@v1
! stderr go1.22.9
! stderr switching
stderr '^go: downloading rsc.io/fortune v1.0.0$'
stderr '^go: rsc.io/fortune/nonexist@v1: module rsc.io/fortune@v1 found \(v1.0.0\), but does not contain package rsc.io/fortune/nonexist'

# go run m@v should use local toolchain if not upgrading
! go run rsc.io/fortune/nonexist@v1
! stderr go1.22.9
! stderr switching
stderr '^go: rsc.io/fortune/nonexist@v1: module rsc.io/fortune@v1 found \(v1.0.0\), but does not contain package rsc.io/fortune/nonexist'
