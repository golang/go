env GO111MODULE=on
env GOTRACEBACK=single

# Go 1.21 work module should leave panicnil with an implicit default.
cp go.mod.21 go.mod
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
! stdout panicnil
stdout randautoseed=0

# Go 1.21 work module should NOT set panicnil=1 in Go 1.20 dependency.
cp go.mod.21 go.mod
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}' q
! stdout panicnil=1
! stdout randautoseed

go mod download rsc.io/panicnil # for go.sum
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}' rsc.io/panicnil
! stdout panicnil=1
! stdout randautoseed

# Go 1.20 work module should set panicnil=1.
cp go.mod.20 go.mod
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
stdout panicnil=1
stdout randautoseed=0

# Go 1.20 work module should set panicnil=1 in Go 1.20 dependency.
cp go.mod.20 go.mod
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}' q
stdout panicnil=1
! stdout randautoseed

# Go 1.21 workspace should leave panicnil with an implicit default.
cat q/go.mod
cp go.work.21 go.work
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
! stdout panicnil
stdout randautoseed=0
rm go.work

# Go 1.20 workspace with Go 1.21 module cannot happen.
cp go.work.20 go.work
cp go.mod.21 go.mod
! go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
stderr 'go: module . listed in go.work file requires go >= 1.21'
rm go.work

# Go 1.21 go.mod with godebug default=go1.20
rm go.work
cp go.mod.21 go.mod
go mod edit -godebug default=go1.20 -godebug asynctimerchan=0
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
stdout panicnil=1
stdout asynctimerchan=0

# Go 1.21 go.work with godebug default=go1.20
cp go.work.21 go.work
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
! stdout panicnil # go.work wins
stdout asynctimerchan=1 # go.work wins
go work edit -godebug default=go1.20 -godebug asynctimerchan=0
go list -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
stdout panicnil=1
stdout asynctimerchan=0
rm go.work

# Go 1.21 go.mod with //go:debug default=go1.20 in program
cp go.mod.21 go.mod
go list -tags godebug -f '{{.Module.GoVersion}} {{.DefaultGODEBUG}}'
stdout panicnil=1
stdout asynctimerchan=0

# Invalid //go:debug line should be diagnosed at build.
! go build -tags godebugbad
stderr 'invalid //go:debug: value contains space'

[short] skip

# Programs in Go 1.21 work module should trigger run-time error.
cp go.mod.21 go.mod
! go run .
stderr 'panic: panic called with nil argument'

! go run rsc.io/panicnil
stderr 'panic: panic called with nil argument'

# Programs in Go 1.20 work module use old panic nil behavior.
cp go.mod.20 go.mod
! go run .
stderr 'panic: nil'

! go run rsc.io/panicnil
stderr 'panic: nil'

# Programs in no module at all should use their go.mod file.
rm go.mod
! go run rsc.io/panicnil@v1.0.0
stderr 'panic: nil'

rm go.mod
! go run rsc.io/panicnil@v1.1.0
stderr 'panic: panic called with nil argument'

-- go.work.21 --
go 1.21
use .
use ./q

-- go.work.20 --
go 1.20
use .
use ./q

-- go.mod.21 --
go 1.21
module m
require q v1.0.0
replace q => ./q
require rsc.io/panicnil v1.0.0

-- go.mod.20 --
go 1.20
module m
require q v1.0.0
replace q => ./q
require rsc.io/panicnil v1.0.0

-- p.go --
//go:debug randautoseed=0

package main

func main() {
	panic(nil)
}

-- godebug.go --
//go:build godebug
//go:debug default=go1.20
//go:debug asynctimerchan=0

package main

-- godebugbad.go --
//go:build godebugbad
//go:debug default=go1.20 asynctimerchan=0

package main

-- q/go.mod --
go 1.20
module q

-- q/q.go --
package main
func main() {}
