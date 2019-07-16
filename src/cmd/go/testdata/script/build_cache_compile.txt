env GO111MODULE=off
[short] skip

# Set up fresh GOCACHE.
env GOCACHE=$WORK/gocache
mkdir $GOCACHE

# Building trivial non-main package should run compiler the first time.
go build -x lib.go
stderr '(compile|gccgo)( |\.exe).*lib\.go'

# ... but not again ...
go build -x lib.go
! stderr '(compile|gccgo)( |\.exe).*lib\.go'

# ... unless we use -a.
go build -a -x lib.go
stderr '(compile|gccgo)( |\.exe)'

-- lib.go --
package lib
