[short] skip
env GO111MODULE=on

# Download everything to avoid "finding" messages in stderr later.
cp go.mod.orig go.mod
go mod download
go mod download example.com@v1.0.0
go mod download example.com/badchain/a@v1.1.0
go mod download example.com/badchain/b@v1.1.0
go mod download example.com/badchain/c@v1.1.0

# Try to update example.com/badchain/a (and its dependencies).
! go get example.com/badchain/a
cmp stderr update-a-expected
cmp go.mod go.mod.orig

# Try to update the main module. This updates everything, including
# modules that aren't direct requirements, so the error stack is shorter.
go get -u ./...
cmp stderr update-main-expected
cmp go.mod go.mod.withc

# Update manually. Listing modules should produce an error.
cp go.mod.orig go.mod
go mod edit -require=example.com/badchain/a@v1.1.0
! go list -m all
cmp stderr list-expected

# Try listing a package that imports a package
# in a module without a requirement.
go mod edit -droprequire example.com/badchain/a
! go list -mod=mod m/use
cmp stderr list-missing-expected

! go list -mod=mod -test m/testuse
cmp stderr list-missing-test-expected

-- go.mod.orig --
module m

go 1.13

require example.com/badchain/a v1.0.0
-- go.mod.withc --
module m

go 1.13

require (
	example.com/badchain/a v1.0.0
	example.com/badchain/c v1.0.0
)
-- go.sum --
example.com/badchain/a v1.0.0 h1:iJDLiHLmpQgr9Zrv+44UqywAE2IG6WkHnH4uG08vf+s=
example.com/badchain/a v1.0.0/go.mod h1:6/gnCYHdVrs6mUgatUYUSbuHxEY+/yWedmTggLz23EI=
example.com/badchain/a v1.1.0 h1:cPxQpsOjaIrn05yDfl4dFFgGSbjYmytLqtIIBfTsEqA=
example.com/badchain/a v1.1.0/go.mod h1:T15b2BEK+RY7h7Lr2dgS38p1pgH5/t7Kf5nQXBlcW/A=
example.com/badchain/b v1.0.0 h1:kjDVlBxpjQavYxHE7ECCyyXhfwsfhWIqvghfRgPktSA=
example.com/badchain/b v1.0.0/go.mod h1:sYsH934pMc3/A2vQZh019qrWmp4+k87l3O0VFUYqL+I=
example.com/badchain/b v1.1.0 h1:iEALV+DRN62FArnYylBR4YwCALn/hCdITvhdagHa0L4=
example.com/badchain/b v1.1.0/go.mod h1:mlCgKO7lRZ+ijwMFIBFRPCGt5r5oqCcHdhSSE0VL4uY=
example.com/badchain/c v1.0.0 h1:lOeUHQKR7SboSH7Bj6eIDWoNHaDQXI0T2GfaH2x9fNA=
example.com/badchain/c v1.0.0/go.mod h1:4U3gzno17SaQ2koSVNxITu9r60CeLSgye9y4/5LnfOE=
example.com/badchain/c v1.1.0 h1:VtTg1g7fOutWKHQf+ag04KLRpdMGSfQ9s9tagVtGW14=
example.com/badchain/c v1.1.0/go.mod h1:tyoJj5qh+qtb48sflwdVvk4R+OjPQEY2UJOoibsVLPk=
-- use/use.go --
package use

import _ "example.com/badchain/c"
-- testuse/testuse.go --
package testuse
-- testuse/testuse_test.go --
package testuse

import (
	"testing"
	_ "example.com/badchain/c"
)

func Test(t *testing.T) {}
-- update-main-expected --
go: example.com/badchain/c@v1.1.0: parsing go.mod:
	module declares its path as: badchain.example.com/c
	        but was required as: example.com/badchain/c
	restoring example.com/badchain/c@v1.0.0
-- update-a-expected --
go: example.com/badchain/a@upgrade (v1.1.0) indirectly requires example.com/badchain/c@v1.1.0: parsing go.mod:
	module declares its path as: badchain.example.com/c
	        but was required as: example.com/badchain/c
-- list-expected --
go: example.com/badchain/a@v1.1.0 requires
	example.com/badchain/b@v1.1.0 requires
	example.com/badchain/c@v1.1.0: parsing go.mod:
	module declares its path as: badchain.example.com/c
	        but was required as: example.com/badchain/c
-- list-missing-expected --
go: finding module for package example.com/badchain/c
go: found example.com/badchain/c in example.com/badchain/c v1.1.0
go: m/use imports
	example.com/badchain/c: example.com/badchain/c@v1.1.0: parsing go.mod:
	module declares its path as: badchain.example.com/c
	        but was required as: example.com/badchain/c
-- list-missing-test-expected --
go: finding module for package example.com/badchain/c
go: found example.com/badchain/c in example.com/badchain/c v1.1.0
go: m/testuse tested by
	m/testuse.test imports
	example.com/badchain/c: example.com/badchain/c@v1.1.0: parsing go.mod:
	module declares its path as: badchain.example.com/c
	        but was required as: example.com/badchain/c
