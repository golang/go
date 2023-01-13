// This file is named go.fake.mod so it does not define a real module, which
// would make the contents of this directory unavailable to the test when run
// from outside the repository.

module αfake1α //@mark(αMarker, "αfake1α")

go 1.14

require golang.org/modfile v0.0.0 //@mark(βMarker, "require golang.org/modfile v0.0.0")
