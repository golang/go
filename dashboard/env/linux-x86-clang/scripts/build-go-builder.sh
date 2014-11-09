set -ex

export GOPATH=/gopath
export GOROOT=/goroot
PREFIX=/usr/local
: ${GO_REV:?"need to be set to the golang repo revision used to build the builder."}
: ${BUILDER_REV:?"need to be set to the go.tools repo revision for the builder."}

mkdir -p $GOROOT
curl -s https://storage.googleapis.com/gobuilder/go-snap.tar.gz | tar x --no-same-owner -zv -C $GOROOT
(cd $GOROOT/src && hg pull -r $GO_REV -u && find && ./make.bash)

GO_TOOLS=$GOPATH/src/golang.org/x/tools
mkdir -p $GO_TOOLS
curl -s https://storage.googleapis.com/gobuilder/go.tools-snap.tar.gz | tar x --no-same-owner -zv -C $GO_TOOLS

mkdir -p $PREFIX/bin
(cd $GO_TOOLS && hg pull -r $BUILDER_REV -u && GOBIN=$PREFIX/bin /goroot/bin/go install golang.org/x/tools/dashboard/builder)

rm -fR $GOROOT/bin $GOROOT/pkg $GOPATH
