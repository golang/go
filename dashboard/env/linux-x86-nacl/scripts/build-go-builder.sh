set -ex

export GOPATH=/gopath
export GOROOT=/goroot
PREFIX=/usr/local
: ${GO_REV:?"need to be set to the golang repo revision used to build the builder."}
: ${BUILDER_REV:?"need to be set to the go.tools repo revision for the builder."}

mkdir -p $GOROOT
curl -s https://storage.googleapis.com/gobuilder/go-snap.tar.gz | tar x --no-same-owner -zv -C $GOROOT
(cd $GOROOT/src && hg pull -r $GO_REV -u && find && ./make.bash)

GO_TOOLS=$GOPATH/src/code.google.com/p/go.tools
mkdir -p $GO_TOOLS
curl -s https://storage.googleapis.com/gobuilder/go.tools-snap.tar.gz | tar x --no-same-owner -zv -C $GO_TOOLS

mkdir -p $PREFIX/bin
(cd $GO_TOOLS && hg pull -r $BUILDER_REV -u && GOBIN=$PREFIX/bin /goroot/bin/go install code.google.com/p/go.tools/dashboard/builder)

rm -fR $GOROOT/bin $GOROOT/pkg $GOPATH

(cd /usr/local/bin && curl -s -O https://storage.googleapis.com/gobuilder/sel_ldr_x86_32 && chmod +x sel_ldr_x86_32)
(cd /usr/local/bin && curl -s -O https://storage.googleapis.com/gobuilder/sel_ldr_x86_64 && chmod +x sel_ldr_x86_64)

ln -s $GOROOT/misc/nacl/go_nacl_386_exec /usr/local/bin/
ln -s $GOROOT/misc/nacl/go_nacl_amd64p32_exec /usr/local/bin/
