

# docker buildx b . -o testdata
FROM golang:1.22 AS build

ARG GO_VERSION=go1.17.13

RUN set -ex; \
    go install golang.org/dl/${GO_VERSION}@latest; \
    ${GO_VERSION} download; \
    go install golang.org/x/vuln/cmd/govulncheck@latest;

WORKDIR /go/src/example.com/go117

COPY <<EOF main.go
package main
func main() {}
EOF

RUN set -ex; \
    GOTOOLCHAIN=${GO_VERSION} go mod init example.com/go117; \
    GOTOOLCHAIN=${GO_VERSION} go build

RUN set -ex; \
    export PATH=$(${GO_VERSION} env GOROOT)/bin:$PATH; \
    govulncheck -format openvex ./... > go117.vex.json

# ----

FROM scratch

COPY --from=build /go/src/example.com/go117/go117* /