FROM golang:1.11

RUN apt-get update && apt-get install --no-install-recommends -y -q build-essential git

# For implicit GOCACHE (Issues 29243 and 29251), set HOME:
RUN mkdir -p /home/gopher
ENV HOME /home/gopher

# golang puts its go install here (weird but true)
ENV GOROOT_BOOTSTRAP /usr/local/go

# BEGIN deps (run `make update-deps` to update)

# Repo cloud.google.com/go at b5eca92 (2018-10-23)
ENV REV=b5eca92245a08e245bc29c4880c9779ea4aeaa9a
RUN go get -d cloud.google.com/go/compute/metadata `#and 7 other pkgs` &&\
    (cd /go/src/cloud.google.com/go && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo github.com/golang/protobuf at b4deda0 (2018-04-30)
ENV REV=b4deda0973fb4c70b50d226b1af49f3da59f5265
RUN go get -d github.com/golang/protobuf/proto `#and 6 other pkgs` &&\
    (cd /go/src/github.com/golang/protobuf && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo github.com/googleapis/gax-go at 317e000 (2017-09-15)
ENV REV=317e0006254c44a0ac427cc52a0e083ff0b9622f
RUN go get -d github.com/googleapis/gax-go &&\
    (cd /go/src/github.com/googleapis/gax-go && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo go.opencensus.io at ebd8d31 (2018-05-16)
ENV REV=ebd8d31470fedf6c27d0e3056653ddff642509b8
RUN go get -d go.opencensus.io/exporter/stackdriver/propagation `#and 12 other pkgs` &&\
    (cd /go/src/go.opencensus.io && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/build at 7b78c20 (2018-12-13)
ENV REV=7b78c2042368d5c56ee9dbd92ab5fa988c763944
RUN go get -d golang.org/x/build/autocertcache &&\
    (cd /go/src/golang.org/x/build && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/crypto at e4dc69e (2018-11-06)
ENV REV=e4dc69e5b2fd71dcaf8bd5d054eb936deb78d1fa
RUN go get -d golang.org/x/crypto/acme `#and 2 other pkgs` &&\
    (cd /go/src/golang.org/x/crypto && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/net at 891ebc4 (2018-12-13)
ENV REV=891ebc4b82d6e74f468c533b06f983c7be918a96
RUN go get -d golang.org/x/net/context `#and 8 other pkgs` &&\
    (cd /go/src/golang.org/x/net && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/oauth2 at f42d051 (2018-11-06)
ENV REV=f42d05182288abf10faef86d16c0d07b8d40ea2d
RUN go get -d golang.org/x/oauth2 `#and 5 other pkgs` &&\
    (cd /go/src/golang.org/x/oauth2 && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/sys at 4d1cda0 (2018-12-13)
ENV REV=4d1cda033e0619309c606fc686de3adcf599539e
RUN go get -d golang.org/x/sys/unix &&\
    (cd /go/src/golang.org/x/sys && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/text at 6f44c5a (2018-10-30)
ENV REV=6f44c5a2ea40ee3593d98cdcc905cc1fdaa660e2
RUN go get -d golang.org/x/text/secure/bidirule `#and 4 other pkgs` &&\
    (cd /go/src/golang.org/x/text && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo google.golang.org/api at 20530fd (2018-05-06)
ENV REV=20530fd5d65ad2caee87891f9896d7547cb400c9
RUN go get -d google.golang.org/api/gensupport `#and 9 other pkgs` &&\
    (cd /go/src/google.golang.org/api && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo google.golang.org/genproto at 86e600f (2018-04-27)
ENV REV=86e600f69ee4704c6efbf6a2a40a5c10700e76c2
RUN go get -d google.golang.org/genproto/googleapis/api/annotations `#and 4 other pkgs` &&\
    (cd /go/src/google.golang.org/genproto && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo google.golang.org/grpc at 07ef407 (2018-08-06)
ENV REV=07ef407d991f1004e6c3367c8f452ed9a02f17ff
RUN go get -d google.golang.org/grpc `#and 26 other pkgs` &&\
    (cd /go/src/google.golang.org/grpc && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Optimization to speed up iterative development, not necessary for correctness:
RUN go install cloud.google.com/go/compute/metadata \
	cloud.google.com/go/iam \
	cloud.google.com/go/internal \
	cloud.google.com/go/internal/optional \
	cloud.google.com/go/internal/trace \
	cloud.google.com/go/internal/version \
	cloud.google.com/go/storage \
	github.com/golang/protobuf/proto \
	github.com/golang/protobuf/protoc-gen-go/descriptor \
	github.com/golang/protobuf/ptypes \
	github.com/golang/protobuf/ptypes/any \
	github.com/golang/protobuf/ptypes/duration \
	github.com/golang/protobuf/ptypes/timestamp \
	github.com/googleapis/gax-go \
	go.opencensus.io/exporter/stackdriver/propagation \
	go.opencensus.io/internal \
	go.opencensus.io/internal/tagencoding \
	go.opencensus.io/plugin/ochttp \
	go.opencensus.io/plugin/ochttp/propagation/b3 \
	go.opencensus.io/stats \
	go.opencensus.io/stats/internal \
	go.opencensus.io/stats/view \
	go.opencensus.io/tag \
	go.opencensus.io/trace \
	go.opencensus.io/trace/internal \
	go.opencensus.io/trace/propagation \
	golang.org/x/build/autocertcache \
	golang.org/x/crypto/acme \
	golang.org/x/crypto/acme/autocert \
	golang.org/x/net/context \
	golang.org/x/net/context/ctxhttp \
	golang.org/x/net/http/httpguts \
	golang.org/x/net/http2 \
	golang.org/x/net/http2/hpack \
	golang.org/x/net/idna \
	golang.org/x/net/internal/timeseries \
	golang.org/x/net/trace \
	golang.org/x/oauth2 \
	golang.org/x/oauth2/google \
	golang.org/x/oauth2/internal \
	golang.org/x/oauth2/jws \
	golang.org/x/oauth2/jwt \
	golang.org/x/sys/unix \
	golang.org/x/text/secure/bidirule \
	golang.org/x/text/transform \
	golang.org/x/text/unicode/bidi \
	golang.org/x/text/unicode/norm \
	google.golang.org/api/gensupport \
	google.golang.org/api/googleapi \
	google.golang.org/api/googleapi/internal/uritemplates \
	google.golang.org/api/googleapi/transport \
	google.golang.org/api/internal \
	google.golang.org/api/iterator \
	google.golang.org/api/option \
	google.golang.org/api/storage/v1 \
	google.golang.org/api/transport/http \
	google.golang.org/genproto/googleapis/api/annotations \
	google.golang.org/genproto/googleapis/iam/v1 \
	google.golang.org/genproto/googleapis/rpc/code \
	google.golang.org/genproto/googleapis/rpc/status \
	google.golang.org/grpc \
	google.golang.org/grpc/balancer \
	google.golang.org/grpc/balancer/base \
	google.golang.org/grpc/balancer/roundrobin \
	google.golang.org/grpc/codes \
	google.golang.org/grpc/connectivity \
	google.golang.org/grpc/credentials \
	google.golang.org/grpc/encoding \
	google.golang.org/grpc/encoding/proto \
	google.golang.org/grpc/grpclog \
	google.golang.org/grpc/internal \
	google.golang.org/grpc/internal/backoff \
	google.golang.org/grpc/internal/channelz \
	google.golang.org/grpc/internal/envconfig \
	google.golang.org/grpc/internal/grpcrand \
	google.golang.org/grpc/internal/transport \
	google.golang.org/grpc/keepalive \
	google.golang.org/grpc/metadata \
	google.golang.org/grpc/naming \
	google.golang.org/grpc/peer \
	google.golang.org/grpc/resolver \
	google.golang.org/grpc/resolver/dns \
	google.golang.org/grpc/resolver/passthrough \
	google.golang.org/grpc/stats \
	google.golang.org/grpc/status \
	google.golang.org/grpc/tap
# END deps.

# golang sets GOPATH=/go
ADD . /go/src/tip
WORKDIR /go/src/tip
RUN go install --tags=autocert
ENTRYPOINT ["/go/bin/tip"]

# We listen on 8080 (for historical reasons). The service.yaml maps public port 80 to 8080.
# We also listen on 443 for LetsEncrypt TLS.
EXPOSE 8080 443
