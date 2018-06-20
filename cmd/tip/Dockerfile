FROM golang:1.9

RUN apt-get update && apt-get install --no-install-recommends -y -q build-essential git

# golang puts its go install here (weird but true)
ENV GOROOT_BOOTSTRAP /usr/local/go

# BEGIN deps (run `make update-deps` to update)

# Repo cloud.google.com/go at 1d0c2da (2018-01-30)
ENV REV=1d0c2da40456a9b47f5376165f275424acc15c09
RUN go get -d cloud.google.com/go/compute/metadata `#and 6 other pkgs` &&\
    (cd /go/src/cloud.google.com/go && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo github.com/golang/protobuf at 9255415 (2018-01-25)
ENV REV=925541529c1fa6821df4e44ce2723319eb2be768
RUN go get -d github.com/golang/protobuf/proto `#and 6 other pkgs` &&\
    (cd /go/src/github.com/golang/protobuf && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo github.com/googleapis/gax-go at 317e000 (2017-09-15)
ENV REV=317e0006254c44a0ac427cc52a0e083ff0b9622f
RUN go get -d github.com/googleapis/gax-go &&\
    (cd /go/src/github.com/googleapis/gax-go && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/build at e879390 (2018-02-01)
ENV REV=e8793909ba350594eea4c7c6bdb0f0d9a0d0f77a
RUN go get -d golang.org/x/build/autocertcache &&\
    (cd /go/src/golang.org/x/build && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/crypto at 1875d0a (2018-01-27)
ENV REV=1875d0a70c90e57f11972aefd42276df65e895b9
RUN go get -d golang.org/x/crypto/acme `#and 2 other pkgs` &&\
    (cd /go/src/golang.org/x/crypto && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/net at 6d90978 (2018-02-01)
ENV REV=6d90978dc4889d44e8cfbd04c05d17b5417823c7
RUN go get -d golang.org/x/net/context `#and 8 other pkgs` &&\
    (cd /go/src/golang.org/x/net && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/oauth2 at 30785a2 (2018-01-04)
ENV REV=30785a2c434e431ef7c507b54617d6a951d5f2b4
RUN go get -d golang.org/x/oauth2 `#and 5 other pkgs` &&\
    (cd /go/src/golang.org/x/oauth2 && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo golang.org/x/text at e19ae14 (2017-12-27)
ENV REV=e19ae1496984b1c655b8044a65c0300a3c878dd3
RUN go get -d golang.org/x/text/secure/bidirule `#and 4 other pkgs` &&\
    (cd /go/src/golang.org/x/text && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo google.golang.org/api at 7d0e2d3 (2018-01-30)
ENV REV=7d0e2d350555821bef5a5b8aecf0d12cc1def633
RUN go get -d google.golang.org/api/gensupport `#and 9 other pkgs` &&\
    (cd /go/src/google.golang.org/api && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo google.golang.org/genproto at 4eb30f4 (2018-01-25)
ENV REV=4eb30f4778eed4c258ba66527a0d4f9ec8a36c45
RUN go get -d google.golang.org/genproto/googleapis/api/annotations `#and 3 other pkgs` &&\
    (cd /go/src/google.golang.org/genproto && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Repo google.golang.org/grpc at 0bd008f (2018-01-25)
ENV REV=0bd008f5fadb62d228f12b18d016709e8139a7af
RUN go get -d google.golang.org/grpc `#and 23 other pkgs` &&\
    (cd /go/src/google.golang.org/grpc && (git cat-file -t $REV 2>/dev/null || git fetch -q origin $REV) && git reset --hard $REV)

# Optimization to speed up iterative development, not necessary for correctness:
RUN go install cloud.google.com/go/compute/metadata \
	cloud.google.com/go/iam \
	cloud.google.com/go/internal \
	cloud.google.com/go/internal/optional \
	cloud.google.com/go/internal/version \
	cloud.google.com/go/storage \
	github.com/golang/protobuf/proto \
	github.com/golang/protobuf/protoc-gen-go/descriptor \
	github.com/golang/protobuf/ptypes \
	github.com/golang/protobuf/ptypes/any \
	github.com/golang/protobuf/ptypes/duration \
	github.com/golang/protobuf/ptypes/timestamp \
	github.com/googleapis/gax-go \
	golang.org/x/build/autocertcache \
	golang.org/x/crypto/acme \
	golang.org/x/crypto/acme/autocert \
	golang.org/x/net/context \
	golang.org/x/net/context/ctxhttp \
	golang.org/x/net/http2 \
	golang.org/x/net/http2/hpack \
	golang.org/x/net/idna \
	golang.org/x/net/internal/timeseries \
	golang.org/x/net/lex/httplex \
	golang.org/x/net/trace \
	golang.org/x/oauth2 \
	golang.org/x/oauth2/google \
	golang.org/x/oauth2/internal \
	golang.org/x/oauth2/jws \
	golang.org/x/oauth2/jwt \
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
	google.golang.org/grpc/grpclb/grpc_lb_v1/messages \
	google.golang.org/grpc/grpclog \
	google.golang.org/grpc/internal \
	google.golang.org/grpc/keepalive \
	google.golang.org/grpc/metadata \
	google.golang.org/grpc/naming \
	google.golang.org/grpc/peer \
	google.golang.org/grpc/resolver \
	google.golang.org/grpc/resolver/dns \
	google.golang.org/grpc/resolver/passthrough \
	google.golang.org/grpc/stats \
	google.golang.org/grpc/status \
	google.golang.org/grpc/tap \
	google.golang.org/grpc/transport
# END deps.

# golang sets GOPATH=/go
ADD . /go/src/tip
WORKDIR /go/src/tip
RUN go install --tags=autocert
ENTRYPOINT ["/go/bin/tip"]

# We listen on 8080 (for historical reasons). The service.yaml maps public port 80 to 8080.
# We also listen on 443 for LetsEncrypt TLS.
EXPOSE 8080 443
