# Builder
#########

FROM golang:1.11 AS build

RUN apt-get update && apt-get install -y \
      zip # required for generate-index.bash

# Check out the desired version of Go, both to build the godoc binary and serve
# as the goroot for content serving.
ARG GO_REF
RUN test -n "$GO_REF" # GO_REF is required.
RUN git clone --single-branch --depth=1 -b $GO_REF https://go.googlesource.com/go /goroot
RUN cd /goroot/src && ./make.bash

ENV GOROOT /goroot
ENV PATH=/goroot/bin:$PATH

RUN go version

RUN go get -v -d \
      golang.org/x/net/context \
      google.golang.org/appengine \
      cloud.google.com/go/datastore \
      golang.org/x/build \
      github.com/gomodule/redigo/redis

COPY . /go/src/golang.org/x/tools

WORKDIR /go/src/golang.org/x/tools/cmd/godoc
RUN GODOC_DOCSET=/goroot ./generate-index.bash

RUN go build -o /godoc -tags=golangorg golang.org/x/tools/cmd/godoc

# Clean up goroot for the final image.
RUN cd /goroot && git clean -xdf

# Add build metadata.
RUN cd /goroot && echo "go repo HEAD: $(git rev-parse HEAD)" >> /goroot/buildinfo
RUN echo "requested go ref: ${GO_REF}" >> /goroot/buildinfo
ARG TOOLS_HEAD
RUN echo "x/tools HEAD: ${TOOLS_HEAD}" >> /goroot/buildinfo
ARG TOOLS_CLEAN
RUN echo "x/tools clean: ${TOOLS_CLEAN}" >> /goroot/buildinfo
ARG DOCKER_TAG
RUN echo "image: ${DOCKER_TAG}" >> /goroot/buildinfo
ARG BUILD_ENV
RUN echo "build env: ${BUILD_ENV}" >> /goroot/buildinfo

RUN rm -rf /goroot/.git

# Final image
#############

FROM gcr.io/distroless/base

WORKDIR /app
COPY --from=build /godoc /app/
COPY --from=build /go/src/golang.org/x/tools/cmd/godoc/hg-git-mapping.bin /app/

COPY --from=build /goroot /goroot
ENV GOROOT /goroot

COPY --from=build /go/src/golang.org/x/tools/cmd/godoc/index.split.* /app/
ENV GODOC_INDEX_GLOB index.split.*

CMD ["/app/godoc"]
